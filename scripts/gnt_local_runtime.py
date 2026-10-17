"""Shared local GNT data preparation and optional inference-only optimizations."""
from types import MethodType
from time import perf_counter
import cv2
import numpy as np
import torch
from run_gnt_trial import ROOT, DATA, camera_vector
from tune_gnt import sources, balanced_targets
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


class Dataset:
    def __init__(self, name):
        self.name = name
        self.root = ROOT/DATA[name]
        self.k, self.poses, self.points = _read_colmap(self.root/'sparse/0')
        self.images = {}

    def image(self, index):
        if index not in self.images:
            image = cv2.imread(str(self.root/'images'/self.poses[index][0]))
            assert image is not None
            self.images[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.images[index]

    def case(self, index):
        name, r, t = self.poses[index]
        return dict(id=f'eval_{index:03}_z1.0', index=index, image=name, r=r.tolist(),
                    t=t.tolist(), k=list(self.k), kind='heldout', zoom=1.)

    def data(self, case, width, training=False):
        r, t = np.array(case['r']), np.array(case['t'])
        indices = sources(r, t, self.poses, self.points, self.k, 6 if training else 8,
                          'nearest', [1.]*len(self.poses), exclude=case['index'] if training else None)
        assert len(indices)>=2 and all(i%8 for i in indices)
        if training:
            assert case['index']%8 and case['index'] not in indices
        xyz = self.points@r.T+t; z=xyz[:,2]
        xy = xyz[:,:2]/np.maximum(z[:,None],.001)
        valid = (z>.01)&(np.abs(xy[:,0])<self.k[4]/(2*self.k[0]))&(np.abs(xy[:,1])<self.k[5]/(2*self.k[1]))
        assert valid.sum()>=2, ('Insufficient depth points', self.name, case['id'])
        near,far=np.quantile(z[valid],[.005,.995])*[.8,1.2];near=max(.02,float(near))
        assert far>near
        factor=width/self.k[4]
        tk=[float(v)*factor for v in case['k'][:4]]+[width,round(self.k[5]*factor)]
        sk=[float(v)*factor for v in self.k[:4]]+[width,round(self.k[5]*factor)] if training else self.k
        images=[cv2.resize(self.image(i),(sk[4],sk[5]),interpolation=cv2.INTER_AREA) if training and width!=self.k[4] else self.image(i) for i in indices]
        result=dict(camera=torch.tensor(camera_vector(r,t,tk))[None],rgb_path=[case['image']],
            depth_range=torch.tensor([[near,float(far)]],dtype=torch.float32),
            src_rgbs=torch.from_numpy(np.stack(images).astype(np.float32)/255)[None],
            src_cameras=torch.from_numpy(np.stack([camera_vector(self.poses[i][1],self.poses[i][2],sk) for i in indices]))[None])
        if training:
            result['rgb']=torch.from_numpy(cv2.resize(self.image(case['index']),(tk[4],tk[5]),interpolation=cv2.INTER_AREA).astype(np.float32)/255)[None]
        return result,indices


def load_model(trial, checkpoint):
    import sys
    sys.path.insert(0,str(trial/'GNT'))
    from config import config_parser
    from gnt.model import GNTModel
    options=config_parser().parse_args(['--trans_depth','8','--N_importance','0',
        '--rootdir',str(trial/'GNT'),'--expname','sfm_finalization','--no_reload'])
    model=GNTModel(options,load_opt=False,load_scheduler=False)
    load_weights(model,checkpoint)
    return model


def load_weights(model, checkpoint):
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.net_coarse.load_state_dict(weights['net_coarse'],strict=True)
    model.feature_net.load_state_dict(weights['feature_net'],strict=True)
    model.switch_to_eval()


def enable_sdpa(model):
    """Replace only qk ray attention; preserve view attention and weight format."""
    from gnt.transformer_network import Attention
    def forward(self,x,pos=None,ret_attn=False):
        if self.attn_mode!='qk' or ret_attn:
            return self._original_forward(x,pos,ret_attn)
        shape=(x.shape[0],x.shape[1],self.n_heads,-1)
        q=self.q_fc(x).view(shape).permute(0,2,1,3)
        k=self.k_fc(x).view(shape).permute(0,2,1,3)
        v=self.v_fc(x).view(shape).permute(0,2,1,3)
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v,
            dropout_p=self.dp.p if self.training else 0.)
        out=out.permute(0,2,1,3).contiguous().view(x.shape[0],x.shape[1],-1)
        return self.dp(self.out_fc(out))
    for layer in model.net_coarse.modules():
        if isinstance(layer,Attention) and not hasattr(layer,'_original_forward'):
            layer._original_forward=layer.forward
            layer.forward=MethodType(forward,layer)


def enable_compiled_view_attention(model):
    """Fuse the expensive view-attention operations without changing weight keys."""
    for layer in model.net_coarse.view_crosstrans:
        layer.attn.forward=torch.compile(layer.attn.forward,dynamic=True)


class Renderer:
    def __init__(self,model,optimized=False,chunk=4096,cache_features=False):
        from gnt.projection import Projector
        self.model=model;self.chunk=chunk;self.optimized=optimized
        self.projector=Projector('cuda:0');self.cache={} if cache_features else None

    @torch.inference_mode()
    def render(self,dataset,case,width):
        from gnt.sample_ray import RaySamplerSingleImage
        from gnt.render_image import render_single_image
        from gnt.render_ray import render_rays
        torch.cuda.synchronize();start=perf_counter()
        data,indices=dataset.data(case,width)
        sampler=RaySamplerSingleImage(data,'cuda:0');batch=sampler.get_all()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();prepared=perf_counter()
        key=(dataset.name,tuple(indices))
        if self.cache is not None and key in self.cache:
            features=tuple(f.to('cuda') if f is not None else None for f in self.cache[key])
            cache_hit=True
        else:
            features=self.model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2));cache_hit=False
            if self.cache is not None:
                # Cache per ordered source set on CPU, with a bounded working set.
                if len(self.cache)>=4:self.cache.pop(next(iter(self.cache)))
                self.cache[key]=tuple(f.cpu() if f is not None else None for f in features)
        torch.cuda.synchronize();featured=perf_counter()
        if self.optimized:
            count=len(batch['ray_o']);rgb=torch.empty((count,3),device='cuda')
            for i in range(0,count,self.chunk):
                shared={'camera','depth_range','src_rgbs','src_cameras'}
                part={k:v if k in shared or v is None else v[i:i+self.chunk] for k,v in batch.items()}
                result=render_rays(part,self.model,features,self.projector,64,
                    inv_uniform=True,N_importance=0,det=True,single_net=True)
                rgb[i:i+self.chunk]=result['outputs_coarse']['rgb']
            rgb=rgb.reshape(sampler.H,sampler.W,3).cpu().numpy()
        else:
            result=render_single_image(sampler,batch,self.model,self.projector,self.chunk,64,
                inv_uniform=True,N_importance=0,det=True,featmaps=features,single_net=True)
            rgb=result['outputs_coarse']['rgb'].numpy()
        torch.cuda.synchronize();finished=perf_counter()
        assert np.isfinite(rgb).all()
        return rgb,dict(sources=indices,width=width,height=sampler.H,seconds=finished-start,
            prepare_seconds=prepared-start,feature_seconds=featured-prepared,
            ray_seconds=finished-featured,peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20,
            feature_cache_hit=cache_hit,chunk=self.chunk)
