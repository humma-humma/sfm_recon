"""Run the authors' pretrained GNT on our cameras, without per-scene training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import cv2
import numpy as np
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap

ROOT = Path(__file__).resolve().parents[1]
DATA = {'box':'outputs/splat_refresh_v1/colmap/box', 'boot':'outputs/stage2_boot_gaussian_colmap_half',
        'milk':'outputs/stage2_milk_gaussian_colmap_half', 'scene':'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced'}


def camera_vector(r, t, k):
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = k[:4]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3], pose[:3, 3] = r.T, -r.T@t
    return np.concatenate(([k[5], k[4]], intrinsic.ravel(), pose.ravel())).astype(np.float32)


def trial_cases(record, dataset):
    cases = record['cases']
    anchor = cases[2]['index'] if dataset != 'scene' else None
    return [c for c in cases if (c['kind'] == 'heldout' and (anchor is None or c['index'] == anchor))
            or c['id'] in ('motion_00', 'motion_04', 'motion_08')]


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--render-scratch', type=Path, required=True)
    parser.add_argument('--dataset', choices=list(DATA), required=True)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--chunk', type=int, default=2048)
    parser.add_argument('--checkpoint', type=Path, help='Optional adapted GNT weights; requires a separate output tag.')
    parser.add_argument('--output-tag', help='Subfolder for a candidate, preserving the original comparison.')
    parser.add_argument('--case-group', choices=['diagnostic','motion','full'], default='diagnostic')
    args = parser.parse_args()
    if args.checkpoint and not args.output_tag:
        parser.error('--checkpoint requires --output-tag to preserve the original trial')
    if args.output_tag and not all(c.isalnum() or c in '_-' for c in args.output_tag):
        parser.error('--output-tag must contain only letters, numbers, underscores or hyphens')
    sys.path.insert(0, str(args.trial/'GNT'))
    from config import config_parser
    from gnt.model import GNTModel
    from gnt.sample_ray import RaySamplerSingleImage
    from gnt.projection import Projector
    from gnt.render_image import render_single_image
    options = config_parser().parse_args(['--trans_depth','8','--N_importance','0','--rootdir',str(args.trial/'GNT'),
                                         '--expname','sfm_trial','--no_reload'])
    model = GNTModel(options, load_opt=False, load_scheduler=False)
    weights_path = args.checkpoint or args.trial/'GNT/generalization_200000.pth'
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.net_coarse.load_state_dict(checkpoint['net_coarse'], strict=True)
    model.feature_net.load_state_dict(checkpoint['feature_net'], strict=True)
    print(f'Loaded GNT checkpoint {weights_path} (strict); inference only.', flush=True)
    del checkpoint
    model.switch_to_eval()
    record = json.loads((args.render_scratch/'render_manifest.json').read_text())['datasets'][args.dataset]
    k, poses, points = _read_colmap(ROOT/DATA[args.dataset]/'sparse/0')
    cases = trial_cases(record,args.dataset)
    if args.case_group == 'diagnostic':
        cases = [c for c in cases if c['kind']=='heldout' and c['zoom']==1 and
                 (args.dataset!='scene' or c['image'].startswith(('64.','128.')))]
    elif args.case_group == 'motion': cases = [c for c in cases if c['kind']=='motion']
    if args.limit: cases = cases[:args.limit]
    folder = args.trial/'results/gnt'/args.dataset
    if args.output_tag: folder = folder/args.output_tag
    if args.width != 960: folder = folder/f'{args.case_group}_{args.width}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []; projector = Projector('cuda:0')
    start = perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for case in cases:
            name = case['id']; destination = folder/f'{name}.png'
            r, t = np.array(case['r']), np.array(case['t'])
            center = -r.T@t
            # Eight source photographs, within the authors' randomized training range.
            candidates = sorted((np.linalg.norm((-sr.T@st)-center)+.5*(1-np.dot(r[2],sr[2])), i)
                                for i, (_,sr,st) in enumerate(poses) if i%8 != 0 and np.dot(r[2],sr[2])>.5)
            indices = [i for _, i in candidates[:8]]
            assert len(indices) >= 2 and all(i%8 != 0 for i in indices)
            source_images, source_cameras = [], []
            for i in indices:
                source_name, sr, st = poses[i]
                image = cv2.cvtColor(cv2.imread(str(ROOT/DATA[args.dataset]/'images'/source_name)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
                source_images.append(image)
                source_cameras.append(camera_vector(sr, st, k))
            # Depth bounds come from visible reconstructed points, never held-out depth/RGB.
            camera_points = points@r.T+t
            z = camera_points[:, 2]
            valid = z > .01
            xy = camera_points[:, :2]/np.maximum(z[:, None], .001)
            valid &= (np.abs(xy[:, 0]) < k[4]/(2*k[0])) & (np.abs(xy[:, 1]) < k[5]/(2*k[1]))
            if valid.sum() < 10: raise RuntimeError('Insufficient points to set depth bounds')
            near, far = np.quantile(z[valid], [.005, .995])
            near, far = max(.02, float(near)*.8), float(far)*1.2
            factor = args.width/k[4]
            target_k = [float(x)*factor for x in case['k'][:4]]+[args.width, round(k[5]*factor)]
            data = dict(camera=torch.tensor(camera_vector(r,t,target_k))[None], rgb_path=[case['image']],
                depth_range=torch.tensor([[near,far]],dtype=torch.float32),
                src_rgbs=torch.from_numpy(np.stack(source_images))[None],
                src_cameras=torch.from_numpy(np.stack(source_cameras))[None])
            sampler = RaySamplerSingleImage(data, 'cuda:0')
            batch = sampler.get_all()
            torch.cuda.synchronize(); tick = perf_counter()
            features = model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2))
            result = render_single_image(sampler, batch, model, projector, args.chunk, N_samples=64,
                                         inv_uniform=True, N_importance=0, det=True, featmaps=features, single_net=True)
            torch.cuda.synchronize(); seconds = perf_counter()-tick
            rgb = result['outputs_coarse']['rgb'].numpy()
            assert np.isfinite(rgb).all() and rgb.shape == (target_k[5],target_k[4],3), (rgb.shape, int((~np.isfinite(rgb)).sum()))
            cv2.imwrite(str(destination),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR))
            row = dict(case, sources=indices, source_images=[poses[i][0] for i in indices],
                       depth_bounds=[near,far], seconds=seconds, width=target_k[4], height=target_k[5])
            rows.append(row)
            report = dict(method='GNT inference, eight-layer transformer, 64 inverse-depth samples, eight nearest eligible training views.',
                checkpoint_kind='user-supplied' if args.checkpoint else 'official pretrained',
                dataset=args.dataset, checkpoint_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                cases=rows, seconds=perf_counter()-start, peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20,
                trainable_parameters=sum(p.numel() for p in model.net_coarse.parameters())+sum(p.numel() for p in model.feature_net.parameters()),
                split_scope='Source RGB excludes interval-8 targets; camera/seed estimation predates rendering split.')
            (folder/'runtime.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(dataset=args.dataset,case=name,seconds=seconds,peak_cuda_mb=report['peak_cuda_mb'])),flush=True)
            del sampler,batch,features,result,data
    print('GNT complete',args.dataset,flush=True)


if __name__ == '__main__': main()
