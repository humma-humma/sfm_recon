"""Export selected checkpoints and degree-3 SH for the WebGL preview.

Run with the project's torch/numpy/scipy/Pillow Python environment. Source models
are never modified. Camera positions use the saved dataparser transform, which
already includes COLMAP's applied world convention transform.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'assets'
SOURCES = {
    'box': 'outputs/splat_refresh_v1/colmap/box',
    'boot': 'outputs/stage2_boot_gaussian_colmap_half',
    'milk': 'outputs/stage2_milk_gaussian_colmap_half',
    'scene': 'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced',
}

def cameras(folder, transform, scale):
    intrinsics = {}
    for line in (folder / 'cameras.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        p = line.split()
        # These selected datasets use PINHOLE intrinsics.
        if p[1] != 'PINHOLE': raise ValueError(p[1])
        intrinsics[int(p[0])] = float(np.degrees(2*np.arctan(float(p[3])/(2*float(p[5])))))
    lines = iter((folder / 'images.txt').read_text().splitlines())
    result = []
    for line in lines:
        if not line or line.startswith('#'): continue
        p = line.split(); next(lines)  # COLMAP's alternating observation row
        q = np.array(list(map(float, p[1:5])))
        r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        c = -r.T @ np.array(list(map(float,p[5:8])))
        pos = scale*(transform[:,:3] @ c + transform[:,3])
        direction = transform[:,:3] @ r.T[:,2]
        up = transform[:,:3] @ -r.T[:,1]
        result.append(dict(name=p[9], position=pos.tolist(), forward=direction.tolist(), up=up.tolist(), fov=intrinsics[int(p[8])]))
    return result

def main():
    OUT.mkdir(exist_ok=True)
    selection = json.loads((ROOT/'configs/splatting/selected_models.json').read_text())
    manifest = {}
    for name, choice in selection['datasets'].items():
        checkpoint = ROOT/choice['presentation_checkpoint']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['pipeline']
        def field(key): return state['_model.gauss_params.'+key].detach().numpy()
        means, scales, quats = field('means'), field('scales'), field('quats')
        opacity = 1/(1+np.exp(-np.clip(field('opacities'), -80,80)))
        colors = np.clip(.5 + .28209479177387814*field('features_dc'), 0,1)
        coefficients = np.concatenate([field('features_dc')[:,None,:],field('features_rest')],axis=1)
        if coefficients.shape[1:] != (16,3): raise ValueError('Expected degree-3 RGB spherical harmonics')
        valid = np.isfinite(np.concatenate([means,scales,quats,opacity,colors,coefficients.reshape(len(means),-1)],axis=1)).all(axis=1) & (np.linalg.norm(quats,axis=1)>0)
        harmonics = np.zeros((int(valid.sum()),16,4),dtype='<f2')
        harmonics[:,:,:3] = coefficients[valid]
        if not np.isfinite(harmonics).all(): raise ValueError('SH coefficients overflow float16')
        means, scales, quats, opacity, colors = [a[valid] for a in (means,scales,quats,opacity,colors)]
        rotation = Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
        variance = np.exp(2*scales.astype(np.float64))
        covariance = np.einsum('nik,nk,njk->nij',rotation,variance,rotation)
        # Four vec4 records: center+opacity, color+pad, symmetric covariance+pad.
        data = np.zeros((len(means),16),dtype='<f4')
        data[:,:3]=means; data[:,3]=opacity[:,0]; data[:,4:7]=colors
        data[:,8:14]=covariance[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
        if not np.isfinite(data).all(): raise ValueError('Nonfinite exported covariance')
        data.tofile(OUT/f'{name}.bin')
        harmonics.tofile(OUT/f'{name}.sh.bin')
        trans = json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
        views = cameras(ROOT/SOURCES[name]/'sparse/0',np.array(trans['transform']),trans['scale'])
        eval_name = 'box_new' if name=='box' else name+'_old'
        image = Image.open(ROOT/f'outputs/splat_refresh_v1/evaluation/{eval_name}/000_img.png')
        image.crop((image.width//2,0,image.width,image.height)).convert('RGB').save(OUT/f'{name}.jpg',quality=90)
        manifest[name]=dict(count=len(data), bytes=data.nbytes, sh_bytes=harmonics.nbytes, sh_degree=3,
            sh_format='16 RGB/pad vec4 coefficients per Gaussian, little-endian float16',
            removed=int((~valid).sum()), checkpoint=choice['presentation_checkpoint'], views=views,
            color='Full degree-3 view-dependent color; SH coefficients quantized to float16',
            format='16 little-endian float32 per Gaussian; center/opacity, RGB/pad, covariance xx xy xz yy yz zz/pad/pad')
        print(f'{name}: {len(data):,} Gaussians, {(data.nbytes+harmonics.nbytes)/1e6:.1f} MB including SH, {len(views)} cameras',flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__ == '__main__': main()
