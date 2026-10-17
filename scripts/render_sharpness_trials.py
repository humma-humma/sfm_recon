"""Render fixed splats and nearby training cameras into disposable scratch storage.

Run in the cached Nerfstudio Docker image. Never writes model parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    'box': 'outputs/splat_refresh_v1/colmap/box',
    'boot': 'outputs/stage2_boot_gaussian_colmap_half',
    'milk': 'outputs/stage2_milk_gaussian_colmap_half',
    'scene': 'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    selected = json.loads((ROOT/'configs/splatting/selected_models.json').read_text())['datasets']
    report = dict(method='Direct gsplat RGB+ED, fixed selected checkpoints, no optimization; black background.', datasets={})
    for dataset, choice in selected.items():
        folder = args.scratch/dataset
        folder.mkdir(exist_ok=True)
        checkpoint = ROOT/choice['presentation_checkpoint']
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        params = torch.load(checkpoint, map_location='cpu', weights_only=False)['pipeline']
        values = {key: params['_model.gauss_params.'+key].cuda() for key in
                  ('means', 'scales', 'quats', 'opacities', 'features_dc', 'features_rest')}
        del params
        transform = json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
        tr, scale = np.array(transform['transform']), transform['scale']
        k, poses, _ = _read_colmap(ROOT/DATA[dataset]/'sparse/0')
        train = [i for i in range(len(poses)) if i % 8 != 0]
        test = [i for i in range(len(poses)) if i % 8 == 0]
        if dataset == 'scene':
            names = {r['image'] for r in json.loads((ROOT/'outputs/scene_diagnosis/geometry.json').read_text())['views']}
            targets = [i for i, p in enumerate(poses) if p[0] in names]
            assert len(targets) == 6 and set(targets) <= set(test)
            anchor = next(i for i in targets if poses[i][0].startswith('128.'))
        else:
            targets = [test[0], test[len(test)//2], test[-1]]
            anchor = targets[1]
        colors = torch.cat((values['features_dc'][:, None], values['features_rest']), dim=1)
        degree = int(np.sqrt(colors.shape[1]))-1
        torch.cuda.reset_peak_memory_stats()
        def render(r, t, intrinsics, mode):
            rotation = r @ tr[:, :3].T
            view = np.eye(4)
            view[:3, :3] = rotation
            view[:3, 3] = scale*(t-rotation@tr[:, 3])
            fx, fy, cx, cy, width, height = intrinsics
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device='cuda', dtype=torch.float32)[None]
            torch.cuda.synchronize(); tick = perf_counter()
            with torch.no_grad():
                pixels, alpha, _ = rasterization(means=values['means'], quats=values['quats'],
                    scales=values['scales'].exp(), opacities=values['opacities'].sigmoid().flatten(),
                    colors=colors, viewmats=torch.tensor(view, device='cuda', dtype=torch.float32)[None],
                    Ks=K, width=width, height=height, sh_degree=degree, packed=False,
                    render_mode='RGB+ED', rasterize_mode=mode)
            torch.cuda.synchronize()
            seconds = perf_counter()-tick
            pixels = pixels[0].cpu().numpy()
            depth = pixels[..., 3]/scale
            depth[alpha[0, ..., 0].cpu().numpy() < .9] = 0
            return np.clip(pixels[..., :3], 0, 1), depth, seconds
        def save_rgb(path, rgb):
            cv2.imwrite(str(path), cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1)*255+.5), cv2.COLOR_RGB2BGR))
        cases = []
        for index in targets:
            name, r, t = poses[index]
            for zoom in (1., 1.7):
                cases.append(dict(id=f'eval_{index:03d}_z{zoom}', index=index, image=name,
                                  r=r.tolist(), t=t.tolist(), zoom=zoom, kind='heldout'))
        name, r, t = poses[anchor]
        # Nine views along a 4 cm lateral path, in original COLMAP world units.
        center = -r.T@t
        for frame, offset in enumerate(np.linspace(-.02, .02, 9)):
            moved = center+r[0]*offset
            cases.append(dict(id=f'motion_{frame:02d}', index=anchor, image=name,
                              r=r.tolist(), t=(-r@moved).tolist(), zoom=1., kind='motion', lateral_offset_m=float(offset)))
        source_indices = set()
        for case in cases:
            r, t = np.array(case['r']), np.array(case['t'])
            center = -r.T@t
            # Camera distance plus angular penalty, using training cameras only.
            scores = [(np.linalg.norm((-sr.T@st)-center)+.5*(1-np.dot(r[2], sr[2])), i)
                      for i in train for _, sr, st in [poses[i]] if np.dot(r[2], sr[2]) > .5]
            nearest = sorted(scores)[:4]
            case['sources'] = [i for _, i in nearest]
            case['weights'] = [float(np.exp(-score/max(nearest[0][0], .05))) for score, _ in nearest]
            source_indices.update(case['sources'])
        sources = {}
        for index in sorted(source_indices):
            name, r, t = poses[index]
            _, depth, _ = render(r, t, k, 'classic')
            np.save(folder/f'source_{index:03d}_depth.npy', depth)
            source_path = ROOT/DATA[dataset]/'images'/name
            image = cv2.imread(str(source_path))
            if image is None or image.shape[:2] != (k[5], k[4]):
                raise ValueError((source_path, None if image is None else image.shape, k))
            cv2.imwrite(str(folder/f'source_{index:03d}.png'), image)
            sources[str(index)] = dict(image=name, r=r.tolist(), t=t.tolist())
        for case in cases:
            intrinsics = list(k)
            intrinsics[0] *= case['zoom']; intrinsics[1] *= case['zoom']
            r, t = np.array(case['r']), np.array(case['t'])
            for mode in ('classic', 'antialiased'):
                rgb, depth, seconds = render(r, t, intrinsics, mode)
                save_rgb(folder/f'{case["id"]}_{mode}.png', rgb)
                case[mode+'_seconds'] = seconds
                if mode == 'classic': np.save(folder/f'{case["id"]}_depth.npy', depth)
            if case['kind'] == 'heldout':
                gt = cv2.imread(str(ROOT/DATA[dataset]/'images'/case['image']))
                if case['zoom'] != 1:
                    yy, xx = np.indices(gt.shape[:2], dtype=np.float32)
                    gt = cv2.remap(gt, (xx-k[2])/case['zoom']+k[2], (yy-k[3])/case['zoom']+k[3], cv2.INTER_LINEAR)
                cv2.imwrite(str(folder/f'{case["id"]}_gt.png'), gt)
            case['k'] = intrinsics
            print(dataset, case['id'], flush=True)
        assert all(i in train for c in cases for i in c['sources'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
        report['datasets'][dataset] = dict(k=k, cases=cases, sources=sources, checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=checksum, data=DATA[dataset], heldout_indices=test, train_indices=train,
            peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20)
        (args.scratch/'render_manifest.json').write_text(json.dumps(report, indent=2))
        del values, colors
        torch.cuda.empty_cache()
    print('Render preparation complete', flush=True)


if __name__ == '__main__': main()
