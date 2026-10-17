"""Train/evaluate the official 2DGS implementation with our fixed camera split.

Only data loading, logging and evaluation are adapted. The upstream optimizer,
surfel rasterizer, densification and geometry losses execute unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
import cv2
import numpy as np
import torch
from run_gnt_trial import DATA, ROOT, trial_cases
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--render-scratch', type=Path, required=True)
    parser.add_argument('--dataset', choices=list(DATA), required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lambda-dist', type=float, default=None,
                        help='Default: 1000 for bounded objects; upstream general-scene default 0 for scene.')
    args = parser.parse_args()
    default_dist=0 if args.dataset=='scene' else 1000
    weight=default_dist if args.lambda_dist is None else args.lambda_dist
    if not math.isfinite(weight) or weight<0:parser.error('--lambda-dist must be finite and nonnegative')
    variant=f'_dist_{weight:g}' if weight!=default_dist else ''
    sys.path.insert(0,str(args.trial/'2dgs'))
    from arguments import ModelParams, PipelineParams, OptimizationParams
    import scene
    from scene import dataset_readers as readers
    from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_points3D_text
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    from gaussian_renderer import render
    import train
    source = ROOT/DATA[args.dataset]
    k, poses, _ = _read_colmap(source/'sparse/0')
    order = {name:i for i,(name,_,_) in enumerate(poses)}
    model_folder = args.trial/'models/2dgs'/args.dataset/f'steps_{args.steps}{variant}'
    model_folder.mkdir(parents=True,exist_ok=True)
    signature = {file:hashlib.sha256((source/'sparse/0'/file).read_bytes()).hexdigest() for file in ('cameras.txt','images.txt','points3D.txt')}
    def load_scene(path, images, evaluate):
        # Avoid upstream timestamp truncation and lexical reordering of scene frames.
        extrinsics = read_extrinsics_text(source/'sparse/0/images.txt')
        intrinsics = read_intrinsics_text(source/'sparse/0/cameras.txt')
        cameras = readers.readColmapCameras(extrinsics,intrinsics,str(source/'images'))
        cameras = [c._replace(image_name=Path(c.image_path).stem) for c in cameras]
        cameras.sort(key=lambda c:order[Path(c.image_path).name])
        training = [c for i,c in enumerate(cameras) if i%8 != 0]
        testing = [c for i,c in enumerate(cameras) if i%8 == 0]
        assert {c.image_name for c in training}.isdisjoint(c.image_name for c in testing)
        xyz, rgb, _ = read_points3D_text(source/'sparse/0/points3D.txt')
        seed = model_folder/'seed.ply'
        if not seed.exists(): readers.storePly(str(seed),xyz,rgb)
        cloud = readers.BasicPointCloud(points=xyz,colors=rgb.astype(np.float32)/255,normals=np.zeros_like(xyz))
        return readers.SceneInfo(cloud,training,testing,readers.getNerfppNorm(training),str(seed))
    scene.sceneLoadTypeCallbacks['Colmap'] = load_scene
    parameter_parser = argparse.ArgumentParser()
    lp,op,pp = ModelParams(parameter_parser),OptimizationParams(parameter_parser),PipelineParams(parameter_parser)
    options = parameter_parser.parse_args(['--source_path',str(source),'--model_path',str(model_folder),
        '--resolution','1','--data_device','cpu','--eval','--iterations',str(args.steps),
        '--lambda_dist',str(weight),
        '--lambda_normal','0.05'])
    dataset,opt,pipe = lp.extract(options),op.extract(options),pp.extract(options)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    train.TENSORBOARD_FOUND=False
    train.network_gui.try_connect=lambda *unused:None
    # Retain upstream reporting but keep bulky TensorBoard images out of the pilot.
    torch.cuda.reset_peak_memory_stats();tick=perf_counter()
    train.training(dataset,opt,pipe,[],[args.steps],[],None)
    torch.cuda.synchronize();training_seconds=perf_counter()-tick
    peak=torch.cuda.max_memory_allocated()/2**20
    model=scene.GaussianModel(options.sh_degree)
    model.load_ply(str(model_folder/f'point_cloud/iteration_{args.steps}/point_cloud.ply'))
    assert all(torch.isfinite(value).all() for value in (model.get_xyz,model.get_opacity,model.get_scaling,model.get_rotation,model.get_features))
    background=torch.zeros(3,device='cuda')
    record=json.loads((args.render_scratch/'render_manifest.json').read_text())['datasets'][args.dataset]
    folder=args.trial/'results/2dgs'/args.dataset
    if variant:folder=folder/variant.lstrip('_')
    if args.steps != 10000: folder=folder/f'steps_{args.steps}'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.no_grad():
        for case in trial_cases(record,args.dataset):
            r,t=np.array(case['r']),np.array(case['t'])
            fx,fy,_,_,width,height=case['k']
            fovx,fovy=2*math.atan(width/(2*fx)),2*math.atan(height/(2*fy))
            view=torch.tensor(getWorld2View2(r.T,t)).T.cuda()
            projection=getProjectionMatrix(.01,100,fovx,fovy).T.cuda()
            camera=MiniCam(width,height,fovy,fovx,.01,100,view,view@projection)
            torch.cuda.synchronize();start=perf_counter()
            output=render(camera,model,pipe,background)
            torch.cuda.synchronize();seconds=perf_counter()-start
            rgb=output['render'].permute(1,2,0).clamp(0,1).cpu().numpy()
            assert rgb.shape==(height,width,3) and np.isfinite(rgb).all()
            cv2.imwrite(str(folder/f'{case["id"]}.png'),cv2.cvtColor(np.uint8(rgb*255+.5),cv2.COLOR_RGB2BGR))
            rows.append(dict(case,seconds=seconds))
    for file,expected in signature.items():assert hashlib.sha256((source/'sparse/0'/file).read_bytes()).hexdigest()==expected
    report=dict(method='Official 2DGS training and rasterizer; fixed COLMAP poses; full 960x540 exported photographs; CPU camera-image cache.',
        dataset=args.dataset,iterations=args.steps,training_seconds=training_seconds,peak_cuda_mb=peak,
        gaussians=len(model.get_xyz),lambda_dist=opt.lambda_dist,lambda_normal=opt.lambda_normal,
        distortion_active_after_step=3000,normal_active_after_step=7000,
        train_views=len([i for i in order.values() if i%8!=0]),heldout_views=len([i for i in order.values() if i%8==0]),
        input_sha256=signature,cases=rows,
        split_scope='RGB training excludes interval-8 targets. Existing camera/seed geometry predates split. Ten-thousand-step pilot, not a claim of full 30k convergence.')
    (folder/'runtime.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key not in ('cases','input_sha256')}),flush=True)


if __name__=='__main__':main()
