"""Evaluate a saved splat on every held-out view; optionally compare sensor depth."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--depth-root',type=Path)
    args=parser.parse_args()
    from nerfstudio.utils.eval_utils import eval_setup
    from torchvision.utils import save_image
    import cv2
    start=perf_counter()
    _,pipeline,loaded_path,step=eval_setup(args.config,test_mode='test')
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        pipeline.load_pipeline(state['pipeline'],state['step'])
        loaded_path,step=args.checkpoint,state['step']
        del state
    pipeline.eval()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    names=pipeline.datamanager.eval_dataset.image_filenames
    depth_paths={float(p.stem):p for p in args.depth_root.glob('*.png')} if args.depth_root else {}
    scale=float(pipeline.datamanager.train_dataparser_outputs.dataparser_scale)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index,(camera,batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            torch.cuda.synchronize();tick=perf_counter()
            outputs=pipeline.model.get_outputs_for_camera(camera=camera)
            torch.cuda.synchronize();seconds=perf_counter()-tick
            metrics,images=pipeline.model.get_image_metrics_and_images(outputs,batch)
            row={'image':Path(names[index]).name,**{k:float(metrics[k]) for k in ('psnr','ssim','lpips')},
                 'opacity_coverage':float((outputs['accumulation']>.1).float().mean()),'render_seconds':seconds}
            if args.depth_root:
                predicted=outputs['depth'].detach().cpu().numpy().squeeze()/scale
                measured=cv2.imread(str(depth_paths[float(Path(names[index]).stem)]),cv2.IMREAD_UNCHANGED).astype(float)/1000
                measured=cv2.resize(measured,(predicted.shape[1],predicted.shape[0]),interpolation=cv2.INTER_NEAREST)
                opacity=outputs['accumulation'].detach().cpu().numpy().squeeze()
                valid=(measured>.1)&(measured<8)&np.isfinite(predicted)&(opacity>.9)
                residual=predicted[valid]-measured[valid]
                row.update(depth_comparison_pixels=int(valid.sum()),
                    median_absolute_rendered_depth_residual_m=float(np.median(np.abs(residual))) if len(residual) else None,
                    rendered_depth_free_space_fraction=float((residual<-(.05+.03*measured[valid])).mean()) if len(residual) else None)
                # Six diagnosis targets only: keep disk use bounded while auditing geometry.
                targets={r['image'] for r in json.loads((Path(__file__).resolve().parents[1]/'outputs/scene_diagnosis/geometry.json').read_text())['views']}
                if row['image'] in targets:
                    np.savez_compressed(args.output/f'{index:03d}_depth.npz',rendered_m=predicted,measured_m=measured,opacity=opacity)
            rows.append(row)
            for key,image in images.items():
                save_image(image.detach().permute(2,0,1).cpu(),args.output/f'{index:03d}_{key}.png')
            print(json.dumps(row),flush=True)
    report={'checkpoint':str(loaded_path),'step':step,'gaussians':len(pipeline.model.gauss_params['means']),'views':rows,
            'mean':{k:float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','lpips','opacity_coverage','render_seconds')},
            'seconds':perf_counter()-start,'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/2**20}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    paths=sorted(args.output.glob('*_img.png'))
    if paths:
        panels=[]
        for i in np.linspace(0,len(paths)-1,min(8,len(paths)),dtype=int):
            picture=cv2.imread(str(paths[i]))
            panels.append(cv2.resize(picture,(960,round(picture.shape[0]*960/picture.shape[1]))))
        cv2.imwrite(str(args.output/'contact_sheet.jpg'),np.vstack(panels))
    print(json.dumps(report['mean'],indent=2))


if __name__=='__main__':main()
