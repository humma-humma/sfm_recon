"""Prepare train-only depth confidence and an independent RGB-D TSDF baseline."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from sfm_reconstruction.depth_supervision import reliable_depth
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap
from diagnose_scene_reconstruction import cross_view

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/depth_research'
EXPORT=ROOT/'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced'
RAW=ROOT.parent/'Experiments/Stage_3_Data/stage3'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mesh-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    k,poses,_=_read_colmap(EXPORT/'sparse/0')
    small_k=tuple(v*.5 for v in k[:4])
    evaluation=json.loads((ROOT/'outputs/splat_surroundings_tests/evaluation/scene_control/metrics.json').read_text())
    heldout={r['image'] for r in evaluation['views']}
    train=[i for i,p in enumerate(poses) if p[0] not in heldout]
    depths={float(p.stem):p for p in (RAW/'depth').glob('*.png')}
    rgbs={float(p.stem):p for p in (RAW/'rgb').glob('*.png')}
    def load(i):
        raw=cv2.imread(str(depths[float(Path(poses[i][0]).stem)]),cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape!=(1080,1920): raise ValueError(poses[i][0])
        return cv2.resize(raw.astype(np.float32)/1000,(480,270),interpolation=cv2.INTER_NEAREST)
    start=perf_counter()
    cache={i:load(i) for i in train}
    if not args.mesh_only:
        rows=[]
        folder=OUT/'depth_cache'
        folder.mkdir(exist_ok=True)
        for index,i in enumerate(train):
            depth=cache[i]
            neighbors=[j for j in train if 0<abs(j-i)<=3]
            support=np.zeros(depth.shape,np.float32)
            for j in neighbors:
                _,xy,_,agree,_=cross_view(depth,poses[i][1:],poses[j][1:],cache[j],small_k)
                support[xy[agree,1],xy[agree,0]]+=1
            confidence=np.where((support>=2)&reliable_depth(depth),support/max(1,len(neighbors)),0).astype(np.float32)
            # Full exported resolution, keeping actual sensor measurements for targets.
            original=cv2.imread(str(depths[float(Path(poses[i][0]).stem)]),cv2.IMREAD_UNCHANGED)
            full=cv2.resize(original.astype(np.float32)/1000,(960,540),interpolation=cv2.INTER_NEAREST)
            weight=cv2.resize(confidence,(960,540),interpolation=cv2.INTER_NEAREST)*reliable_depth(full)
            np.savez_compressed(folder/f'{Path(poses[i][0]).stem}.npz',depth=full,weight=weight)
            rows.append({'image':poses[i][0],'neighbors':[poses[j][0] for j in neighbors],
                         'supervised_image_fraction':float((weight>0).mean()),'mean_weight':float(weight.mean())})
            if index%30==0: print(f'Confidence {index+1}/{len(train)}',flush=True)
        source=ROOT/'outputs/splat_surroundings_tests/train/scene_control/splatfacto/run_5000'
        plan={'training_views':len(train),'heldout_views':len(heldout),'depth_weight':.1,'additional_steps':5000,
              'source_checkpoint':(source/'nerfstudio_models/step-000004999.ckpt').relative_to(ROOT).as_posix(),
              'source_checkpoint_sha256':hashlib.sha256((source/'nerfstudio_models/step-000004999.ckpt').read_bytes()).hexdigest(),
              'source_config_sha256':hashlib.sha256((source/'config.yml').read_bytes()).hexdigest(),
              'pose_export_sha256':hashlib.sha256((EXPORT/'sparse/0/images.txt').read_bytes()).hexdigest(),
              'rows':rows,'heldout_depth_used_for_training':False,'selected_models_unchanged':True,
              'exclusion_scope':'New depth loss, confidence cache and TSDF only. Inherited 5k initialization uses the existing sequence seed cloud and was not rebuilt with a training-only split.',
              'method':'5cm+3% cross-view agreement with >=2 temporal training neighbors; reject sensor depth discontinuities; robust relative Huber loss'}
        (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    import open3d as o3d
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.025,sdf_trunc=.075,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intrinsic=o3d.camera.PinholeCameraIntrinsic(480,270,*small_k)
    for index,i in enumerate(train):
        depth=cache[i].copy()
        with np.load(OUT/'depth_cache'/f'{Path(poses[i][0]).stem}.npz') as data:
            weight=cv2.resize(data['weight'],(480,270),interpolation=cv2.INTER_NEAREST)
        depth[weight<=0]=0
        color=cv2.imread(str(rgbs[float(Path(poses[i][0]).stem)]))
        color=cv2.cvtColor(cv2.resize(color,(480,270),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color),o3d.geometry.Image(depth),depth_scale=1,depth_trunc=8,convert_rgb_to_intensity=False)
        extrinsic=np.eye(4);extrinsic[:3,:3]=poses[i][1];extrinsic[:3,3]=poses[i][2]
        volume.integrate(rgbd,intrinsic,extrinsic)
        if index%40==0: print(f'TSDF {index+1}/{len(train)}',flush=True)
    mesh=volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(OUT/'scene_tsdf.ply'),mesh)
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    yy,xx=np.indices((270,480))
    directions=np.stack(((xx-small_k[2])/small_k[0],(yy-small_k[3])/small_k[1],np.ones_like(xx)),axis=-1)
    metrics=[]
    target_names={v['image'] for v in json.loads((ROOT/'outputs/scene_diagnosis/geometry.json').read_text())['views']}
    for i,(name,r,t) in enumerate(poses):
        if name not in target_names: continue
        rays=np.concatenate((np.broadcast_to(-r.T@t,directions.shape),directions@r),axis=-1).astype(np.float32)
        prediction=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        measured=load(i) # Held-out depth is accessed only for evaluation, after mesh extraction.
        valid=reliable_depth(measured)
        hits=np.isfinite(prediction)&(prediction>0)
        mask=valid&hits
        error=np.abs(prediction[mask]-measured[mask])
        metrics.append({'image':name,'valid_sensor_pixels':int(valid.sum()),'compared_pixels':int(mask.sum()),
                        'sensor_surface_coverage':float(mask.sum()/max(1,valid.sum())),
                        'median_absolute_depth_error_m':float(np.median(error)) if len(error) else None})
        np.savez_compressed(OUT/f'tsdf_{Path(name).stem}.npz',rendered_m=prediction,measured_m=measured)
    report={'vertices':len(mesh.vertices),'triangles':len(mesh.triangles),'voxel_m':.025,'truncation_m':.075,
            'train_views':len(train),'heldout_views_excluded':len(heldout),'views':metrics,'seconds':perf_counter()-start,
            'caveat':'Confidence-masked sensor fusion using estimated COLMAP poses; not BundleFusion reproduction. Coverage must accompany depth error.'}
    (OUT/'tsdf_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
