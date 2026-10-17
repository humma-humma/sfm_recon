"""Read-only RGB-D consistency and seed-support diagnosis of the scene export.

Depth is treated as aligned optical-axis depth in millimetres, as in stage3.py.
Residuals test that assumption jointly with poses; they do not isolate its cause.
"""
from pathlib import Path
import json
import cv2
import numpy as np

from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced'
RAW = ROOT.parent / 'Experiments/Stage_3_Data/stage3'
OUT = ROOT / 'outputs/scene_diagnosis'


def project(points, rotation, translation, k):
    camera = points @ rotation.T + translation
    z = camera[:, 2]
    uv = camera[:, :2] / np.maximum(z[:, None], 1e-9)
    uv = uv * np.array(k[:2]) + np.array(k[2:4])
    return uv, z


def cross_view(depth, source_pose, target_pose, target_depth, k):
    """Project valid source pixels; distinguish agreement from occlusion."""
    yy, xx = np.indices(depth.shape)
    valid = (depth > .1) & (depth < 8)
    # Exclude depth discontinuities to reduce mixed foreground/background pixels.
    high = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    low = cv2.erode(depth, np.ones((3, 3), np.uint8))
    valid &= (high-low) < (.05 + .03*depth)
    x, y, z = xx[valid], yy[valid], depth[valid]
    camera = np.column_stack(((x-k[2])*z/k[0], (y-k[3])*z/k[1], z))
    r, t = source_pose
    world = (camera-t) @ r
    uv, projected_z = project(world, *target_pose, k)
    ij = np.rint(uv).astype(int)
    h, w = target_depth.shape
    inside = (projected_z > .1) & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    source_xy = np.column_stack((x, y))[inside]
    ij, projected_z = ij[inside], projected_z[inside]
    observed = target_depth[ij[:, 1], ij[:, 0]]
    good = (observed > .1) & (observed < 8)
    target_high = cv2.dilate(target_depth, np.ones((3, 3), np.uint8))
    target_low = cv2.erode(target_depth, np.ones((3, 3), np.uint8))
    good &= (target_high[ij[:, 1], ij[:, 0]]-target_low[ij[:, 1], ij[:, 0]]) < (.05+.03*observed)
    ij, source_xy = ij[good], source_xy[good]
    delta = projected_z[good]-observed[good]
    tolerance = .05+.03*observed[good]
    agree = np.abs(delta) <= tolerance
    # Positive residual can be legitimate occlusion. Negative is free-space conflict.
    stats = {'source_valid_interior_pixels': int(valid.sum()), 'compared_pixels': len(delta),
             'agreement_fraction': float(agree.mean()) if len(delta) else None,
             'free_space_conflict_fraction': float((delta < -tolerance).mean()) if len(delta) else None,
             'occluded_fraction': float((delta > tolerance).mean()) if len(delta) else None,
             'median_absolute_depth_residual_m': float(np.median(np.abs(delta))) if len(delta) else None}
    return stats, source_xy, ij, agree, delta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    k, poses, seeds = _read_colmap(EXPORT / 'sparse/0')
    raw_k = np.array(json.loads((RAW/'camera_parameters.json').read_text())['intrinsics'])
    raw_depths = {float(p.stem): p for p in (RAW/'depth').glob('*.png')}
    raw_rgbs = {float(p.stem): p for p in (RAW/'rgb').glob('*.png')}
    # Half of export resolution: quarter of raw RGB resolution.
    small_k = tuple(v*.5 for v in k[:4])
    cache = {}
    feature_cache = {}
    def frame(i):
        if i not in cache:
            name = poses[i][0]
            timestamp = float(Path(name).stem)
            depth_raw = cv2.imread(str(raw_depths[timestamp]), cv2.IMREAD_UNCHANGED)
            rgb_raw = cv2.imread(str(raw_rgbs[timestamp]))
            if depth_raw is None or rgb_raw is None:
                raise ValueError(name)
            depth = cv2.resize(depth_raw.astype(np.float32)/1000, (480, 270), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb_raw, (480, 270), interpolation=cv2.INTER_AREA)
            cache[i] = (depth, rgb, depth_raw.shape, rgb_raw.shape)
        return cache[i]
    def features(i):
        if i not in feature_cache:
            feature_cache[i] = cv2.SIFT_create(nfeatures=1800).detectAndCompute(cv2.cvtColor(frame(i)[1],cv2.COLOR_BGR2GRAY),None)
        return feature_cache[i]
    def feature_check(i, j):
        a, da = features(i)
        b, db = features(j)
        if da is None or db is None:
            return {'matches':0}
        matches = [m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance < .7*n.distance]
        if len(matches)<8:
            return {'matches':len(matches)}
        x = np.array([a[m.queryIdx].pt for m in matches])
        y = np.array([b[m.trainIdx].pt for m in matches])
        # Reject descriptor mismatches independently of the exported pose.
        _, inliers = cv2.findFundamentalMat(x,y,cv2.FM_RANSAC,1.5,.999)
        if inliers is None:
            return {'matches':len(matches)}
        keep = inliers.ravel().astype(bool)
        x,y=x[keep],y[keep]
        d=frame(i)[0][np.rint(x[:,1]).astype(int),np.rint(x[:,0]).astype(int)]
        keep=(d>.1)&(d<8)
        x,y,d=x[keep],y[keep],d[keep]
        camera=np.column_stack(((x[:,0]-small_k[2])*d/small_k[0],(x[:,1]-small_k[3])*d/small_k[1],d))
        _,r,t=poses[i]
        prediction,_=project((camera-t)@r,*poses[j][1:],small_k)
        errors=np.linalg.norm(prediction-y,axis=1)*2 # report export960 pixels
        relative_r=poses[j][1]@r.T
        relative_t=poses[j][2]-relative_r@t
        tx,ty,tz=relative_t
        skew=np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])
        matrix=np.array([[small_k[0],0,small_k[2]],[0,small_k[1],small_k[3]],[0,0,1]])
        inv=np.linalg.inv(matrix)
        fundamental=inv.T@skew@relative_r@inv
        lines=np.column_stack((x,np.ones(len(x))))@fundamental.T
        distance=np.abs((lines*np.column_stack((y,np.ones(len(y))))).sum(axis=1))/np.maximum(np.linalg.norm(lines[:,:2],axis=1),1e-12)*2
        return {'matches':len(matches),'depth_valid_epipolar_inliers':len(errors),
                'median_pose_depth_reprojection_error_export_pixels':float(np.median(errors)) if len(errors) else None,
                'median_export_pose_epipolar_distance_pixels':float(np.median(distance)) if len(distance) else None,
                'p90_pose_depth_reprojection_error_export_pixels':float(np.percentile(errors,90)) if len(errors) else None}
    targets = [4.568511, 51.501833, 64.301844, 98.435166, 128.301833, 145.368511]
    heldout_names = {r['image'] for r in json.loads((ROOT/'outputs/splat_surroundings_tests/evaluation/scene_control/metrics.json').read_text())['views']}
    rows, selections = [], []
    for target in targets:
        i = min(range(len(poses)), key=lambda j: abs(float(Path(poses[j][0]).stem)-target))
        name, r, t = poses[i]
        depth, rgb, depth_shape, rgb_shape = frame(i)
        original_rgb=cv2.imread(str(raw_rgbs[float(Path(name).stem)]))
        exported_rgb=cv2.imread(str(EXPORT/'images'/name))
        resized_rgb=cv2.resize(original_rgb,(960,540),interpolation=cv2.INTER_AREA)
        export_rgb_mae=float(np.abs(resized_rgb.astype(float)-exported_rgb.astype(float)).mean())
        neighbors = [j for j in range(max(0, i-4), min(len(poses), i+5)) if poses[j][0] not in heldout_names]
        support = np.zeros(depth.shape, dtype=np.uint8)
        pairs = []
        for j in neighbors:
            other_depth, other_rgb, _, _ = frame(j)
            stats, xy, ij, agree, delta = cross_view(depth, (r,t), poses[j][1:], other_depth, small_k)
            support[xy[agree, 1], xy[agree, 0]] += 1
            if agree.any():
                stats['rgb_mae_on_depth_agreement_0_255'] = float(np.abs(rgb[xy[agree,1],xy[agree,0]].astype(float)-other_rgb[ij[agree,1],ij[agree,0]]).mean())
            stats.update(image=poses[j][0], time_gap_s=abs(float(Path(name).stem)-float(Path(poses[j][0]).stem)),
                         camera_baseline_m=float(np.linalg.norm(-r.T@t + poses[j][1].T@poses[j][2])),
                         rotation_degrees=float(np.degrees(np.arccos(np.clip((np.trace(r@poses[j][1].T)-1)/2,-1,1)))))
            pairs.append(stats)
        for j in sorted(neighbors,key=lambda j:abs(j-i))[:2]:
            pairs[neighbors.index(j)]['independent_feature_check']=feature_check(i,j)
        uv, z = project(seeds, r, t, small_k)
        ij = np.rint(uv).astype(int)
        inside = (z>.1)&(ij[:,0]>=0)&(ij[:,0]<480)&(ij[:,1]>=0)&(ij[:,1]<270)
        ij, z = ij[inside], z[inside]
        measured = depth[ij[:,1],ij[:,0]]
        valid_seed = (measured>.1)&(measured<8)
        surface = valid_seed & (np.abs(z-measured)<.05+.03*measured)
        grid = np.zeros((9,16),np.uint8)
        grid[np.minimum(8,ij[surface,1]//30), np.minimum(15,ij[surface,0]//30)] = 1
        valid = (depth>.1)&(depth<8)
        record = {'image':name, 'held_out':name in heldout_names, 'raw_depth_shape':depth_shape, 'raw_rgb_shape':rgb_shape,
                  'valid_depth_fraction':float(valid.mean()), 'valid_depth_supported_by_2_nearby_train_views_fraction':float((support[valid]>=2).mean()),
                  'export_vs_resized_raw_rgb_mae_0_255':export_rgb_mae,
                  'seeds_in_frustum':int(inside.sum()), 'seeds_agreeing_with_depth':int(surface.sum()),
                  'surface_seed_tile_coverage_16x9':float(grid.mean()), 'pairs':pairs}
        rows.append(record)
        selections.extend([{'image':poses[j][0], 'split':'train'} for j in neighbors])
        selections.append({'image':name,'split':'eval'})
        color = cv2.applyColorMap(np.uint8(np.minimum(depth,8)/8*255),cv2.COLORMAP_TURBO)
        color[~valid]=0
        heat = cv2.applyColorMap(np.uint8(support/max(1,len(neighbors))*255),cv2.COLORMAP_VIRIDIS)
        panel = np.hstack((rgb,color,heat))
        cv2.imwrite(str(OUT/f'{Path(name).stem}.jpg'),panel)
        print(json.dumps({k:v for k,v in record.items() if k!='pairs'}),flush=True)
    expected = [raw_k[0,0]*.5,raw_k[1,1]*.5,raw_k[0,2]*.5,raw_k[1,2]*.5]
    report = {'assumptions':['Depth is aligned to RGB, optical-axis Z, units millimetres; consistency tests this jointly with poses.',
                              'Support uses up to eight temporal training neighbors, not all cameras.',
                              'Depth agreement tolerance 5cm + 3%; 480x270 nearest depth; excludes local depth discontinuities.',
                              'Seed tile coverage is coarse projected surface support, not reconstructed surface completeness.'],
              'intrinsics_max_error_pixels':float(np.max(np.abs(np.array(k[:4])-expected))),
              'export_cameras':len(poses),'seed_points':len(seeds),'views':rows}
    report['all_export_rgb_timestamps_found']=all(float(Path(p[0]).stem) in raw_rgbs for p in poses)
    report['all_export_depth_timestamps_found']=all(float(Path(p[0]).stem) in raw_depths for p in poses)
    (OUT/'geometry.json').write_text(json.dumps(report,indent=2))
    unique = {(s['split'],s['image']):s for s in selections}
    (OUT/'render_selection.json').write_text(json.dumps(list(unique.values()),indent=2))
    html = '<html><body><h1>Scene diagnosis</h1><p>Source | measured depth (0–8m) | agreement with nearby training cameras (brighter = more support).</p>'
    for row in rows:
        html += f'<h2>{row["image"]}</h2><img width="1440" src="{Path(row["image"]).stem}.jpg"><pre>{json.dumps({k:v for k,v in row.items() if k!="pairs"},indent=2)}</pre>'
    (OUT/'index.html').write_text(html+'</body></html>',encoding='utf-8')


if __name__ == '__main__':
    main()
