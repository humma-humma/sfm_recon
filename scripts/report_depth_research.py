"""Compare the matched 10k control, measured-depth candidate and TSDF baseline."""
from pathlib import Path
import html
import json
import cv2
import numpy as np
from sfm_reconstruction.depth_supervision import reliable_depth
from sfm_reconstruction.splat_cleanup_evaluation import compare_candidate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/depth_research'


def main():
    candidate=json.loads((OUT/'evaluation/metrics.json').read_text())
    reference=json.loads((ROOT/'outputs/scene_control_continuation/evaluation/scene_10000/metrics.json').read_text())
    gate=compare_candidate(reference,candidate)
    baseline_indices={r['image']:i for i,r in enumerate(reference['views'])}
    targets={r['image'] for r in json.loads((ROOT/'outputs/scene_diagnosis/geometry.json').read_text())['views']}
    settings=json.loads((ROOT/'configs/splatting/depth_research.json').read_text())
    folder=ROOT/'web_demo/assets/research'
    folder.mkdir(parents=True,exist_ok=True)
    gallery=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Depth research — Reconstruction Atlas</title>',
             '<style>body{background:#101211;color:#edf1eb;font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1500px;line-height:1.6}a{color:#bdd994}img{max-width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #444}section{margin:32px 0}</style></head><body>',
             '<a href="index.html">← Explore four reconstructions</a><h1>Measured-depth reconstruction experiment</h1>',
             '<p>A controlled adaptation of uncertainty-aware depth-prior ideas. Both splat models start from the same fresh 5k checkpoint and receive 5k additional iterations. Poses, images, initialization and resolution schedule are fixed. Only the candidate adds confidence-weighted sensor depth. All 35 held-out RGB and depth frames are excluded from the added loss, confidence cache and TSDF. The inherited sequence seed cloud was not rebuilt with a training-only split: this is a matched continuation ablation, not a strictly held-out end-to-end reconstruction benchmark.</p>',
             '<p><a href="https://barbararoessle.github.io/dense_depth_priors_nerf/">Dense Depth Priors</a> inspires the depth/confidence constraint; this is not its NeRF model or learned depth completion. The independent TSDF mesh uses estimated poses and training RGB-D; it is not a BundleFusion reproduction.</p>',
             '<p>Left to right below: source photograph / RGB-only 10k control / depth-supervised 10k candidate.</p>']
    gallery.append('<h2>Sensor surface baseline</h2><p>Measured depth | train-only TSDF raycast. Both use a fixed 0–4 m color scale; black indicates invalid sensor depth or no mesh hit. Compare coverage as well as surface accuracy.</p>')
    for name in sorted(targets,key=lambda n:float(Path(n).stem)):
        with np.load(OUT/f'tsdf_{Path(name).stem}.npz') as d:
            measured,mesh=d['measured_m'],d['rendered_m']
        panels=[]
        for depth in (measured,mesh):
            valid=np.isfinite(depth)&(depth>.1)&(depth<8)
            color=cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(depth,nan=0,posinf=0)/4,0,1)*255),cv2.COLORMAP_TURBO)
            color[~valid]=0;panels.append(color)
        file=f'tsdf_{Path(name).stem}.jpg'
        cv2.imwrite(str(folder/file),np.hstack(panels))
        gallery.append(f'<p>{html.escape(name)}</p><img loading="lazy" src="assets/research/{file}" alt="Measured and fused mesh depth">')
    gallery.append('<h2>All held-out RGB comparisons</h2>')
    rows=[]
    for i,row in enumerate(candidate['views']):
        name=row['image'];j=baseline_indices[name]
        before=cv2.imread(str(ROOT/f'outputs/scene_control_continuation/evaluation/scene_10000/{j:03d}_img.png'))
        after=cv2.imread(str(OUT/f'evaluation/{i:03d}_img.png'))
        assert before.shape==after.shape==(540,1920,3)
        assert np.array_equal(before[:,:960],after[:,:960])
        composite=cv2.resize(np.hstack((before[:,:960],before[:,960:],after[:,960:])),(1440,270),interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder/f'{i:03d}.jpg'),composite)
        record={'image':name,'psnr_delta':row['psnr']-reference['views'][j]['psnr']}
        if name in targets:
            with np.load(OUT/f'evaluation/{i:03d}_depth.npz') as d:
                pred,measured,alpha=(d[k] for k in ('rendered_m','measured_m','opacity'))
            with np.load(ROOT/f'outputs/scene_diagnosis/render_10000/eval_{Path(name).stem}_depth.npz') as d:
                base,base_alpha=d['rendered_m'],d['opacity']
            mask=reliable_depth(measured)&np.isfinite(pred)&np.isfinite(base)&(alpha>.9)&(base_alpha>.9)
            record.update(common_depth_pixels=int(mask.sum()),
                          control_depth_error_m=float(np.median(np.abs(base[mask]-measured[mask]))),
                          candidate_depth_error_m=float(np.median(np.abs(pred[mask]-measured[mask]))))
            with np.load(OUT/f'tsdf_{Path(name).stem}.npz') as d:
                mesh=d['rendered_m'];sensor=d['measured_m']
            small_mask=cv2.resize(mask.astype(np.uint8),(480,270),interpolation=cv2.INTER_NEAREST).astype(bool)&np.isfinite(mesh)&reliable_depth(sensor)
            sm_base=cv2.resize(base,(480,270),interpolation=cv2.INTER_NEAREST)
            sm_pred=cv2.resize(pred,(480,270),interpolation=cv2.INTER_NEAREST)
            record['common_three_method_depth']={
                'pixels':int(small_mask.sum()),'tsdf_sensor_coverage':float((np.isfinite(mesh)&reliable_depth(sensor)).sum()/max(1,reliable_depth(sensor).sum())),
                'control_error_m':float(np.median(np.abs(sm_base[small_mask]-sensor[small_mask]))),
                'candidate_error_m':float(np.median(np.abs(sm_pred[small_mask]-sensor[small_mask]))),
                'tsdf_error_m':float(np.median(np.abs(mesh[small_mask]-sensor[small_mask])))}
        rows.append(record)
        gallery.append(f'<section><h2>{html.escape(name)}</h2><p>PSNR {reference["views"][j]["psnr"]:.2f} → {row["psnr"]:.2f} dB; LPIPS {reference["views"][j]["lpips"]:.3f} → {row["lpips"]:.3f} (lower is better).</p><a href="assets/research/{i:03d}.jpg"><img loading="lazy" src="assets/research/{i:03d}.jpg" alt="Source, RGB-only control, and measured-depth candidate"></a></section>')
    depth_rows=[r for r in rows if 'common_depth_pixels' in r]
    baseline_median=float(np.median([r['control_depth_error_m'] for r in depth_rows]))
    candidate_median=float(np.median([r['candidate_depth_error_m'] for r in depth_rows]))
    pavement=next(r for r in depth_rows if r['image']=='64.3018440.jpg')
    geometry_checks={'median_depth_error':candidate_median<=settings['geometry_gate']['six_view_median_depth_error_ratio_max']*baseline_median,
                     'pavement_preserved':pavement['candidate_depth_error_m']<=pavement['control_depth_error_m']+settings['geometry_gate']['pavement_depth_error_increase_m_max']}
    runtime=json.loads((OUT/'training_runtime.json').read_text())
    tsdf=json.loads((OUT/'tsdf_metrics.json').read_text())
    summary={'status':'complete','against_matched_rgb_control':gate,'geometry_checks':geometry_checks,
             'six_view_median_of_per_view_depth_errors_m':{'control':baseline_median,'candidate':candidate_median},
             'candidate_mean':candidate['mean'],'control_mean':reference['mean'],'runtime':runtime,'tsdf':tsdf,
             'views':rows,'selected_models_changed':False,
             'limitations':['Single exploratory run and preselected diagnostic regions; no statistical significance claim.',
                            'Inherited initialization/poses use existing sequence reconstruction, not a train-only rebuilt seed cloud. Holdout exclusion applies to added supervision, confidence and TSDF.',
                            'Expected splat depth is not a surface-thickness measurement.',
                            'Sensor RGB/depth calibration remains assumed; unreliable pixels excluded.',
                            'TSDF error must be interpreted with its visible coverage.']}
    review_path=OUT/'visual_review.json'
    if review_path.exists():
        summary['visual_review']=json.loads(review_path.read_text())
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    (folder/'results.json').write_text(json.dumps(summary,indent=2))
    introduction=f'<p><strong>RGB gate: {gate["status"]}. Geometry checks: {geometry_checks}.</strong> Selected presentation models remain unchanged.</p><p>Mean PSNR change: {gate["mean_deltas"]["psnr"]:+.3f} dB; worst view: {gate["worst_view_psnr_delta"]:+.3f} dB. Median of six per-view depth errors: {baseline_median:.3f} → {candidate_median:.3f} m.</p><p><a href="assets/research/results.json">Full results and limitations</a></p>'
    gallery.insert(6,introduction)
    gallery.append('</body></html>')
    (ROOT/'web_demo/research.html').write_text('\n'.join(gallery),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="../../web_demo/research.html">Open research comparison</a>',encoding='utf-8')
    lines=['# Measured-depth research results','',
           'A single controlled continuation experiment, adapting confidence-aware depth supervision to Splatfacto. This is not a paper reproduction.','',
           f'RGB gate versus matched 10k control: **{gate["status"]}**. Selected models unchanged.','',
           '| Metric | RGB-only 10k | Depth-supervised 10k |','|---|---:|---:|']
    for metric in ('psnr','ssim','lpips'):
        lines.append(f'| {metric} | {reference["mean"][metric]:.6f} | {candidate["mean"][metric]:.6f} |')
    lines+=['',f'Worst held-out PSNR delta: {gate["worst_view_psnr_delta"]:+.3f} dB.',
            f'Training: {runtime["seconds"]:.1f} seconds; peak allocated CUDA memory {runtime["peak_cuda_allocated_mb"]:.1f} MiB. Shared GPU workload was left running; this is not an isolated speed benchmark.','',
            '| Diagnostic image | Control depth error (m) | Candidate depth error (m) |','|---|---:|---:|']
    lines += [f'| {r["image"]} | {r["control_depth_error_m"]:.4f} | {r["candidate_depth_error_m"]:.4f} |' for r in depth_rows]
    lines+=['','Depth errors use identical valid sensor/interior/opacity masks for control and candidate. The JSON also compares all three methods on a common intersection mask and reports TSDF coverage.',
            '',f'Geometry checks: `{geometry_checks}`. TSDF has {tsdf["vertices"]:,} vertices and {tsdf["triangles"]:,} triangles; the stair entrance has only {tsdf["views"][3]["sensor_surface_coverage"]:.1%} sensor-surface coverage.',
            '', '## Reproduction and claims','',
            'Run `scripts/run_depth_research.ps1` with the local Open3D/PyTorch Python environment. Prepared depth uses training frames and their training neighbors only; all 35 held-out depths are evaluation-only. Both splat runs start from the same original 5k checkpoint and receive 5k further steps with identical poses, Gaussian initialization and resolution schedule. The sole training change is the measured-depth loss. The reference is the already completed 10k RGB-only continuation.',
            '', 'The loss is robust relative Huber with weight 0.1, using measured optical-axis depth in metres. Confidence requires agreement within 5cm + 3% in at least two nearby training cameras, and excludes depth discontinuities. These are heuristic reliability weights, not calibrated learned uncertainty. Configuration and pre-evaluation gates are in `configs/splatting/depth_research.json`.',
            '', 'The project drive filled during the first attempt to save an intermediate checkpoint. The incomplete output was removed and training reran from the original source. Only final inference weights are saved; this candidate checkpoint cannot resume Adam optimization. Originals were untouched.',
            '', 'The confidence-masked TSDF uses 2.5cm voxels and 7.5cm truncation, estimated poses, and training RGB-D only. It supplies a geometric baseline, not a complete/collision-ready scene or a BundleFusion reproduction.',
            '', 'Paper inspiration: [Dense Depth Priors, CVPR 2022](https://barbararoessle.github.io/dense_depth_priors_nerf/). Alignment and consistent fusion context: [BundleFusion, TOG 2017](https://graphics.stanford.edu/projects/bundlefusion/). 2DGS and learned-prior integrations remain future work; neither is claimed as implemented.',
            '', 'Visual results: `web_demo/research.html`. Numeric and runtime artifacts: `outputs/depth_research/results.json`. Geometry: `outputs/depth_research/scene_tsdf.ply`. Sensor alignment/units remain assumptions; expected depth is not thickness; these are exploratory dataset-specific results.']
    if 'visual_review' in summary:
        lines+=['','## Sampled visual review','',summary['visual_review']['findings'],
                '', 'Review scope: selected diagnostics, pavement control and worst-regression views; not every novel camera. Keep the selected presentation checkpoints unchanged. Numeric gate status retains its standard visual-review-pending label; the separate review record documents this sampled assessment.']
    lines+=['','## Split scope','', 'The inherited 5k initialization comes from the existing full-sequence estimated dense cloud and was not rebuilt using training views only. Pose estimation also precedes the rendering split. Held-out exclusion is verified for the added loss, its confidence neighbors and the TSDF integration; it is not an end-to-end split-isolation claim. Shared initialization keeps the continuation comparison controlled, but these results must not be presented as a strictly held-out reconstruction benchmark.']
    (ROOT/'docs/DEPTH_RESEARCH_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rgb_gate':gate,'geometry_checks':geometry_checks,'depth_control':baseline_median,'depth_candidate':candidate_median},indent=2))


if __name__=='__main__':main()
