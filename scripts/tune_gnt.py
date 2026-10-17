"""Bounded local GNT ablations and adaptation, preserving the original trial.

All interval-8 images are excluded from adaptation targets and source RGB.
Development views select settings; separate cameras check transfer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from run_gnt_trial import ROOT, DATA, camera_vector
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


CONFIGS = {
    'control': dict(samples=64, views=8, selection='nearest', inverse=True, bounds='original'),
    'samples128': dict(samples=128),
    'samples192': dict(samples=192),
    'views6': dict(views=6),
    'views12': dict(views=12),
    'overlap8': dict(selection='overlap'),
    'linear64': dict(inverse=False),
    'wide_bounds': dict(bounds='wide'),
}


def visible(points, r, t, k):
    xyz = points @ r.T + t
    z = xyz[:, 2]
    uv = xyz[:, :2] / np.maximum(z[:, None], 1e-6)
    uv = uv * np.array(k[:2]) + np.array(k[2:4])
    return (z > .01) & (uv[:, 0] >= 0) & (uv[:, 0] < k[4]) & (uv[:, 1] >= 0) & (uv[:, 1] < k[5])


def sources(r, t, poses, points, k, count, mode, sharpness, exclude=None):
    center = -r.T @ t
    ranked = sorted((np.linalg.norm((-sr.T@st)-center)+.5*(1-np.dot(r[2],sr[2])), i)
                    for i, (_,sr,st) in enumerate(poses) if i%8 != 0 and i!=exclude and np.dot(r[2],sr[2])>.5)
    if mode == 'nearest':
        return [i for _, i in ranked[:count]]
    candidates = [i for _, i in ranked[:24]]
    target_points = points[visible(points, r, t, k)]
    if len(target_points) > 4096:
        target_points = target_points[np.linspace(0, len(target_points)-1, 4096, dtype=int)]
    scale = max(np.percentile([sharpness[i] for i in candidates], 90), 1e-6)
    masks = {i: visible(target_points, poses[i][1], poses[i][2], k) for i in candidates}
    covered = np.zeros(len(target_points), dtype=bool)
    selected = []
    while candidates and len(selected) < count:
        def score(i):
            overlap = masks[i].mean() if len(target_points) else 0
            novel = (masks[i] & ~covered).mean() if len(target_points) else 0
            return .65*overlap + .2*novel + .1*min(sharpness[i]/scale, 1) + .05*np.dot(r[2], poses[i][1][2])
        best = max(candidates, key=score)
        candidates.remove(best)
        selected.append(best)
        covered |= masks[best]
    return selected


def balanced_targets(count):
    """Six equal capture intervals; interval-8 RGB remains excluded everywhere."""
    return [[int(i) for i in group if i % 8 != 0]
            for group in np.array_split(np.arange(count), 6)]


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--render-scratch', type=Path, required=True)
    parser.add_argument('--phase', choices=['sweep', 'adapt', 'final', 'balanced', 'balanced-final', 'detail', 'detail-final', 'traversal-bench', 'traversal-precision', 'traversal'], required=True)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--traversal-width', type=int, default=960)
    parser.add_argument('--traversal-limit', type=int)
    parser.add_argument('--detail-arm', choices=['control','patch'], default='control')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--resume-step', type=int, default=0)
    parser.add_argument('--train-width', type=int, default=480)
    args = parser.parse_args()
    sys.path.insert(0, str(args.trial/'GNT'))
    from config import config_parser
    from gnt.model import GNTModel
    from gnt.sample_ray import RaySamplerSingleImage
    from gnt.sample_ray import rng as ray_rng
    from gnt.projection import Projector
    from gnt.render_image import render_single_image
    from gnt.render_ray import render_rays
    torch.manual_seed(2026)
    np.random.seed(2026)
    ray_rng.seed(2026)
    torch.set_num_threads(4)
    options = config_parser().parse_args(['--trans_depth', '8', '--N_importance', '0',
        '--rootdir', str(args.trial/'GNT'), '--expname', 'sfm_tuning', '--no_reload'])
    model = GNTModel(options, load_opt=False, load_scheduler=False)
    weights = args.trial/'GNT/generalization_200000.pth'
    original_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    def load(path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.net_coarse.load_state_dict(checkpoint['net_coarse'], strict=True)
        model.feature_net.load_state_dict(checkpoint['feature_net'], strict=True)
        model.switch_to_eval()
    load(weights)
    projector = Projector('cuda:0')
    output = args.trial/'gnt_tuning'
    output.mkdir(exist_ok=True)
    manifest = json.loads((args.render_scratch/'render_manifest.json').read_text())['datasets']
    report_path = output/'results.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        checkpoint_sha256=original_hash, seed=2026, runs=[], adaptation={},
        scope='Development uses previously inspected scene cameras 120/240. Confirmation cameras 96/184/272 were not used to select inference settings. All interval-8 RGB is excluded from adaptation and sources. Camera estimation predates the split. This is an exploratory pilot, not a pristine benchmark.')
    def save_report():
        report_path.write_text(json.dumps(report, indent=2))
    def choose_inference():
        dev_ids = [c['id'] for c in development]
        candidates = [r for r in report['runs'] if r['width']==480 and r['case'] in dev_ids and r['label'] in list(CONFIGS)+['combined']]
        labels = sorted(set(r['label'] for r in candidates))
        scores = {label:float(np.mean([r['psnr'] for r in candidates if r['label']==label])) for label in labels}
        costs = {label:float(np.mean([r['seconds'] for r in candidates if r['label']==label])) for label in labels}
        best_score = max(scores.values())
        winner = min((label for label in labels if scores[label]>=best_score-.05),key=costs.get)
        report['inference_selection'] = dict(label=winner,
            config=next(r['config'] for r in candidates if r['label']==winner),development_psnr=scores,
            policy='Lowest measured render time within 0.05 dB of the best development mean; heuristic cost tolerance, not a statistical significance threshold.')
        save_report()
    cache = {}
    def dataset_data(dataset):
        if dataset not in cache:
            k, poses, points = _read_colmap(ROOT/DATA[dataset]/'sparse/0')
            images = [cv2.cvtColor(cv2.imread(str(ROOT/DATA[dataset]/'images'/name)), cv2.COLOR_BGR2RGB) for name,_,_ in poses]
            sharp = [cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for im in images]
            cache[dataset] = k, poses, points, images, sharp
        return cache[dataset]
    def make_data(dataset, case, config, width, target_rgb=False):
        k, poses, points, images, sharp = dataset_data(dataset)
        r, t = np.array(case['r']), np.array(case['t'])
        indices = sources(r,t,poses,points,k,config['views'],config['selection'],sharp,
                          exclude=case['index'] if target_rgb else None)
        assert len(indices) >= 2 and all(i%8 != 0 for i in indices)
        xyz = points@r.T+t
        valid = visible(points,r,t,k)
        if config['bounds']=='original':
            # Match the original trial's centered-frustum calculation exactly.
            z = xyz[:, 2]
            xy = xyz[:, :2]/np.maximum(z[:, None], .001)
            valid = (z>.01) & (np.abs(xy[:,0])<k[4]/(2*k[0])) & (np.abs(xy[:,1])<k[5]/(2*k[1]))
            near,far = np.quantile(z[valid],[.005,.995]) * [.8,1.2]
        else:
            near,far = np.quantile(xyz[valid,2],[.001,.999]) * [.7,1.3]
        near = max(.02,float(near))
        factor = width/k[4]
        tk = [float(x)*factor for x in case['k'][:4]]+[width,round(k[5]*factor)]
        sk = [float(x)*factor for x in k[:4]]+[width,round(k[5]*factor)]
        # Evaluation retains original full-resolution source photographs.
        source_width = width if target_rgb else k[4]
        source_k = sk if target_rgb else k
        src = [cv2.resize(images[i],(source_width,round(k[5]*source_width/k[4])),interpolation=cv2.INTER_AREA) if source_width!=k[4] else images[i] for i in indices]
        data = dict(camera=torch.tensor(camera_vector(r,t,tk))[None],rgb_path=[case['image']],
            depth_range=torch.tensor([[near,float(far)]],dtype=torch.float32),
            src_rgbs=torch.from_numpy(np.stack(src).astype(np.float32)/255)[None],
            src_cameras=torch.from_numpy(np.stack([camera_vector(poses[i][1],poses[i][2],source_k) for i in indices]))[None])
        if target_rgb:
            target_index = case['index']
            assert target_index%8 != 0 and target_index not in indices
            data['rgb'] = torch.from_numpy(cv2.resize(images[target_index],(tk[4],tk[5]),interpolation=cv2.INTER_AREA).astype(np.float32)/255)[None]
        return data, indices, [near,float(far)]
    def render(dataset, case, label, config, width=480):
        dest = output/'renders'/dataset/label/f'{case["id"]}_{width}.png'
        existing = next((row for row in report['runs'] if row['dataset']==dataset and row['case']==case['id'] and row['label']==label and row['width']==width),None)
        if existing and dest.exists():
            assert existing['config']==config, 'Cached render uses different inference settings'
            return existing
        dest.parent.mkdir(parents=True,exist_ok=True)
        with torch.inference_mode():
            data,indices,bounds = make_data(dataset,case,config,width)
            legacy_dir = args.trial/'results/gnt'/dataset
            if case['kind']=='motion': legacy_dir = legacy_dir/'motion_480'
            legacy_path = legacy_dir/f'{case["id"]}.png'
            reuse = label=='control' and config==CONFIGS['control'] and legacy_path.exists() and (width==960 or case['kind']=='motion')
            if reuse:
                metadata = json.loads((legacy_dir/'runtime.json').read_text())
                assert metadata['checkpoint_sha256']==original_hash
                old_row = next(r for r in metadata['cases'] if r['id']==case['id'])
                assert old_row['sources']==indices and old_row['width']==width
                rgb = cv2.cvtColor(cv2.imread(str(legacy_path)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                seconds,peak = old_row['seconds'],metadata['peak_cuda_mb']
            else:
                sampler = RaySamplerSingleImage(data,'cuda:0')
                batch = sampler.get_all()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); tick = perf_counter()
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=config.get('precision')=='fp16'):
                    features = model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2))
                    result = render_single_image(sampler,batch,model,projector,config.get('chunk',args.chunk),N_samples=config['samples'],
                        inv_uniform=config['inverse'],N_importance=0,det=True,featmaps=features,single_net=True)
                torch.cuda.synchronize(); seconds = perf_counter()-tick
                rgb = result['outputs_coarse']['rgb'].float().numpy()
                peak = torch.cuda.max_memory_allocated()/2**20
            assert np.isfinite(rgb).all()
            cv2.imwrite(str(dest),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR))
            _,_,_,images,_ = dataset_data(dataset)
            gt = cv2.resize(images[case['index']],(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            from skimage.metrics import structural_similarity
            row = dict(dataset=dataset,case=case['id'],image=case['image'],label=label,width=width,
                kind=case['kind'],
                path=str(dest.relative_to(output)),config=config,sources=indices,depth_bounds=bounds,
                seconds=seconds,peak_cuda_mb=peak,reused_original_control=reuse,
                psnr=float(-10*np.log10(np.mean((rgb-gt)**2))),
                ssim=float(structural_similarity(rgb,gt,channel_axis=2,data_range=1)))
            if case['kind']=='motion' or case.get('zoom',1.)!=1.:
                row.update(psnr=None,ssim=None)
            report['runs'].append(row); save_report()
            print(json.dumps(row),flush=True)
        torch.cuda.empty_cache()
        return row
    scene_cases = [c for c in manifest['scene']['cases'] if c['kind']=='heldout' and c['zoom']==1]
    development = [c for c in scene_cases if c['index'] in (120,240)]
    confirmation = [c for c in scene_cases if c['index'] in (96,184,272)]
    if args.phase in ('traversal-bench','traversal-precision','traversal'):
        from gnt_traversal import traversal_cases
        k,poses,_,_,_ = dataset_data('scene')
        checkpoint = output/'detail_control_006000.pth'
        expected = report['detail_training']['control']['checkpoint']['sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
        load(checkpoint)
        cases = traversal_cases(poses,k)
        config = CONFIGS['control']
        if args.phase=='traversal-bench':
            probe = cases[240]
            trials = []
            for chunk in (512,2048,4096,8192):
                try:
                    row = render('scene',probe,f'traversal_probe_chunk{chunk}',config|dict(chunk=chunk),160)
                    trials.append(row)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(json.dumps(dict(chunk=chunk,status='out_of_memory')),flush=True)
                    break
            reference = cv2.imread(str(output/trials[0]['path'])).astype(np.int16)
            eligible = []
            for row in trials:
                image = cv2.imread(str(output/row['path'])).astype(np.int16)
                row['max_pixel_difference'] = int(np.abs(image-reference).max())
                if row['max_pixel_difference']<=1 and row['peak_cuda_mb']<6500:
                    eligible.append(row)
            winner = min(eligible,key=lambda row:row['seconds'])
            report['traversal_benchmark'] = dict(trials=trials,chunk=winner['config']['chunk'],
                note='160px probe, same 6000-step RGB model and inference settings; chunk sizes only. Pixel differences checked after lossless 8-bit encoding. Full-resolution throughput may differ.')
            save_report()
        elif args.phase=='traversal-precision':
            rows = []
            failure = None
            for index in (0,120,240):
                reference_row = render('scene',cases[index],'traversal_probe_chunk4096',config|dict(chunk=4096),160)
                reference = cv2.imread(str(output/reference_row['path'])).astype(np.float64)
                for chunk in (4096,8192):
                    try:
                        row = render('scene',cases[index],f'traversal_probe_fp16_{chunk}',config|dict(chunk=chunk,precision='fp16'),160)
                    except (AssertionError,torch.cuda.OutOfMemoryError) as error:
                        failure = dict(case=cases[index]['id'],chunk=chunk,error=str(error) or 'Nonfinite rendered RGB')
                        torch.cuda.empty_cache()
                        break
                    image = cv2.imread(str(output/row['path'])).astype(np.float64)
                    difference = image-reference
                    row['max_pixel_difference'] = float(np.abs(difference).max())
                    row['equivalence_psnr'] = float(-10*np.log10(max(np.mean(difference**2),1e-12)/255**2))
                    rows.append(row)
                if failure: break
            valid = [chunk for chunk in (4096,8192) if not failure and sum(row['config']['chunk']==chunk for row in rows)==3 and all(
                row['max_pixel_difference']<=2 and row['equivalence_psnr']>=60 and row['peak_cuda_mb']<6500
                for row in rows if row['config']['chunk']==chunk)]
            winner = min(valid,key=lambda chunk:sum(row['seconds'] for row in rows if row['config']['chunk']==chunk)) if valid else None
            report['traversal_precision'] = dict(trials=rows,chunk=winner,precision='fp16' if winner else 'fp32',
                failure=failure,
                note='Three 160px probes against FP32. FP16 allowed only with max 2 grey levels, PSNR at least 60 dB, and peak allocation below 6500 MiB on every probe. Native pilot must also pass before full export.')
            save_report()
            if winner:
                checks=[]
                for case in cases[:3]:
                    reference_row=render('scene',case,'traversal_control6000',config|dict(chunk=4096),960)
                    candidate_row=render('scene',case,'traversal_control6000_fp16',config|dict(chunk=winner,precision='fp16'),960)
                    reference=cv2.imread(str(output/reference_row['path'])).astype(np.float64)
                    candidate=cv2.imread(str(output/candidate_row['path'])).astype(np.float64)
                    difference=candidate-reference
                    checks.append(dict(case=case['id'],max_pixel_difference=float(np.abs(difference).max()),
                        equivalence_psnr=float(-10*np.log10(max(np.mean(difference**2),1e-12)/255**2))))
                passed=all(row['max_pixel_difference']<=2 and row['equivalence_psnr']>=60 for row in checks)
                report['traversal_precision'].update(native_checks=checks,native_verified=passed)
                if not passed:
                    report['traversal_precision']['precision']='fp32'
                save_report()
        else:
            chunk = report['traversal_benchmark']['chunk']
            label = 'traversal_control6000'
            precision = report.get('traversal_precision',{})
            if precision.get('precision')=='fp16':
                assert args.traversal_limit is not None or precision.get('native_verified'), 'Verify native precision pilot before full export'
                chunk = precision['chunk']
                config = config|dict(precision='fp16')
                label += '_fp16'
            report['traversal'] = dict(label=label,checkpoint_sha256=expected,width=args.traversal_width,
                chunk=chunk,cases=cases,source_exclusion='All interval-8 RGB remains excluded.',
                note='Midpoint cameras between every adjacent captured pose; interpolated camera centers and spherical rotations. Full capture route, not just the indoor segment. Each output is an independent GNT render. No frame blending or learned video interpolation.')
            save_report()
            progress_folder=output/'renders/scene'/label
            progress_folder.mkdir(parents=True,exist_ok=True)
            tick=perf_counter();durations=[]
            for i,case in enumerate(cases[:args.traversal_limit]):
                row=render('scene',case,label,config|dict(chunk=chunk),args.traversal_width)
                durations.append(row['seconds'])
                (progress_folder/'progress.json').write_text(json.dumps(dict(completed=i+1,total=len(cases),
                    width=args.traversal_width,last_frame=Path(row['path']).name,
                    source_time=case['source_time'],elapsed_seconds=perf_counter()-tick,
                    estimated_remaining_seconds=float(np.mean(durations))*(len(cases)-i-1),published=False)))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    elif args.phase in ('detail','detail-final'):
        from gnt_detail_loss import patch_indices, structural_loss
        k,poses,points,images,sharp = dataset_data('scene')
        baseline = output/'balanced_005000.pth'
        baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
        config = CONFIGS['control']
        def captured_case(index,kind='heldout'):
            name,r,t = poses[index]
            return dict(id=f'eval_{index:03}_z1.0',index=index,image=name,
                        r=r.tolist(),t=t.tolist(),k=list(k),kind=kind,zoom=1.)
        validation = [captured_case(i,'validation') for i in (24,72,112,152,200,248)]
        confirmation_new = [captured_case(i,'confirmation') for i in (56,144,224)]
        if args.phase=='detail':
            arm = args.detail_arm
            label = f'detail_{arm}6000'
            path = output/f'detail_{arm}_006000.pth'
            assert not path.exists(), 'Completed detail arm already exists; use detail-final to evaluate'
            assert not any(r['label']==label for r in report['runs']), 'Do not reuse stale labelled renders'
            load(baseline)
            optimizer = torch.optim.Adam([
                {'params':model.net_coarse.parameters(),'lr':1e-5},
                {'params':model.feature_net.parameters(),'lr':5e-6}])
            state = torch.load(output/'balanced_005000_continuation.pth',map_location='cpu',weights_only=False)
            assert state['step']==5000
            optimizer.load_state_dict(state['optimizer'])
            rng = np.random.default_rng()
            rng.bit_generator.state = state['numpy_rng']
            patch_rng = np.random.default_rng(2028)
            ray_rng.set_state(state['ray_rng'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            groups = report['balanced_training']['groups']
            training = dict(initial_step=5000,total_steps=6000,additional_steps=1000,
                baseline_sha256=baseline_hash,arm=arm,label=label,groups=groups,
                targets=report['balanced_training']['targets'],validation=[c['index'] for c in validation],
                fresh_confirmation=[c['index'] for c in confirmation_new],width=960,rays=256,samples=64,source_views=6,
                learning_rates=[group['lr'] for group in optimizer.param_groups],patch_side=16 if arm=='patch' else None,
                structural_weight=.01 if arm=='patch' else 0.,structural_window=7 if arm=='patch' else None,
                losses=[],optimizer_state_restored=True,patch_seed=2028,
                note='Matched target-camera sequence and initial optimizer/random states. Control uses uniform rays and MSE; patch arm uses a contiguous 16x16 patch and MSE+0.01*DSSIM. Sampling and loss change together; this is a combined-method A/B, not an isolated loss ablation. All interval-8 RGB excluded; cameras estimated before split. Previously inspected held-out views are exploratory; three fresh confirmation cameras excluded from model selection.')
            report.setdefault('detail_training',{})[arm] = training
            save_report()
            targets_seen = []
            tick = perf_counter(); torch.cuda.reset_peak_memory_stats()
            for step in range(5001,6001):
                group = groups[(step-1801)%len(groups)]
                idx = int(rng.choice(group)); targets_seen.append(idx)
                case = captured_case(idx)
                data,indices,_ = make_data('scene',case,config|dict(views=6),960,target_rgb=True)
                assert idx not in indices and idx%8!=0
                sampler = RaySamplerSingleImage(data,'cuda:0')
                batch = sampler.random_sample(256,'uniform')
                if arm=='patch':
                    selected = patch_indices(sampler.H,sampler.W,patch_rng)
                    for key,value in (('ray_o',sampler.rays_o),('ray_d',sampler.rays_d),('rgb',sampler.rgb)):
                        batch[key] = value[selected].cuda()
                    batch['selected_inds'] = selected
                model.switch_to_train(); optimizer.zero_grad(set_to_none=True)
                features = model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2))
                result = render_rays(batch,model,features,projector,64,inv_uniform=True,N_importance=0,det=False,single_net=True)
                prediction = result['outputs_coarse']['rgb']
                mse = ((prediction-batch['rgb'])**2).mean()
                structure = structural_loss(prediction,batch['rgb']) if arm=='patch' else mse.new_zeros(())
                loss = mse+.01*structure
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite detail-training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.net_coarse.parameters())+list(model.feature_net.parameters()),1.)
                optimizer.step()
                training['peak_cuda_mb'] = max(training.get('peak_cuda_mb',0),torch.cuda.max_memory_allocated()/2**20)
                if step%50==0:
                    row = dict(step=step,target=idx,mse=mse.item(),dssim=structure.item(),objective=loss.item(),seconds=perf_counter()-tick)
                    training['losses'].append(row);save_report();print(json.dumps(dict(arm=arm,**row)),flush=True)
                del features,result,prediction,mse,structure,loss,batch,sampler,data
            torch.cuda.synchronize()
            training['optimization_seconds'] = perf_counter()-tick
            training['target_sequence_sha256'] = hashlib.sha256(json.dumps(targets_seen).encode()).hexdigest()
            training['first_targets'] = targets_seen[:12]
            torch.save(dict(net_coarse=model.net_coarse.state_dict(),feature_net=model.feature_net.state_dict()),path)
            training['checkpoint'] = dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            torch.save(dict(step=6000,optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                patch_rng=patch_rng.bit_generator.state,ray_rng=ray_rng.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()),output/f'detail_{arm}_006000_continuation.pth')
            save_report(); model.switch_to_eval()
            for case in validation:
                render('scene',case,label,config,width=480)
            training['seconds_including_validation'] = perf_counter()-tick
            save_report()
        else:
            arms = report['detail_training']
            assert arms['control']['target_sequence_sha256']==arms['patch']['target_sequence_sha256']
            labels = ['balanced5000','detail_control6000','detail_patch6000']
            report['detail_cases'] = report['balanced_cases']+validation+confirmation_new
            ids = {c['id'] for c in validation}
            report['detail_validation'] = {label:dict(
                psnr=float(np.mean([r['psnr'] for r in report['runs'] if r['label']==label and r['width']==480 and r['case'] in ids])),
                ssim=float(np.mean([r['ssim'] for r in report['runs'] if r['label']==label and r['width']==480 and r['case'] in ids]))) for label in labels}
            assert all(sum(r['label']==label and r['width']==480 and r['case'] in ids for r in report['runs'])==6 for label in labels)
            save_report()
            for label in labels:
                path = baseline if label=='balanced5000' else output/f'detail_{"control" if label=="detail_control6000" else "patch"}_006000.pth'
                load(path)
                for case in report['detail_cases']:
                    width = 320 if case['kind']=='motion' else (480 if case['kind'] in ('validation','confirmation') else 960)
                    render('scene',case,label,config,width=width)
        assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_hash
    elif args.phase in ('balanced', 'balanced-final'):
        k,poses,points,images,sharp = dataset_data('scene')
        def captured_case(index):
            name,r,t = poses[index]
            return dict(id=f'eval_{index:03}_z1.0',index=index,image=name,
                        r=r.tolist(),t=t.tolist(),k=list(k),kind='heldout',zoom=1.)
        validation = [captured_case(i) for i in (24,72,112,152,200,248)]
        test = [captured_case(i) for i in (40,176,264)]
        baseline = output/'adapt_001800.pth'
        config = CONFIGS['control']
        if args.phase == 'balanced':
            assert args.steps == 5000, 'This bounded experiment ends at 5,000 total steps'
            assert not (output/'balanced_003200.pth').exists(), 'Balanced checkpoint already exists; use balanced-final to evaluate without overwriting'
            load(baseline)
            groups = balanced_targets(len(poses))
            # Only camera geometry and eligible training views determine eligibility.
            rejected = []
            for group in groups:
                for idx in list(group):
                    _,r,t = poses[idx]
                    xyz = points@r.T+t
                    z = xyz[:,2]
                    xy = xyz[:,:2]/np.maximum(z[:,None],.001)
                    mask = (z>.01)&(np.abs(xy[:,0])<k[4]/(2*k[0]))&(np.abs(xy[:,1])<k[5]/(2*k[1]))
                    selected = sources(r,t,poses,points,k,6,'nearest',sharp,exclude=idx)
                    if mask.sum()<2 or len(selected)<6:
                        group.remove(idx); rejected.append(idx)
            assert all(groups)
            training = dict(initial_step=1800,total_steps=5000,groups=groups,rejected_targets=rejected,
                targets=sorted(i for group in groups for i in group),validation=[c['index'] for c in validation],
                test=[c['index'] for c in test],width=960,rays=256,source_views=6,samples=64,
                learning_rates=[1e-5,5e-6],losses=[],checkpoints=[],
                target_sampling_seed=2027,group_sampling='Deterministic round robin; uniform random eligible target within each group',
                baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                note='Equal probability for six contiguous capture intervals, not semantic indoor/outdoor labels. All interval-8 RGB excluded from training and sources. Camera estimation predates split. Fresh Adam at step 1800; exact continuation state saved thereafter.')
            report['balanced_training'] = training
            save_report()
            for case in validation:
                render('scene',case,'adapt1800',config,width=480)
            optimizer = torch.optim.Adam([
                {'params':model.net_coarse.parameters(),'lr':1e-5},
                {'params':model.feature_net.parameters(),'lr':5e-6}])
            rng = np.random.default_rng(2027)
            tick = perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for step in range(1801,5001):
                group = groups[(step-1801)%len(groups)]
                idx = int(rng.choice(group))
                case = captured_case(idx)
                data,indices,_ = make_data('scene',case,config|dict(views=6),960,target_rgb=True)
                assert idx not in indices and idx%8 != 0
                sampler = RaySamplerSingleImage(data,'cuda:0')
                batch = sampler.random_sample(256,'uniform')
                model.switch_to_train(); optimizer.zero_grad(set_to_none=True)
                features = model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2))
                result = render_rays(batch,model,features,projector,64,inv_uniform=True,N_importance=0,det=False,single_net=True)
                loss = ((result['outputs_coarse']['rgb']-batch['rgb'])**2).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.net_coarse.parameters())+list(model.feature_net.parameters()),1.)
                optimizer.step()
                training['peak_cuda_mb'] = max(training.get('peak_cuda_mb',0),torch.cuda.max_memory_allocated()/2**20)
                if step%50==0:
                    row = dict(step=step,mse=loss.item(),seconds=perf_counter()-tick)
                    training['losses'].append(row); save_report(); print(json.dumps(row),flush=True)
                del features,result,batch,sampler,data,loss
                if step in (3200,5000):
                    path = output/f'balanced_{step:06}.pth'
                    torch.save(dict(net_coarse=model.net_coarse.state_dict(),feature_net=model.feature_net.state_dict()),path)
                    torch.save(dict(step=step,optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                        ray_rng=ray_rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),
                        output/f'balanced_{step:06}_continuation.pth')
                    training['checkpoints'].append(dict(step=step,path=path.name,seconds=perf_counter()-tick))
                    save_report(); model.switch_to_eval()
                    for case in validation:
                        render('scene',case,f'balanced{step}',config,width=480)
            training['seconds_including_validation'] = perf_counter()-tick
            save_report()
        else:
            report['balanced_training'].setdefault('target_sampling_seed',2027)
            report['balanced_training'].setdefault('group_sampling','Deterministic round robin; uniform random eligible target within each group')
            labels = ['adapt1800','balanced3200','balanced5000']
            ids = {c['id'] for c in validation}
            scores = {}
            for label in labels:
                rows = [r for r in report['runs'] if r['label']==label and r['width']==480 and r['case'] in ids]
                assert len(rows)==len(ids), (label,len(rows))
                scores[label] = float(np.mean([r['psnr'] for r in rows]))
            winner = max(labels,key=scores.get)
            report['balanced_selection'] = dict(label=winner,validation_psnr=scores,
                policy='Highest mean PSNR across six predeclared validation cameras; test cameras excluded from selection.')
            save_report()
            # Show the 5k model even if validation selects the existing control.
            comparison_labels = list(dict.fromkeys(['adapt1800',winner,'balanced5000']))
            closeups = [c for c in manifest['scene']['cases'] if c['kind']=='heldout' and c['index'] in (120,240) and c['zoom']==1.7]
            for row in report['runs']:
                if row['case'] in {c['id'] for c in closeups}:
                    row.update(psnr=None,ssim=None)
            path_cases = []
            from scipy.spatial.transform import Rotation, Slerp
            # Follow the indoor capture trajectory; avoid exact training-camera poses.
            anchors = [captured_case(i) for i in (232,236,240,244,248)]
            times = np.array([232.,236.,240.,244.,248.])
            rotations = Rotation.from_matrix(np.array([c['r'] for c in anchors]))
            centers = np.array([-np.array(c['r']).T@np.array(c['t']) for c in anchors])
            interp = Slerp(times,rotations)
            for i,u in enumerate(np.linspace(232.25,247.75,17)):
                r = interp(u).as_matrix()
                center = np.array([np.interp(u,times,centers[:,axis]) for axis in range(3)])
                case = dict(anchors[2],id=f'path_walk_{i:02}',kind='motion',r=r.tolist(),t=(-r@center).tolist())
                path_cases.append(case)
            report['balanced_cases'] = test+development+closeups+path_cases
            save_report()
            for label in comparison_labels:
                load(baseline if label=='adapt1800' else output/f'balanced_{int(label[8:]):06}.pth')
                for case in test+development+closeups:
                    render('scene',case,label,config,width=960)
                for case in path_cases:
                    render('scene',case,label,config,width=320)
    elif args.phase == 'sweep':
        for label, delta in CONFIGS.items():
            config = CONFIGS['control'] | delta
            for case in development:
                render('scene',case,label,config)
        scores = {label:np.mean([r['psnr'] for r in report['runs'] if r['dataset']=='scene' and r['width']==480 and r['label']==label and r['case'] in [c['id'] for c in development]]) for label in CONFIGS}
        winner = max(scores,key=scores.get)
        report['inference_selection'] = dict(label=winner,config=CONFIGS['control']|CONFIGS[winner],development_psnr=scores)
        save_report()
        # A combined setting is checked separately rather than assumed additive.
        combined = CONFIGS['control'] | dict(samples=192,selection='overlap')
        for case in development:
            render('scene',case,'combined',combined)
        combined_score = np.mean([r['psnr'] for r in report['runs'] if r['label']=='combined' and r['width']==480])
        if combined_score > scores[winner]:
            report['inference_selection'].update(label='combined',config=combined)
            save_report()
        choose_inference()
    elif args.phase == 'adapt':
        choose_inference()
        previous = report.get('adaptation',{}) if args.resume_step else {}
        if args.resume_step:
            load(output/f'adapt_{args.resume_step:06}.pth')
        k,poses,points,images,sharp = dataset_data('scene')
        # Balanced local adaptation around both inspected regions, never held-out RGB.
        targets = set()
        for case in development:
            r,t = np.array(case['r']),np.array(case['t'])
            targets.update(sources(r,t,poses,points,k,24,'nearest',sharp))
        targets = sorted(targets)
        assert targets and all(i%8 != 0 for i in targets)
        train_config = CONFIGS['control'] | dict(views=6)
        rates = [1e-5,5e-6] if args.resume_step else [2e-5,1e-5]
        optimizer = torch.optim.Adam([
            {'params':model.net_coarse.parameters(),'lr':rates[0]},
            {'params':model.feature_net.parameters(),'lr':rates[1]}])
        rng = np.random.default_rng(2026)
        history = previous.get('phases',[])
        if previous:
            history = history + [{key:value for key,value in previous.items() if key not in ('phases','checkpoints','losses')}]
        training = dict(targets=targets,source_views=6,width=args.train_width,rays=256,samples=64,
            learning_rates=rates,steps=args.steps,resume_step=args.resume_step,
            checkpoints=previous.get('checkpoints',[]),losses=previous.get('losses',[]),phases=history,
            note='Bounded full-network RGB-MSE adaptation, not the official 60k-step schedule. Continuation loads weights with fresh Adam state.')
        report['adaptation'] = training; save_report()
        tick = perf_counter()
        for step in range(args.resume_step+1,args.steps+1):
            idx = targets[int(rng.integers(len(targets)))]
            name,r,t = poses[idx]
            case = dict(id=f'train_{idx}',index=idx,image=name,r=r.tolist(),t=t.tolist(),k=k)
            data,indices,_ = make_data('scene',case,train_config,args.train_width,target_rgb=True)
            # A training target cannot serve as its own source.
            if idx in indices:
                raise AssertionError('Training target appears as a source')
            sampler = RaySamplerSingleImage(data,'cuda:0')
            batch = sampler.random_sample(256,'uniform')
            model.switch_to_train()
            optimizer.zero_grad(set_to_none=True)
            features = model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2))
            result = render_rays(batch,model,features,projector,64,inv_uniform=True,N_importance=0,det=False,single_net=True)
            loss = ((result['outputs_coarse']['rgb']-batch['rgb'])**2).mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.net_coarse.parameters())+list(model.feature_net.parameters()),1.)
            optimizer.step()
            training['peak_cuda_mb'] = max(training.get('peak_cuda_mb',0),torch.cuda.max_memory_allocated()/2**20)
            if step%25==0:
                training['losses'].append(dict(step=step,mse=loss.item(),seconds=perf_counter()-tick))
                save_report(); print(json.dumps(training['losses'][-1]),flush=True)
            del features,result,batch,sampler,data,loss
            if step in (200,args.steps):
                path = output/f'adapt_{step:06}.pth'
                torch.save(dict(net_coarse=model.net_coarse.state_dict(),feature_net=model.feature_net.state_dict()),path)
                training['checkpoints'].append(dict(step=step,path=path.name,seconds=perf_counter()-tick))
                save_report()
                model.switch_to_eval()
                for case in development:
                    render('scene',case,f'adapt{step}',report['inference_selection']['config'])
        training['seconds_including_validation'] = perf_counter()-tick
        save_report()
    else:
        inference = report['inference_selection']
        labels = ['control',inference['label']] + [f'adapt{x["step"]}' for x in report['adaptation'].get('checkpoints',[])]
        labels = list(dict.fromkeys(labels))
        dev_ids = [c['id'] for c in development]
        scores = {label:np.mean([r['psnr'] for r in report['runs'] if r['label']==label and r['width']==480 and r['case'] in dev_ids]) for label in labels}
        best = max(scores,key=scores.get)
        report['final_selection'] = dict(label=best,development_psnr=scores)
        save_report()
        for label in list(dict.fromkeys(['control',inference['label'],best])):
            load(output/f'adapt_{int(label[5:]):06}.pth' if label.startswith('adapt') else weights)
            config = CONFIGS['control'] if label=='control' else inference['config']
            for case in development+confirmation:
                render('scene',case,label,config,width=960)
        for label in list(dict.fromkeys(['control',best])):
            load(output/f'adapt_{int(label[5:]):06}.pth' if label.startswith('adapt') else weights)
            config = CONFIGS['control'] if label=='control' else inference['config']
            for case in manifest['scene']['cases']:
                if case['id'] in ('motion_00','motion_04','motion_08'):
                    render('scene',case,label,config,width=480)
        load(weights)
        for dataset in ('box','boot','milk'):
            if inference['label']=='control':
                report.setdefault('object_decision',{})[dataset]='Inference tuning did not justify a setting change; retain the existing original-GNT comparison and preferred restoration. Scene-adapted weights are not applied to unrelated objects.'
                save_report()
                continue
            case = next(c for c in manifest[dataset]['cases'] if c['kind']=='heldout' and c['index']==24 and c['zoom']==1)
            render(dataset,case,inference['label'],inference['config'],width=960)
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==original_hash
    print('Complete',args.phase,flush=True)


if __name__ == '__main__':
    main()
