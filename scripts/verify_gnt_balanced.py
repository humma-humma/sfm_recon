"""Validate delivered balanced-GNT images, split integrity, and perceptual scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['balanced','detail'],default='balanced')
    args = parser.parse_args()
    detail = args.experiment=='detail'
    slug = 'gnt_'+args.experiment
    os.environ['TORCH_HOME'] = str(ROOT/'outputs/nerfstudio_cache/torch')
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from skimage.metrics import structural_similarity
    torch.set_num_threads(4)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).eval()
    output = ROOT/'outputs'/slug
    data = json.loads((output/'results.json').read_text())
    trial = Path(os.environ['TEMP'])/'sfm_lightweight_methods_v1/gnt_tuning'
    report = json.loads((trial/'results.json').read_text())
    train = data['training']
    assert all(i%8!=0 for i in train['targets'])
    assert set(train['validation']).isdisjoint(train['test'])
    assert all(i%8==0 for i in train['validation']+train['test'])
    assert all(all(i%8!=0 for i in r['sources']) for r in report['runs'])
    assert hashlib.sha256((trial/('balanced_005000.pth' if detail else 'adapt_001800.pth')).read_bytes()).hexdigest()==train['baseline_sha256']
    for checkpoint in data['checkpoints']:
        assert hashlib.sha256((trial/checkpoint['path']).read_bytes()).hexdigest()==checkpoint['sha256']
    weights = trial.parent/'GNT/generalization_200000.pth'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==report['checkpoint_sha256']
    resources = {'gnt-'+args.experiment+'.html','assets/'+slug+'/results.json'}
    if detail:
        arms = report['detail_training']
        assert arms['control']['target_sequence_sha256']==arms['patch']['target_sequence_sha256']
        for key in ('baseline_sha256','groups','targets','validation','fresh_confirmation',
                    'width','rays','samples','source_views','learning_rates','optimizer_state_restored'):
            assert arms['control'][key]==arms['patch'][key],key
        assert arms['control']['optimizer_state_restored']
    scores = []
    for case in data['cases']:
        rows = [r for r in report['runs'] if r['case']==case['id'] and r['width']==case['width']
                and r['label'] in case['images']]
        assert len(rows)==len(case['images'])-('photo' in case['images'])
        assert all(r['sources']==rows[0]['sources'] and r['depth_bounds']==rows[0]['depth_bounds']
                   and r['config']==rows[0]['config'] for r in rows)
        images = {}
        for label,url in case['images'].items():
            resources.add(url)
            image = cv2.imread(str(ROOT/'web_demo'/url))
            assert image is not None and image.shape[:2]==(round(case['width']*540/960),case['width']),url
            images[label] = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if case['kind']=='motion':
            assert 'photo' not in images
            continue
        gt = images.pop('photo')
        gt_tensor = torch.from_numpy(gt.copy()).permute(2,0,1)[None]
        for label,image in images.items():
            with torch.inference_mode():
                distance = metric(torch.from_numpy(image.copy()).permute(2,0,1)[None],gt_tensor).item()
            scores.append(dict(case=case['id'],kind=case['kind'],label=label,width=case['width'],
                psnr=float(-10*np.log10(np.mean((image-gt)**2))),
                ssim=float(structural_similarity(image,gt,channel_axis=2,data_range=1)),lpips=distance))
    for resource in sorted(resources):
        with urllib.request.urlopen('http://127.0.0.1:8088/'+resource,timeout=15) as response:
            assert response.status==200,resource
    path_rows = [r for r in report['runs'] if r['label']==data.get('baseline_label','adapt1800') and r['case'].startswith('path_walk_')]
    assert len(path_rows)==17
    source_sets = len({tuple(r['sources']) for r in path_rows})
    assert source_sets>1, 'Trajectory must exercise source transitions'
    model_path = output/'models.json'
    if model_path.exists():
        for model in json.loads(model_path.read_text()):
            assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()==model['sha256']
    (output/'published_metrics.json').write_text(json.dumps(dict(scores=scores,
        note='Delivered JPEGs against matching photo references; close-up photo intrinsics adjusted.'),indent=2))
    result = dict(status='passed',cases=len(data['cases']),images=sum(len(c['images']) for c in data['cases']),
        scored_comparisons=len(scores),http_resources=len(resources),heldout_exclusion=True,
        baseline_unchanged=True,original_gnt_unchanged=True,path_frames=17,path_source_sets=source_sets)
    (output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
