"""Check gallery assets and score the delivered JPEG comparisons on CPU."""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ['TORCH_HOME'] = str(ROOT/'outputs/nerfstudio_cache/torch')
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from skimage.metrics import structural_similarity
    torch.set_num_threads(4)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).eval()
    page = (ROOT/'web_demo/gnt-tuning.html').read_text(encoding='utf-8')
    data = json.loads(re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>',page,re.S).group(1))
    resources = {'gnt-tuning.html','gnt-tuning-inspector.js'}
    output = ROOT/'outputs/gnt_tuning'
    old_path = output/'published_metrics.json'
    old_scores = json.loads(old_path.read_text())['scores'] if old_path.exists() else []
    cached = {(r['dataset'],r['case'],r['method']):r for r in old_scores}
    scores = []
    for dataset,cases in data['cases'].items():
        for case in cases:
            images = {}
            for method,url in case['urls'].items():
                resources.add(url)
                image = cv2.imread(str(ROOT/'web_demo'/url))
                assert image is not None and image.shape[:2]==(case['height'],case['width']),url
                images[method] = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            if case['kind']=='motion':
                assert 'gt' not in images
                continue
            gt = images['gt']
            gt_hash = hashlib.sha256((ROOT/'web_demo'/case['urls']['gt']).read_bytes()).hexdigest()
            gt_tensor = torch.from_numpy(gt.copy()).permute(2,0,1)[None]
            for method,image in images.items():
                if method=='gt': continue
                source_hash = hashlib.sha256((ROOT/'web_demo'/case['urls'][method]).read_bytes()).hexdigest()
                old = cached.get((dataset,case['id'],method),{})
                if old.get('source_sha256')==source_hash and old.get('reference_sha256')==gt_hash:
                    scores.append(old)
                    continue
                with torch.inference_mode():
                    distance = metric(torch.from_numpy(image.copy()).permute(2,0,1)[None],gt_tensor).item()
                scores.append(dict(dataset=dataset,case=case['id'],method=method,
                    source_sha256=source_hash,reference_sha256=gt_hash,
                    psnr=float(-10*np.log10(np.mean((image-gt)**2))),
                    ssim=float(structural_similarity(image,gt,channel_axis=2,data_range=1)),lpips=distance))
    for url in sorted(resources):
        with urllib.request.urlopen('http://localhost:8088/'+url,timeout=15) as response:
            assert response.status==200,url
    report = json.loads((ROOT/'outputs/gnt_tuning/results.json').read_text())
    assert all(all(i%8!=0 for i in row['sources']) for row in report['runs'])
    assert all(i%8!=0 for i in report['adaptation']['targets'])
    weights = Path(os.environ['TEMP'])/'sfm_lightweight_methods_v1/GNT/generalization_200000.pth'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==report['checkpoint_sha256']
    manifest = json.loads((Path(os.environ['TEMP'])/'sfm_sharpness_trials_v1/render_manifest.json').read_text())
    for dataset in manifest['datasets'].values():
        assert hashlib.sha256((ROOT/dataset['checkpoint']).read_bytes()).hexdigest()==dataset['checkpoint_sha256']
    model_record = json.loads((output/'model.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256(Path(model_record['path']).read_bytes()).hexdigest()==model_record['sha256']
    (output/'published_metrics.json').write_text(json.dumps(dict(scope='Delivered JPEG comparison against the same held-out JPEG photograph at the listed gallery resolution.',scores=scores),indent=2))
    verification = dict(status='passed',cases=sum(len(c) for c in data['cases'].values()),
        images=sum(len(c['urls']) for cases in data['cases'].values() for c in cases),http_resources=len(resources),
        excluded_heldout_rgb=True,original_gnt_checkpoint_unchanged=True,scored_comparisons=len(scores))
    verification.update(unchanged_control_checkpoints=len(manifest['datasets']),durable_model_verified=True)
    (output/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
