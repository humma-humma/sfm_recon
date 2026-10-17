"""Score the delivered JPEG comparisons with cached AlexNet LPIPS in Docker."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'outputs/sharpness_trials/results.json'
    report = json.loads(path.read_text())
    assert report['status'] == 'complete'
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda().eval()
    def load(path):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
    with torch.inference_mode():
        for dataset, record in report['datasets'].items():
            for case in record['cases']:
                if case['kind'] != 'heldout': continue
                folder = ROOT/'web_demo/assets/sharpness'/dataset
                gt = load(folder/f'{case["id"]}_gt.jpg')
                for method in report['methods']:
                    prediction = load(folder/f'{case["id"]}_{method}.jpg')
                    value = float(metric(prediction, gt))
                    metric.reset()
                    case['metrics'][method]['lpips_jpeg'] = value
            for method in report['methods']:
                record['mean'][method]['lpips_jpeg'] = float(np.mean([c['metrics'][method]['lpips_jpeg'] for c in record['cases'] if c['kind'] == 'heldout']))
            print(dataset, {m: v['lpips_jpeg'] for m, v in record['mean'].items()}, flush=True)
    report['perceptual_metric'] = 'AlexNet LPIPS on delivered quality-90 JPEGs at source resolution. Separate from pre-JPEG PSNR/SSIM; lower is better. Numeric fidelity gates are unchanged.'
    path.write_text(json.dumps(report, indent=2))
    (ROOT/'web_demo/assets/sharpness/results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
