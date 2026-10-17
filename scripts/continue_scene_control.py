"""Continue the exact fresh scene control, then compare 5k/10k/15k renders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/splat_surroundings_tests'
OUT = ROOT / 'outputs/scene_control_continuation'
SOURCE = BASE / 'train/scene_control/splatfacto/run_5000'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def audit():
    config = SOURCE / 'config.yml'
    checkpoint = SOURCE / 'nerfstudio_models/step-000004999.ckpt'
    assert config.is_file() and checkpoint.is_file()
    assert (BASE / 'evaluation/scene_control/metrics.json').is_file()
    record = dict(source=checkpoint.relative_to(ROOT).as_posix(),
                  config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  total_steps=[5000, 10000, 15000],
                  changes='Continue optimization only; preserve splitting stop and resolution schedule',
                  selected_models_unchanged=True)
    record['signature'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = OUT / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError('Source changed; refusing to mix experiments')
    save(path, record)
    return record


def segment_steps(start_step, total):
    count = total - (start_step + 1)
    if count <= 0:
        raise ValueError('Target must be after checkpoint step')
    return count


def bind_loaded_optimizers(trainer, state):
    # Splatfacto load_state_dict replaces Gaussian Parameters. Rebuild optimizers
    # AFTER loading the pipeline so they update the live tensors, not old seeds.
    trainer.optimizers = trainer.setup_optimizers()
    trainer.optimizers.load_optimizers(state['optimizers'])
    if 'schedulers' in state and trainer.config.load_scheduler:
        trainer.optimizers.load_schedulers(state['schedulers'])
    trainer.grad_scaler.load_state_dict(state['scalers'])
    groups = trainer.pipeline.get_param_groups()
    for name, optimizer in trainer.optimizers.optimizers.items():
        actual = [p for group in optimizer.param_groups for p in group['params']]
        assert [id(p) for p in actual] == [id(p) for p in groups[name]], name


def train_segment(total, plan):
    import torch
    import yaml
    from nerfstudio.engine.trainer import Trainer
    from nerfstudio.scripts.train import main as train

    previous = SOURCE if total == 10000 else OUT / 'train/scene_10000/splatfacto/continued'
    start = previous / f'nerfstudio_models/step-{total-5001:09d}.ckpt'
    assert start.is_file(), start
    state = torch.load(start, map_location='cpu', weights_only=False)
    count = len(state['pipeline']['_model.gauss_params.means'])
    config = yaml.load((SOURCE / 'config.yml').read_text(), Loader=yaml.Loader)
    config.output_dir = OUT / 'train'
    config.experiment_name = f'scene_{total}'
    config.timestamp = 'continued'
    config.load_dir = config.load_config = config.load_step = None
    config.load_checkpoint = start
    # Nerfstudio 1.1.5 treats max_num_iterations as ADDITIONAL iterations on resume.
    config.max_num_iterations = segment_steps(state['step'], total)
    config.steps_per_save = 1000
    config.steps_per_eval_all_images = 1000000
    config.vis = 'tensorboard'
    assert config.pipeline.model.stop_split_at == 2500
    assert config.pipeline.model.resolution_schedule == 3000
    assert config.pipeline.model.camera_optimizer.mode == 'off'
    checkpoint = config.get_checkpoint_dir() / f'step-{total-1:09d}.ckpt'
    runtime_path = checkpoint.parent.parent / 'continuation_runtime.json'
    if checkpoint.exists():
        runtime = json.loads(runtime_path.read_text())
        assert runtime['signature'] == plan['signature']
        return

    original = Trainer._load_checkpoint

    def load(self):
        self._start_step = state['step'] + 1
        self.pipeline.load_pipeline(state['pipeline'], state['step'])
        bind_loaded_optimizers(self, state)
        assert len(self.pipeline.model.gauss_params['means']) == count
        print('Loaded checkpoint; optimizer bindings verified', flush=True)

    Trainer._load_checkpoint = load
    tick = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        train(config)
    finally:
        Trainer._load_checkpoint = original
    result = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert result['step'] == total - 1
    params = {k: v for k, v in result['pipeline'].items() if k.startswith('_model.gauss_params.')}
    assert all(len(v) == count and torch.isfinite(v).all() for v in params.values())
    assert not torch.equal(params['_model.gauss_params.means'], state['pipeline']['_model.gauss_params.means']), 'Means did not update'
    save(runtime_path, dict(signature=plan['signature'], total_steps=total,
         additional_steps=config.max_num_iterations, seconds=perf_counter()-tick,
         peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
         gaussians=count, optimizer_bindings_verified=True, means_changed=True,
         source=str(start), checkpoint=str(checkpoint)))


def run(label, arguments):
    log = OUT / 'logs' / f'{label}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        process = subprocess.Popen([sys.executable, *map(str, arguments)], cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONUNBUFFERED='1'),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError(f'{label} failed; see {log}')


def finalize():
    from html import escape
    import numpy as np
    from PIL import Image
    from sfm_reconstruction.splat_cleanup_evaluation import compare_candidate

    reference = json.loads((BASE/'evaluation/scene_control/metrics.json').read_text())
    reports = {}
    html = ['<!doctype html><meta charset="utf-8"><title>Scene continuation</title>',
            '<style>body{background:#171717;color:#eee;font:16px Segoe UI;margin:24px}img{max-width:100%}a{color:#9cf}</style>',
            '<h1>Scene continuation</h1><p>Left to right: source photograph / fresh control 5k / continued 10k / continued 15k. Selected models unchanged.</p>']
    metrics = {}
    for total in [10000, 15000]:
        metrics[total] = json.loads((OUT/f'evaluation/scene_{total}/metrics.json').read_text())
        runtime = json.loads((OUT/f'train/scene_{total}/splatfacto/continued/continuation_runtime.json').read_text())
        reports[str(total)] = dict(against_fresh_control=compare_candidate(reference, metrics[total]), runtime=runtime)
        gate = reports[str(total)]['against_fresh_control']
        html.append(f'<p>{total:,} steps: mean PSNR change {gate["mean_deltas"]["psnr"]:+.3f} dB; worst view {gate["worst_view_psnr_delta"]:+.3f} dB; segment training {runtime["seconds"]/60:.1f} min.</p>')
    indices = {total: {v['image']: i for i, v in enumerate(m['views'])} for total, m in metrics.items()}
    for i, row in enumerate(reference['views']):
        pair = Image.open(BASE/f'evaluation/scene_control/{i:03d}_img.png').convert('RGB')
        w, h = pair.width//2, pair.height
        panels = [pair.crop((0,0,w,h)), pair.crop((w,0,2*w,h))]
        for total in [10000, 15000]:
            other = Image.open(OUT/f'evaluation/scene_{total}/{indices[total][row["image"]]:03d}_img.png').convert('RGB')
            assert other.size == pair.size
            assert np.array_equal(np.array(panels[0]), np.array(other.crop((0,0,w,h))))
            panels.append(other.crop((w,0,2*w,h)))
        sheet = Image.new('RGB', (4*w,h))
        for j, panel in enumerate(panels):
            sheet.paste(panel,(j*w,0))
        file = f'comparisons/{i:03d}.jpg'
        (OUT/file).parent.mkdir(parents=True,exist_ok=True)
        sheet.save(OUT/file,quality=95)
        html.append(f'<p>{escape(row["image"])}</p><a href="{file}"><img loading="lazy" src="{file}"></a>')
    save(OUT/'comparisons.json',reports)
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--train-total',type=int,choices=[10000,15000])
    args = parser.parse_args()
    plan = audit()
    if args.prepare_only:
        print(json.dumps(plan,indent=2))
        return
    if args.train_total:
        train_segment(args.train_total,plan)
        return
    save(OUT/'workflow.json',dict(status='running',signature=plan['signature']))
    for total in [10000,15000]:
        run(f'train_{total}',['scripts/continue_scene_control.py','--train-total',total])
        checkpoint = OUT/f'train/scene_{total}/splatfacto/continued/nerfstudio_models/step-{total-1:09d}.ckpt'
        run(f'eval_{total}',['scripts/evaluate_splat_checkpoint.py','--config',SOURCE/'config.yml',
            '--checkpoint',checkpoint,'--output',OUT/f'evaluation/scene_{total}'])
    finalize()
    save(OUT/'workflow.json',dict(status='complete',signature=plan['signature']))
    print('Complete: outputs/scene_control_continuation/index.html',flush=True)


if __name__ == '__main__':
    main()
