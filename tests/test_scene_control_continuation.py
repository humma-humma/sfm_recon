import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/continue_scene_control.py'
    spec = importlib.util.spec_from_file_location('scene_continuation',path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_resume_iteration_budget():
    runner = module()
    assert runner.segment_steps(4999,10000) == 5000
    assert runner.segment_steps(9999,15000) == 5000
    with pytest.raises(ValueError):
        runner.segment_steps(9999,10000)


def test_optimizer_resume_targets_replaced_parameters():
    runner = module()
    source = torch.nn.Parameter(torch.tensor([2.]))
    previous = torch.optim.Adam([source],lr=.1)
    source.square().sum().backward()
    previous.step()
    state = dict(optimizers={'means': previous.state_dict()},scalers={})
    live = torch.nn.Parameter(source.detach().clone())
    stale = torch.nn.Parameter(torch.tensor([100.]))

    class Optimizers:
        def __init__(self,param):
            self.optimizers = {'means': torch.optim.Adam([param],lr=.1)}
        def load_optimizers(self,records):
            self.optimizers['means'].load_state_dict(records['means'])

    trainer = SimpleNamespace(
        optimizers=Optimizers(stale),setup_optimizers=lambda: Optimizers(live),
        pipeline=SimpleNamespace(get_param_groups=lambda: {'means':[live]}),
        config=SimpleNamespace(load_scheduler=True),
        grad_scaler=SimpleNamespace(load_state_dict=lambda state: None))
    runner.bind_loaded_optimizers(trainer,state)
    optimizer = trainer.optimizers.optimizers['means']
    assert optimizer.param_groups[0]['params'][0] is live
    assert optimizer.state[live]['step'].item() == 1
    before = live.detach().clone()
    live.square().sum().backward()
    optimizer.step()
    assert not torch.equal(live,before)
    assert stale.item() == 100.
    assert optimizer.state[live]['step'].item() == 2


def test_gallery_keeps_source_and_three_model_columns(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    runner = module()
    monkeypatch.setattr(runner,'BASE',tmp_path/'base')
    monkeypatch.setattr(runner,'OUT',tmp_path/'continuation')
    row = dict(image='view.jpg',psnr=20.,ssim=.8,lpips=.2,opacity_coverage=1.)
    reference = runner.BASE/'evaluation/scene_control'
    runner.save(reference/'metrics.json',dict(views=[row]))
    source = np.full((16,16,3),100,dtype=np.uint8)
    Image.fromarray(np.concatenate([source,source+10],axis=1)).save(reference/'000_img.png')
    for total,shade in [(10000,120),(15000,130)]:
        path = runner.OUT/f'evaluation/scene_{total}'
        runner.save(path/'metrics.json',dict(views=[row]))
        Image.fromarray(np.concatenate([source,np.full_like(source,shade)],axis=1)).save(path/'000_img.png')
        runner.save(runner.OUT/f'train/scene_{total}/splatfacto/continued/continuation_runtime.json',dict(seconds=60))
    runner.finalize()
    sheet = Image.open(runner.OUT/'comparisons/000.jpg')
    assert sheet.size == (64,16)
    # JPEG values within uniform column interiors retain their intended order.
    assert [sheet.getpixel((x,8))[0] for x in [8,24,40,56]] == [100,110,120,130]
    assert set(json.loads((runner.OUT/'comparisons.json').read_text())) == {'10000','15000'}
