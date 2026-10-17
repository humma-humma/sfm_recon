"""Clone a saved training recipe, replacing only the reconstruction inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    import yaml
    import torch
    from nerfstudio.scripts.train import main as train

    config = yaml.load(args.baseline_config.read_text(), Loader=yaml.Loader)
    config.data = args.data.resolve()
    config.pipeline.datamanager.data = config.data
    config.pipeline.datamanager.dataparser.data = config.data
    config.output_dir = Path("/workspace/outputs/splat_refresh_v1/train")
    config.experiment_name = args.name
    config.timestamp = "probe_1000" if args.probe else f"run_{args.iterations}"
    config.max_num_iterations = args.iterations
    config.steps_per_save = min(1000, args.iterations)
    config.vis = "tensorboard"
    config.load_config = config.load_dir = config.load_checkpoint = config.load_step = None
    assert config.pipeline.model.camera_optimizer.mode == "off"
    assert config.pipeline.datamanager.dataparser.eval_mode == "interval"
    assert config.pipeline.datamanager.dataparser.eval_interval == 8
    checkpoint = config.get_checkpoint_dir() / f"step-{args.iterations-1:09d}.ckpt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["step"] != args.iterations - 1:
            raise ValueError("Existing checkpoint has the wrong step")
        print(f"Already trained: {checkpoint}")
        return
    start = perf_counter()
    train(config)
    if not checkpoint.exists():
        raise RuntimeError(f"Training ended without the expected checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key, value in state["pipeline"].items():
        if key.startswith("_model.gauss_params.") and (not len(value) or not torch.isfinite(value).all()):
            raise RuntimeError(f"Training produced empty or nonfinite Gaussian parameters: {key}")
    (checkpoint.parent.parent / "refresh_runtime.json").write_text(json.dumps({
        "seconds": perf_counter()-start, "iterations": args.iterations,
        "baseline_config": str(args.baseline_config), "data": str(config.data),
        "seed": config.machine.seed, "fresh_start": True, "probe": args.probe,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated()/2**20}, indent=2))


if __name__ == "__main__":
    main()
