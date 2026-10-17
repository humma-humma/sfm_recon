"""Profile gsplat forward/backward on saved splats; never update parameters.

Run inside the existing Nerfstudio container with /workspace mounted. This is
an isolated rasterizer benchmark, not an end-to-end training benchmark.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--colmap-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "status": "running", "training_updates": 0}

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    try:
        from gsplat.rendering import rasterization
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        report["gsplat"] = importlib.metadata.version("gsplat")
    except (ImportError, RuntimeError) as error:
        report.update(status="blocked", reason=str(error))
        save()
        print(json.dumps(report, indent=2))
        return 2

    checkpoint_path = sorted((args.run_dir / "nerfstudio_models").glob("*.ckpt"))[-1]
    params = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["pipeline"]
    device = torch.device("cuda")
    values = {name: params[f"_model.gauss_params.{name}"].to(device).detach().requires_grad_()
              for name in ["means", "quats", "scales", "opacities", "features_dc", "features_rest"]}
    record = json.loads((args.run_dir / "dataparser_transforms.json").read_text())
    transform, scale = np.asarray(record["transform"]), float(record["scale"])
    k, poses, _ = _read_colmap(args.colmap_dir)
    K = torch.tensor([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], device=device, dtype=torch.float32)[None]
    selected_views = sorted({0, (len(poses)//16)*8, ((len(poses)-1)//8)*8})
    degree = int(np.sqrt(1 + values["features_rest"].shape[1])) - 1
    report.update(checkpoint=str(checkpoint_path), gaussians=len(values["means"]),
                  device=torch.cuda.get_device_name(), views=selected_views,
                  benchmark="rasterizer forward/backward only; sparse gradients require a compatible optimizer in training",
                  results=[])
    for index in selected_views:
        image, r, t = poses[index]
        # Convert the original COLMAP world-to-camera pose into the normalized
        # coordinate system used by the checkpoint's means and scales.
        rotation = r @ transform[:, :3].T
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = scale * (t - rotation @ transform[:, 3])
        view = torch.tensor(view, dtype=torch.float32, device=device)[None]
        reference = reference_gradient = None
        for packed, sparse in [(False, False), (True, False), (True, True)]:
            def run():
                for value in values.values():
                    value.grad = None
                colors = torch.cat([values["features_dc"][:, None], values["features_rest"]], dim=1)
                rgb, alpha, _ = rasterization(
                    means=values["means"], quats=values["quats"],
                    scales=values["scales"].exp(), opacities=values["opacities"].sigmoid().flatten(),
                    colors=colors, viewmats=view, Ks=K, width=k[4], height=k[5],
                    sh_degree=degree, packed=packed, sparse_grad=sparse,
                    rasterize_mode="classic", absgrad=False,
                )
                (rgb.square().mean() + .01*alpha.mean()).backward()
                return rgb.detach()
            try:
                for _ in range(3):
                    rendered = run()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                samples = []
                for _ in range(10):
                    start = perf_counter()
                    rendered = run()
                    torch.cuda.synchronize()
                    samples.append(perf_counter()-start)
                peak = torch.cuda.max_memory_allocated()
                grad = values["means"].grad
                grad = grad.to_dense() if grad.is_sparse else grad
                if reference is None:
                    reference = rendered.clone()
                    reference_gradient = grad.clone()
                item = dict(image=image, packed=packed, sparse_grad=sparse, seconds=samples,
                            median_seconds=float(np.median(samples)),
                            peak_allocated_mb=peak/2**20,
                            incremental_peak_mb=(peak-allocated)/2**20,
                            rgb_max_difference=float((rendered-reference).abs().max()),
                            means_gradient_max_difference=float((grad-reference_gradient).abs().max()))
                report["results"].append(item)
                print(json.dumps(item), flush=True)
                save()
            except torch.cuda.OutOfMemoryError as error:
                report["results"].append(dict(image=image, packed=packed, sparse_grad=sparse,
                                               status="out_of_memory", reason=str(error)))
                for value in values.values():
                    value.grad = None
                torch.cuda.empty_cache()
                save()
    report["status"] = "complete"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
