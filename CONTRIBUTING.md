# Contributing

Contributions that improve correctness, reproducibility, diagnostics, or
performance are welcome.

## Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,evaluation,visualization]"
```

## Before opening a pull request

```bash
python -m pytest -q -p no:cacheprovider
python -m build
```

Please:

- Add tests for geometry or behavior changes.
- Keep optional learned, Open3D, and GPU dependencies behind existing extras.
- Document coordinate conventions and whether transforms are camera-to-world or
  world-to-camera.
- Do not commit datasets, outputs, model weights, checkpoints, or source video
  frames.
- Distinguish metric improvements from qualitative presentation changes.

Bug reports should include the command, minimal dataset layout, Python version,
platform, traceback, and whether optional extras were installed.
