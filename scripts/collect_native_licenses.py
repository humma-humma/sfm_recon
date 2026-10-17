"""Preserve conda dependency license texts and provenance before wheel repair."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--pybind-license", type=Path,
        help="License bundle for a non-conda pybind11 header provider (for example PyTorch)")
    args = parser.parse_args()
    records = {r["name"]: r for p in (args.prefix/"conda-meta").glob("*.json")
        for r in [json.loads(p.read_text())]}
    destination = Path(__file__).resolve().parents[1]/"src/sfm_reconstruction/_third_party_licenses"
    destination.mkdir(exist_ok=True)
    pending, seen, notices = ["ceres-solver", "eigen", "pybind11"], set(), []
    while pending:
        name = pending.pop()
        if name in seen or name not in records:
            continue
        seen.add(name)
        record = records[name]
        pending.extend(d.split()[0] for d in record.get("depends", []))
        cached = Path(record.get("link", {}).get("source", ""))/"info/licenses"
        copied = []
        if cached.is_dir():
            for source in cached.rglob("*"):
                if source.is_file():
                    target = destination/name/source.relative_to(cached)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    copied.append(target.relative_to(destination).as_posix())
        notices.append({"name": name, "version": record["version"],
            "build": record["build"], "license": record.get("license"),
            "package_url": record.get("url"), "license_files": copied})
    if args.pybind_license:
        target = destination/"pybind11-header-provider/LICENSE"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(args.pybind_license, target)
        notices.append({"name":"pybind11-header-provider", "license_files":[target.relative_to(destination).as_posix()],
            "source":args.pybind_license.name})
    (destination/"dependencies.json").write_text(json.dumps(notices, indent=2), encoding="utf-8")
    for required in ("ceres-solver", "eigen", "pybind11"):
        if not any(r["name"].startswith(required) and r["license_files"] for r in notices):
            raise RuntimeError(f"Missing license texts for {required}; inspect the conda package cache")
    print(f"Collected notices for {len(notices)} dependency packages")


if __name__ == "__main__":
    main()
