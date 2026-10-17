"""Opt-in CMake extensions; ordinary builds remain pure Python."""
import os
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def run(self):
        destination = Path(self.get_ext_fullpath(self.extensions[0].name)).resolve().parent
        build = Path(self.build_temp).resolve() / "cmake"
        cmake = os.environ.get("SFM_CMAKE", "cmake")
        args = [cmake, "-S", str(Path(__file__).resolve().parent/"native"),
            "-B", str(build), f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DSFM_WITH_CERES={'ON' if len(self.extensions) == 2 else 'OFF'}"]
        prefix = os.environ.get("SFM_CERES_PREFIX")
        headers = os.environ.get("PYBIND11_INCLUDE_DIR")
        if headers:
            args.append(f"-DPYBIND11_INCLUDE_DIR={Path(headers).resolve()}")
        if prefix:
            prefix = Path(prefix).resolve()
            if (prefix/"Library").is_dir():
                prefix = prefix/"Library"
            args.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        subprocess.run(args, check=True)
        command = [cmake, "--build", str(build), "--config", "Release", "--parallel",
            "1" if os.name == "nt" else "2"]
        if "CMAKE_GENERATOR:INTERNAL=Visual Studio" in (build/"CMakeCache.txt").read_text():
            command += ["--", "/p:TrackFileAccess=false"]
        subprocess.run(command, check=True)
        subprocess.run([cmake, "--install", str(build), "--config", "Release",
            "--prefix", str(destination)], check=True)


native = os.environ.get("SFM_BUILD_NATIVE") == "1"
ceres = os.environ.get("SFM_WITH_CERES") == "1"
if ceres and not native:
    raise RuntimeError("SFM_WITH_CERES=1 also requires SFM_BUILD_NATIVE=1")
extensions = [Extension("sfm_reconstruction._native", sources=[])] if native else []
if ceres:
    extensions.append(Extension("sfm_reconstruction._ceres", sources=[]))
mode = "native-ceres" if ceres else "native" if native else "pure"
setup(ext_modules=extensions, cmdclass={"build_ext": CMakeBuild},
    options={"build": {"build_base": f"build/{mode}"}},
    exclude_package_data={} if native else {"sfm_reconstruction": ["_third_party_licenses/*"]})
