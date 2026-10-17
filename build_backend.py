"""PEP 517 hooks: request binding headers only for opt-in native builds."""
import importlib.util
import os
from setuptools.build_meta import *  # noqa: F403 - forward standard backend hooks
from setuptools import build_meta as _backend


def get_requires_for_build_wheel(config_settings=None):
    requirements = _backend.get_requires_for_build_wheel(config_settings)
    if os.environ.get("SFM_BUILD_NATIVE") == "1" and not os.environ.get("PYBIND11_INCLUDE_DIR") and not any(
        importlib.util.find_spec(name) for name in ("pybind11", "torch")
    ):
        requirements.append("pybind11>=2.12")
    return requirements


get_requires_for_build_editable = get_requires_for_build_wheel
