"""Exact 15-view/s timing and conservative cached-camera matching."""
import numpy as np
from gnt_traversal import dense_route_cases


def scene15_cases(route):
    # Round to a whole number of 15fps frames: 2198 / 15 = 146.533... seconds.
    return dense_route_cases(route,start_seconds=0,duration=2198/15,fps=15,hold_end=True)


def matching_camera(case,candidates):
    return next((r for r in candidates if abs(r['case']['source_time']-case['source_time'])<1e-8
        and np.allclose(r['case']['r'],case['r'],atol=1e-8,rtol=0)
        and np.allclose(r['case']['t'],case['t'],atol=1e-8,rtol=0)
        and np.allclose(r['case']['k'],case['k'],atol=1e-8,rtol=0)),None)
