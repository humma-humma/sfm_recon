"""Time-local comparisons for the GNT-to-Gaussian transfer diagnosis."""
import numpy as np


def alternating_unseen(rows):
    """Split chronologically ordered dense poses after excluding fitted times."""
    unseen=sorted((r for r in rows if not r['exact_training_time']),key=lambda r:r['source_time'])
    indices=[r['index'] for r in unseen]
    if len(indices)!=len(set(indices)):raise ValueError('Duplicate dense indices')
    return indices[::2],indices[1::2]


def bracket_training(times,time):
    times=np.asarray(times,dtype=float)
    if len(times)<2 or not np.all(np.diff(times)>0):
        raise ValueError('Training timestamps must be strictly increasing')
    if time<times[0]-1e-4 or time>times[-1]+1e-4:
        raise ValueError('Novel time outside training route')
    nearest=int(np.argmin(abs(times-time)))
    # Timestamp rounding can put the dense grid microseconds away from a fitted
    # camera. Treat views within 0.1 ms as fitted, rather than novel evidence.
    if abs(times[nearest]-time)<1e-4:return nearest,nearest,0.
    left=int(np.searchsorted(times,time)-1)
    return left,left+1,float((time-times[left])/(times[left+1]-times[left]))


def local_gap(training,left,right,fraction,novel):
    """Negative dB means the novel view fits worse than its nearby endpoints.

    Endpoint MSE is interpolated, rather than averaging logarithmic PSNRs.
    This is a local diagnostic, not a controlled perceptual/geometric metric.
    """
    mse=(1-fraction)*training[left]['mse']+fraction*training[right]['mse']
    expected=-10*np.log10(max(mse,1e-16))
    return dict(endpoint_psnr=float(expected),psnr_gap=float(novel['psnr']-expected))
