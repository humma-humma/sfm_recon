"""Follow the entire captured camera route with novel midpoint cameras."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def dense_route_cases(cases,start_seconds=46.,duration=6.,fps=30,hold_end=False):
    """Sample the existing piecewise camera path on an exact video-time grid."""
    times=np.array([c['source_time'] for c in cases])
    if len(times)<2 or not np.all(np.diff(times)>0) or duration<=0 or fps<=0:
        raise ValueError('A chronological route and positive duration/rate are required')
    count=round(duration*fps)
    if count<1 or abs(count-duration*fps)>1e-8:
        raise ValueError('Duration must contain an integral number of frames')
    samples=times[0]+start_seconds+np.arange(count)/fps
    if samples[0]<times[0] or (samples[-1]>times[-1] and not hold_end):
        raise ValueError('Dense clip falls outside the existing route')
    samples=np.minimum(samples,times[-1])
    rotations=np.array([c['r'] for c in cases])
    centers=np.array([-np.array(c['r']).T@np.array(c['t']) for c in cases])
    matrices=Slerp(times,Rotation.from_matrix(rotations))(samples).as_matrix()
    result=[]
    for i,(time,r) in enumerate(zip(samples,matrices)):
        j=min(len(cases)-2,max(0,int(np.searchsorted(times,time,side='right')-1)))
        fraction=float((time-times[j])/(times[j+1]-times[j]))
        center=(1-fraction)*centers[j]+fraction*centers[j+1]
        result.append(dict(id=f'dense_{i:04}',index=cases[j]['index'],image=cases[j]['image'],
            r=r.tolist(),t=(-r@center).tolist(),k=cases[j]['k'],kind='motion',zoom=1.,
            source_time=float(time),clip_time=i/fps,route_time=start_seconds+i/fps,
            bracket=[j,j+1],fraction=fraction))
    return result


def object_route_cases(poses, k, count=60):
    """Uniform samples along the captured order, without inventing a closing orbit."""
    if len(poses)<2 or count<2:
        raise ValueError('At least two poses and samples are required')
    rotations=np.array([r for _,r,_ in poses])
    centers=np.array([-r.T@t for _,r,t in poses])
    positions=(np.arange(count)+.5)*(len(poses)-1)/count
    matrices=Slerp(np.arange(len(poses)),Rotation.from_matrix(rotations))(positions).as_matrix()
    cases=[]
    for j,(position,r) in enumerate(zip(positions,matrices)):
        i=int(position);fraction=float(position-i)
        center=(1-fraction)*centers[i]+fraction*centers[i+1]
        cases.append(dict(id=f'route_{j:04}',index=i,image=poses[i][0],
            r=r.tolist(),t=(-r@center).tolist(),k=list(k),kind='motion',zoom=1.,
            bracket=[i,i+1],fraction=fraction))
    return cases


def traversal_cases(poses,k):
    times = np.array([float(Path(name).stem) for name,_,_ in poses])
    if len(poses)<2 or not np.all(np.diff(times)>0):
        raise ValueError('Traversal requires at least two chronologically ordered camera poses')
    rotations = np.array([r for _,r,_ in poses])
    centers = np.array([-r.T@t for _,r,t in poses])
    sample_times = (times[:-1]+times[1:])/2
    interpolated = Slerp(times,Rotation.from_matrix(rotations))(sample_times).as_matrix()
    result = []
    for i,(time,r) in enumerate(zip(sample_times,interpolated)):
        center = (centers[i]+centers[i+1])/2
        result.append(dict(id=f'traversal_{i:04}',index=i,image=poses[i][0],
            r=r.tolist(),t=(-r@center).tolist(),k=list(k),kind='motion',zoom=1.,
            source_time=float(time),bracket=[i,i+1]))
    return result
