"""Camera interpolation and diagnostic proxies for bounded adaptive rendering."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def midpoint(a,b):
    rotations=np.array([a['r'],b['r']])
    r=Slerp([0,1],Rotation.from_matrix(rotations))([.5]).as_matrix()[0]
    centers=[-np.array(c['r']).T@np.array(c['t']) for c in (a,b)]
    time=(a['source_time']+b['source_time'])/2
    return dict(id=f'adaptive_{time:.9f}',index=a['index'],image=a['image'],
        r=r.tolist(),t=(-r@np.mean(centers,axis=0)).tolist(),k=a['k'],
        kind='motion',zoom=1.,source_time=time)


def rotation_degrees(a,b):
    return float(np.degrees(Rotation.from_matrix(np.array(b['r'])@np.array(a['r']).T).magnitude()))


def motion_proxy(a,b):
    """Cheap bidirectional-flow midpoint approximation; not the FFmpeg renderer."""
    gray=[cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) for image in (a,b)]
    flows=[cv2.calcOpticalFlowFarneback(gray[i],gray[1-i],None,.5,3,15,3,5,1.2,0) for i in (0,1)]
    y,x=np.mgrid[:a.shape[0],:a.shape[1]].astype(np.float32)
    warped=[];valid=np.ones(a.shape[:2],dtype=bool)
    for image,flow in zip((a,b),flows):
        mx=x-.5*flow[:,:,0];my=y-.5*flow[:,:,1]
        valid &= (mx>=0)&(mx<a.shape[1]-1)&(my>=0)&(my<a.shape[0]-1)
        warped.append(cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(float)/255)
    return (warped[0]+warped[1])/2,valid,float(np.quantile(np.linalg.norm(flows[0],axis=2),.9))


def needs_refinement(angle,error,valid_fraction,source_churn):
    return angle>3. or error>.035 or valid_fraction<.90 or source_churn>.60


def sparse_source_support(dataset,case,indices):
    """Fraction of visible sparse points inside at least two eligible sources.

    Frustum support only: this does not test occlusion or dense surface coverage.
    """
    def visible(points,r,t):
        xyz=points@np.array(r).T+np.array(t);z=xyz[:,2]
        xy=xyz[:,:2]/np.maximum(z[:,None],.001)
        k=dataset.k
        return (z>.01)&(np.abs(xy[:,0]*k[0]+k[2]-k[4]/2)<k[4]/2)&(np.abs(xy[:,1]*k[1]+k[3]-k[5]/2)<k[5]/2)
    points=dataset.points[visible(dataset.points,case['r'],case['t'])]
    if not len(points):return 0.
    points=points[np.linspace(0,len(points)-1,min(len(points),512)).astype(int)]
    count=np.zeros(len(points),dtype=int)
    for i in indices:count+=visible(points,dataset.poses[i][1],dataset.poses[i][2])
    return float(np.mean(count>=2))
