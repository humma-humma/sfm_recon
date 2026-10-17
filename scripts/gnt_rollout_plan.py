"""Playback-time route planning shared by all-four export and teacher generation."""
import numpy as np
from scipy.spatial.transform import Rotation
from gnt_traversal import object_route_cases,dense_route_cases
from gnt_transfer_metrics import bracket_training,alternating_unseen


def object_plan(poses,k):
    sparse=object_route_cases(poses,k)
    for i,case in enumerate(sparse):case['source_time']=i*8/15
    rotations=Rotation.from_matrix([c['r'] for c in sparse])
    angles=(rotations[:-1].inv()*rotations[1:]).magnitude()
    times=np.array([c['source_time'] for c in sparse])
    scores=[float(angles[(times[:-1]>=start)&(times[:-1]<start+6)].sum()) for start in range(26)]
    start=int(np.argmax(scores));dense=dense_route_cases(sparse,start_seconds=start,duration=6,fps=30)
    rows=[]
    for i,case in enumerate(dense):
        left,right,_=bracket_training(times,case['source_time'])
        rows.append(dict(index=i,source_time=case['source_time'],exact_training_time=left==right))
    added,withheld=alternating_unseen(rows)
    return dict(sparse=sparse,dense=dense,added=added,withheld=withheld,
                dense_start_seconds=start,duration_seconds=32.,fps=30,
                section_selection='Highest summed rotation over an integer-start six-second route window; fixed before rendering')
