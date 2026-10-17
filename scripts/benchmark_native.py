"""Compare optional kernels on identical inputs, excluding data setup from timing."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

import cv2
import numpy as np
import scipy

from sfm_reconstruction.bundle_adjustment import bundle_adjust
from sfm_reconstruction.dataset import load_dataset
from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.geometry import project_points
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct, _filter_points
from sfm_reconstruction.io import write_camera_parameters, write_point_cloud, write_summary


def synthetic_bundle():
    rng = np.random.default_rng(42)
    k = np.array([[700., 0., 320.], [0., 710., 240.], [0., 0., 1.]])
    pts = np.column_stack((rng.uniform(-1, 1, (300, 2)), rng.uniform(4, 7, 300)))
    true = {i: Pose(cv2.Rodrigues(np.array([.02*i, -.03*i, .01*i]))[0], [-.3*i, .01*i, .02*i]) for i in range(8)}
    poses = {i: p if i < 2 else Pose(cv2.Rodrigues(cv2.Rodrigues(p.rotation)[0].ravel()+rng.normal(0,.002,3))[0], p.translation.ravel()+rng.normal(0,.01,3)) for i,p in true.items()}
    pixels = {i: project_points(pts,p,k)+rng.normal(0,.1,(len(pts),2)) for i,p in true.items()}
    tracks = [Track({i:pixels[i][j] for i in poses}) for j in range(len(pts))]
    points = {i:p+rng.normal(0,.01,3) for i,p in enumerate(pts)}
    return k, tracks, poses, points, {0,1}


def measure(call, repeats):
    times = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        times.append(perf_counter()-start)
        print(f"  repeat {len(times)}: {times[-1]:.3f} s", flush=True)
    return result, {"seconds": times, "median_seconds": median(times)}


def reprojection_metrics(k, tracks, poses, points):
    errors = []
    for p, xyz in points.items():
        for c, xy in tracks[p].observations.items():
            if c in poses:
                errors.append(project_points(xyz.reshape(1,3),poses[c],k)[0]-xy)
    residual = np.asarray(errors)
    return {"reprojection_rmse_px": float(np.sqrt(np.mean(residual**2))),
            "soft_l1_cost": float(np.sum(4*(np.sqrt(1+(residual/2)**2)-1)))}


def benchmark_bundle(args):
    dataset = result = config = None
    if args.dataset:
        dataset = load_dataset(args.dataset, correspondence_dir=args.correspondences)
        config = ReconstructionConfig(bundle_adjustment=False, min_track_observations=args.min_track_observations,
                                      max_point_distance_factor=args.max_point_distance_factor)
        print("Building shared pre-BA input...", flush=True)
        result = reconstruct(dataset, config)
        k, tracks, poses, points, fixed = dataset.intrinsics, result.tracks, result.poses, result.points, set(result.initial_pair)
    else:
        k, tracks, poses, points, fixed = synthetic_bundle()
    payload = {"candidate":"bundle_adjustment", "cameras":len(poses), "points":len(points), "max_nfev":args.max_nfev,
               "initial":reprojection_metrics(k,tracks,poses,points), "backends":{}}
    solutions = {}
    for backend in (("python", "cpp", "ceres") if args.include_ceres else ("python", "cpp")):
        report = {}
        print(f"Bundle adjustment: {backend}",flush=True)
        solution, timing = measure(lambda: bundle_adjust(k,tracks,poses,points,fixed,max_nfev=args.max_nfev,backend=backend,ceres_max_iterations=args.ceres_max_iterations,solver_report=report),args.repeats)
        solutions[backend] = solution
        record = {**timing, **reprojection_metrics(k,tracks,*solution), "solver":dict(report)}
        if dataset:
            filtered = _filter_points(k,tracks,solution[0],solution[1],config)
            final = replace(result,poses=solution[0],points=filtered)
            dest=args.output.parent/backend
            dest.mkdir(parents=True,exist_ok=True)
            write_camera_parameters(dest/"estimated_camera_parameters.json",dataset,final)
            write_point_cloud(dest/"estimated_points.ply",final)
            write_summary(dest/"summary.json",dataset,final)
            metrics=evaluate_poses(solution[0],dataset.ground_truth_extrinsics)
            record["pose_metrics"]=metrics.to_dict() if metrics else None
            record["filtered_points"]=len(filtered)
            from sfm_reconstruction.stage1_evaluation import _load_vertices,evaluate_stage1_points
            if (dataset.root/"gt_points.ply").is_file():
                record["point_metrics"]=asdict(evaluate_stage1_points(_load_vertices(dataset.root/"gt_points.ply"),np.asarray(list(filtered.values()))))
            elif (dataset.root/"gt_mesh.ply").is_file() and metrics:
                from sfm_reconstruction.stage2_evaluation import evaluate_mesh_proxy
                record["mesh_proxy"]=asdict(evaluate_mesh_proxy(_load_vertices(dataset.root/"gt_mesh.ply"),np.asarray(list(filtered.values())),metrics.scale))
        payload["backends"][backend]=record
    a,b=solutions["python"],solutions["cpp"]
    payload["max_camera_matrix_difference"]=float(max(np.max(np.abs(a[0][i].matrix()-b[0][i].matrix())) for i in poses))
    payload["max_point_difference"]=float(max(np.linalg.norm(a[1][i]-b[1][i]) for i in points))
    return payload


def benchmark_graph(args):
    from sfm_reconstruction.pose_graph import (
        PoseGraph, PoseGraphNode, PoseGraphEdge, PoseGraphConfig,
        _estimate_loop_candidate, build_keyframe_pose_graph,
        optimize_pose_graph, expand_keyframe_optimization,
    )
    from sfm_reconstruction.stage3 import (
        load_stage3_dataset, load_trajectory, write_trajectory,
        Stage3OdometryConfig, _transform_from_trajectory_pose,
    )
    config=PoseGraphConfig(keyframe_stride=8,optimizer_max_nfev=args.max_nfev)
    raw=dataset=None
    setup={}
    if args.dataset:
        if not args.trajectory:
            raise ValueError("graph dataset benchmark requires --trajectory")
        dataset=load_stage3_dataset(args.dataset)
        raw=load_trajectory(args.trajectory)
        print("Verifying the documented 4387 -> 32 loop once...",flush=True)
        loop=_estimate_loop_candidate(dataset,4387,32,Stage3OdometryConfig(),config,{})
        if not loop.accepted:
            raise ValueError(f"documented loop failed verification: {loop.reject_reason}")
        setup={"loop_source":loop.source_index,"loop_target":loop.target_index,"loop_inliers":loop.inliers,"loop_median_error_px":loop.median_reprojection_error,"trajectory":str(args.trajectory)}
        graph=build_keyframe_pose_graph(raw,[loop],config)
    else:
        rng=np.random.default_rng(14)
        true=[]
        for i in range(100):
            angle=2*np.pi*i/99
            transform=np.eye(4)
            transform[:3,:3]=cv2.Rodrigues(np.array([0.,0.,.2*np.sin(angle)]))[0]
            transform[:3,3]=[2*np.sin(angle),2*(1-np.cos(angle)),.1*np.sin(2*angle)]
            true.append(transform)
        initial=[transform.copy() for transform in true]
        for i in range(1,len(initial)):
            initial[i][:3,3]+=rng.normal(0,.03,3)+np.array([.002*i,0,0])
            initial[i][:3,:3]=cv2.Rodrigues(rng.normal(0,.01,3))[0]@initial[i][:3,:3]
        nodes=[PoseGraphNode(i,float(i),t) for i,t in enumerate(initial)]
        edges=[PoseGraphEdge(i,i+1,np.linalg.inv(true[i+1])@true[i],1.,1.,50,"odometry") for i in range(99)]
        edges.append(PoseGraphEdge(0,99,np.linalg.inv(true[99])@true[0],.5,.5,50,"loop"))
        graph=PoseGraph(nodes,edges)
    payload={"candidate":"pose_graph","nodes":len(graph.nodes),"edges":len(graph.edges),"max_nfev":args.max_nfev,"setup":setup,"backends":{}}
    # Preserve the exact shared graph independently of feature RANSAC or future code.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output.parent/"graph_input.npz",node_indices=[n.index for n in graph.nodes],timestamps=[n.timestamp for n in graph.nodes],transforms=[n.camera_to_world for n in graph.nodes],edge_indices=[(e.source_index,e.target_index) for e in graph.edges],measured=[e.measured_target_from_source for e in graph.edges],weights=[(e.rotation_weight,e.translation_weight) for e in graph.edges])
    solutions={}
    for backend in ("python","cpp"):
        print(f"Pose graph: {backend}",flush=True)
        solution,timing=measure(lambda:optimize_pose_graph(graph,replace(config,optimizer_backend=backend)),args.repeats)
        solutions[backend]=solution
        record={**timing,"initial_cost":solution.initial_cost,"final_cost":solution.final_cost,"nfev":solution.iterations,"success":solution.success,"message":solution.message}
        poses=expand_keyframe_optimization(raw,solution.poses) if raw else solution.poses
        dest=args.output.parent/backend
        dest.mkdir(parents=True,exist_ok=True)
        trajectory_path=dest/"estimated_camera_trajectory_pose_graph.txt"
        write_trajectory(trajectory_path,poses)
        if dataset:
            from sfm_reconstruction.official_eval import run_official_eval
            evaluation=run_official_eval(dataset.ground_truth_trajectory_path,trajectory_path,dest,label=backend,evaluator_dir=dataset.root)
            if evaluation.returncode!=0 or evaluation.rmse is None:
                raise RuntimeError(f"official evaluation failed: {evaluation.stdout_path}")
            record["official_ate"]=evaluation.rmse
            record["evaluated_pose_pairs"]=evaluation.compared_pose_pairs
        payload["backends"][backend]=record
    a,b=solutions["python"],solutions["cpp"]
    payload["max_pose_matrix_difference"]=float(max(np.max(np.abs(_transform_from_trajectory_pose(p)-_transform_from_trajectory_pose(q))) for p,q in zip(a.poses,b.poses)))
    return payload


def benchmark_geometry(args):
    from sfm_reconstruction.geometry import reprojection_errors, points_in_front, triangulation_angles
    rng=np.random.default_rng(15)
    k=np.array([[700.,3.,320.],[0.,710.,240.],[0.,0.,1.]])
    first=Pose(cv2.Rodrigues(np.array([.1,-.2,.3]))[0],[.1,-.2,.05])
    second=Pose(np.eye(3),[-1.,0.,0.])
    scenarios=[]
    for count in (1,100,10000,100000):
        points=rng.normal(size=(count,3))
        points[:,2]+=8
        pixels=project_points(points,first,k)+.1
        loops=1000 if count==1 else 100 if count==100 else 10
        kernels={
            "projection":lambda backend:project_points(points,first,k,backend=backend),
            "reprojection":lambda backend:reprojection_errors(points,pixels,first,k,backend=backend),
            "cheirality":lambda backend:points_in_front(points,first,backend=backend),
            "angles":lambda backend:triangulation_angles(points,first,second,backend=backend),
        }
        for name,kernel in kernels.items():
            a,b=kernel("python"),kernel("cpp")
            np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9)
            record={"kernel":name,"points":count,"calls_per_sample":loops,"max_absolute_difference":float(np.max(np.abs(a.astype(float)-b.astype(float))))}
            for backend in ("python","cpp"):
                times=[]
                for _ in range(args.repeats):
                    start=perf_counter()
                    for _ in range(loops):
                        kernel(backend)
                    times.append((perf_counter()-start)/loops)
                record[backend]={"seconds":times,"median_seconds":median(times)}
            record["speedup"]=record["python"]["median_seconds"]/record["cpp"]["median_seconds"]
            scenarios.append(record)
    payload={"candidate":"geometry","scenarios":scenarios,"backends":{}}
    if args.dataset:
        dataset=load_dataset(args.dataset,correspondence_dir=args.correspondences)
        config=ReconstructionConfig(bundle_adjustment=False,min_track_observations=args.min_track_observations,max_point_distance_factor=args.max_point_distance_factor)
        print("Building shared reconstruction for geometry filtering...",flush=True)
        result=reconstruct(dataset,config)
        selections={}
        for backend in ("python","cpp"):
            print(f"Geometry filter: {backend}",flush=True)
            selected,timing=measure(lambda:_filter_points(dataset.intrinsics,result.tracks,result.poses,result.points,replace(config,geometry_backend=backend)),args.repeats)
            selections[backend]=selected
            payload["backends"][backend]={**timing,"retained_points":len(selected)}
        if set(selections["python"])!=set(selections["cpp"]):
            raise AssertionError("geometry backends changed accepted point IDs")
        payload["identical_selected_point_ids"]=True
        payload["timed_scope"]="complete reconstruction point filtering, including scalar binding overhead"
    else:
        # Summary timing is the sum of the four independently measured large kernels.
        for backend in ("python","cpp"):
            payload["backends"][backend]={"median_seconds":sum(r[backend]["median_seconds"] for r in scenarios if r["points"]==100000)}
        payload["timed_scope"]="sum of four kernels on 100000 points; see individual scenarios"
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",choices=("bundle","graph","geometry"),default="bundle")
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--correspondences",type=Path)
    parser.add_argument("--trajectory",type=Path)
    parser.add_argument("--min-track-observations",type=int,default=2)
    parser.add_argument("--max-point-distance-factor",type=float)
    parser.add_argument("--max-nfev",type=int,default=20)
    parser.add_argument("--include-ceres", action="store_true")
    parser.add_argument("--ceres-max-iterations", type=int, default=100)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.include_ceres and args.candidate != "bundle":
        parser.error("--include-ceres applies only to bundle adjustment")
    if args.repeats<1 or args.max_nfev<1 or args.ceres_max_iterations<1:
        parser.error("repeats and max-nfev must be positive")
    from sfm_reconstruction.native_backend import load_native
    load_native()
    payload={"bundle":benchmark_bundle,"graph":benchmark_graph,"geometry":benchmark_geometry}[args.candidate](args)
    if args.include_ceres:
        from sfm_reconstruction.native_backend import load_ceres
        payload["ceres_version"] = load_ceres().version
        payload["ceres_max_iterations"] = args.ceres_max_iterations
        payload["ceres_speedup"] = payload["backends"]["python"]["median_seconds"] / payload["backends"]["ceres"]["median_seconds"]
    payload["environment"]={"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"numpy":np.__version__,"scipy":scipy.__version__,"opencv":cv2.__version__,"implementation":"Python/cpp use SciPy TRF; optional ceres uses single-thread DENSE_SCHUR"}
    payload["speedup"]=payload["backends"]["python"]["median_seconds"]/payload["backends"]["cpp"]["median_seconds"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":
    main()
