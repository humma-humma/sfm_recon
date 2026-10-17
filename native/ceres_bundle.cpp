#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices =
    py::array_t<long long, py::array::c_style | py::array::forcecast>;

struct CoordinateError {
  std::array<double, 9> k;
  double observed;
  int coordinate;
  template <typename T>
  bool operator()(const T *camera, const T *point, T *error) const {
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    for (int j = 0; j < 3; ++j)
      p[j] += camera[j + 3];
    T numerator = T(0), denominator = T(0);
    for (int j = 0; j < 3; ++j) {
      numerator += T(k[3 * coordinate + j]) * p[j];
      denominator += T(k[6 + j]) * p[j];
    }
    error[0] = numerator / denominator - T(observed);
    return true;
  }
};

py::tuple solve(Array cameras_in, Array points_in, Indices observations,
                Array pixels, Array intrinsics, Indices fixed,
                int max_iterations, double loss_scale, int num_threads,
                const std::string &linear_solver) {
  ceres::LinearSolverType solver_type;
  if (num_threads < 1)
    throw py::value_error("num_threads must be positive");
  if (linear_solver == "dense_schur") solver_type = ceres::DENSE_SCHUR;
  else if (linear_solver == "sparse_schur") solver_type = ceres::SPARSE_SCHUR;
  else if (linear_solver == "iterative_schur") solver_type = ceres::ITERATIVE_SCHUR;
  else throw py::value_error("Unknown Ceres linear solver");
  if (solver_type == ceres::SPARSE_SCHUR &&
      !ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    throw py::value_error("sparse_schur requires Ceres built with Eigen sparse support");
  if (cameras_in.ndim() != 2 || cameras_in.shape(1) != 6 ||
      points_in.ndim() != 2 || points_in.shape(1) != 3 ||
      observations.ndim() != 2 || observations.shape(1) != 2 ||
      pixels.ndim() != 2 || pixels.shape(1) != 2 ||
      observations.shape(0) != pixels.shape(0) || intrinsics.ndim() != 2 ||
      intrinsics.shape(0) != 3 || intrinsics.shape(1) != 3 ||
      fixed.ndim() != 1 || observations.shape(0) == 0 || max_iterations < 1 ||
      !std::isfinite(loss_scale) || loss_scale <= 0)
    throw py::value_error("Invalid Ceres bundle inputs");
  Array cameras({cameras_in.shape(0), py::ssize_t(6)}),
      points({points_in.shape(0), py::ssize_t(3)});
  std::copy(cameras_in.data(), cameras_in.data() + cameras_in.size(),
            cameras.mutable_data());
  std::copy(points_in.data(), points_in.data() + points_in.size(),
            points.mutable_data());
  std::array<double, 9> k;
  std::copy(intrinsics.data(), intrinsics.data() + 9, k.begin());
  auto obs = observations.unchecked<2>();
  auto xy = pixels.unchecked<2>();
  auto fixed_ids = fixed.unchecked<1>();
  for (py::ssize_t i = 0; i < observations.shape(0); ++i)
    if (obs(i, 0) < 0 || obs(i, 0) >= cameras.shape(0) || obs(i, 1) < 0 ||
        obs(i, 1) >= points.shape(0))
      throw py::value_error("Observation index out of range");
  for (py::ssize_t i = 0; i < fixed.size(); ++i)
    if (fixed_ids(i) < 0 || fixed_ids(i) >= cameras.shape(0))
      throw py::value_error("Fixed camera index out of range");
  ceres::Solver::Summary summary;
  {
    py::gil_scoped_release release;
    ceres::Problem problem;
    auto *loss = new ceres::SoftLOneLoss(loss_scale);
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (py::ssize_t i = 0; i < observations.shape(0); ++i) {
      double *c = cameras.mutable_data() + 6 * obs(i, 0);
      double *p = points.mutable_data() + 3 * obs(i, 1);
      // Separate scalar blocks exactly match SciPy's coordinate-wise soft_l1.
      for (int axis = 0; axis < 2; ++axis)
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoordinateError, 1, 6, 3>(
                new CoordinateError{k, xy(i, axis), axis}),
            loss, c, p);
      ordering->AddElementToGroup(p, 0);
      ordering->AddElementToGroup(c, 1);
    }
    for (py::ssize_t i = 0; i < fixed.size(); ++i) {
      double *c = cameras.mutable_data() + 6 * fixed_ids(i);
      if (problem.HasParameterBlock(c))
        problem.SetParameterBlockConstant(c);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = solver_type;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.linear_solver_ordering = ordering;
    options.max_num_iterations = max_iterations;
    options.num_threads = num_threads;
    options.dense_linear_algebra_library_type = ceres::EIGEN;
    options.function_tolerance = 1e-8;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;
    ceres::Solve(options, &problem, &summary);
  }
  if (!summary.IsSolutionUsable())
    throw std::runtime_error(summary.BriefReport());
  py::dict report;
  report["initial_cost"] = summary.initial_cost;
  report["final_cost"] = summary.final_cost;
  report["iterations"] = summary.iterations.size();
  report["message"] = summary.BriefReport();
  report["converged"] = summary.termination_type == ceres::CONVERGENCE;
  report["solver_seconds"] = summary.total_time_in_seconds;
  report["linear_solver"] = ceres::LinearSolverTypeToString(summary.linear_solver_type_used);
  report["linear_solver_requested"] = linear_solver;
  report["num_threads"] = summary.num_threads_used;
  report["num_threads_requested"] = num_threads;
  report["loss_scale"] = loss_scale;
  return py::make_tuple(cameras, points, report);
}
PYBIND11_MODULE(_ceres, m) {
  m.attr("version") = CERES_VERSION_STRING;
  m.def("bundle_adjust", &solve, py::arg("cameras"), py::arg("points"),
        py::arg("observations"), py::arg("pixels"), py::arg("intrinsics"),
        py::arg("fixed"), py::arg("max_iterations"), py::arg("loss_scale") = 2.0,
        py::arg("num_threads") = 1, py::arg("linear_solver") = "dense_schur");
}
