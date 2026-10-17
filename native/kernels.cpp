// Small fixed-size kernels. NumPy owns inputs/outputs; no Python calls in
// loops.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices =
    py::array_t<long long, py::array::c_style | py::array::forcecast>;
using V = std::array<double, 3>;
using M = std::array<double, 9>;
const M identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V sub(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V mv(const M &a, V b) {
  return {a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
          a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
          a[6] * b[0] + a[7] * b[1] + a[8] * b[2]};
}
M mm(const M &a, const M &b) {
  M c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return c;
}
M transpose(M a) {
  return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
}
M skew(V w) { return {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0}; }
double norm(V w) { return std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); }
M combination(double a, double b, V w) {
  M s = skew(w), ss = mm(s, s), r = identity;
  for (int i = 0; i < 9; ++i)
    r[i] += a * s[i] + b * ss[i];
  return r;
}
M rotation(V w) {
  double t = norm(w), t2 = t * t;
  return t < 1e-4 ? combination(1 - t2 / 6 + t2 * t2 / 120,
                                .5 - t2 / 24 + t2 * t2 / 720, w)
                  : combination(std::sin(t) / t, (1 - std::cos(t)) / t2, w);
}
M right_jacobian(V w) {
  double t = norm(w), t2 = t * t;
  return t < 1e-4 ? combination(-.5 + t2 / 24 - t2 * t2 / 720,
                                1. / 6 - t2 / 120 + t2 * t2 / 5040, w)
                  : combination(-(1 - std::cos(t)) / t2,
                                (t - std::sin(t)) / (t2 * t), w);
}
M inverse_jacobian(V w, bool right) {
  double t = norm(w), t2 = t * t;
  double b = t < 1e-4 ? 1. / 12 + t2 / 720 + t2 * t2 / 30240
                      : (1 - .5 * t / std::tan(.5 * t)) / t2;
  return combination(right ? .5 : -.5, b, w);
}
// Quaternion branch avoids the singular sin(theta) denominator near pi.
V logarithm(const M &r) {
  double qw, qx, qy, qz;
  double trace = r[0] + r[4] + r[8];
  if (trace > 0) {
    double s = 2 * std::sqrt(trace + 1);
    qw = s / 4;
    qx = (r[7] - r[5]) / s;
    qy = (r[2] - r[6]) / s;
    qz = (r[3] - r[1]) / s;
  } else if (r[0] > r[4] && r[0] > r[8]) {
    double s = 2 * std::sqrt(1 + r[0] - r[4] - r[8]);
    qw = (r[7] - r[5]) / s;
    qx = s / 4;
    qy = (r[1] + r[3]) / s;
    qz = (r[2] + r[6]) / s;
  } else if (r[4] > r[8]) {
    double s = 2 * std::sqrt(1 + r[4] - r[0] - r[8]);
    qw = (r[2] - r[6]) / s;
    qx = (r[1] + r[3]) / s;
    qy = s / 4;
    qz = (r[5] + r[7]) / s;
  } else {
    double s = 2 * std::sqrt(1 + r[8] - r[0] - r[4]);
    qw = (r[3] - r[1]) / s;
    qx = (r[2] + r[6]) / s;
    qy = (r[5] + r[7]) / s;
    qz = s / 4;
  }
  if (qw < 0) {
    qw = -qw;
    qx = -qx;
    qy = -qy;
    qz = -qz;
  }
  double n = std::sqrt(qx * qx + qy * qy + qz * qz),
         factor = n < 1e-12 ? 2 : 2 * std::atan2(n, qw) / n;
  return {factor * qx, factor * qy, factor * qz};
}
void shape(const py::array &a, std::initializer_list<py::ssize_t> dims) {
  if (a.ndim() != static_cast<int>(dims.size()))
    throw std::invalid_argument("invalid array rank");
  int i = 0;
  for (auto d : dims) {
    if (d >= 0 && a.shape(i) != d)
      throw std::invalid_argument("invalid array shape");
    ++i;
  }
}
V vector_at(const double *a) { return {a[0], a[1], a[2]}; }
M matrix_at(const double *a) {
  M m;
  std::copy(a, a + 9, m.begin());
  return m;
}

py::tuple ba_evaluate(Array values, Array cameras, Indices offsets,
                      Indices observations, Array pixels, Array intrinsics,
                      bool jacobian) {
  // values = variable [angle-axis, translation] blocks followed by XYZ blocks.
  // Observation rows contain a camera row and the XYZ offset in values.
  // Jacobian columns are [camera rotation (3), translation (3), point (3)].
  shape(values, {-1});
  shape(cameras, {-1, 6});
  shape(offsets, {cameras.shape(0)});
  shape(observations, {-1, 2});
  shape(pixels, {observations.shape(0), 2});
  shape(intrinsics, {3, 3});
  auto o = observations.unchecked<2>();
  auto off = offsets.unchecked<1>();
  const auto n = observations.shape(0);
  py::array_t<double> residual({n, py::ssize_t(2)}),
      derivatives({jacobian ? n : 0, py::ssize_t(2), py::ssize_t(9)});
  auto res = residual.mutable_unchecked<2>();
  auto d = derivatives.mutable_unchecked<3>();
  M k = matrix_at(intrinsics.data());
  // Check indexes before releasing the GIL and accessing raw buffers.
  for (py::ssize_t i = 0; i < n; ++i) {
    if (o(i, 0) < 0 || o(i, 0) >= cameras.shape(0) || o(i, 1) < 0 ||
        o(i, 1) + 3 > values.size())
      throw std::invalid_argument("invalid BA index");
    if (off(o(i, 0)) >= 0 && off(o(i, 0)) + 6 > values.size())
      throw std::invalid_argument("invalid camera offset");
  }
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      auto ci = o(i, 0), po = o(i, 1), co = off(ci);
      const double *c = co < 0 ? cameras.data() + 6 * ci : values.data() + co;
      V w = vector_at(c), point = vector_at(values.data() + po);
      M r = rotation(w);
      V camera = add(mv(r, point), vector_at(c + 3)), h = mv(k, camera);
      for (int row = 0; row < 2; ++row)
        res(i, row) = h[row] / h[2] - pixels.data()[2 * i + row];
      if (jacobian) {
        // d(R(w) X)/dw = -R [X]x J_right(w).
        M dr = mm(mm(r, skew(point)), right_jacobian(w));
        for (int row = 0; row < 2; ++row) {
          V projection{};
          for (int j = 0; j < 3; ++j)
            projection[j] = (k[3 * row + j] - h[row] / h[2] * k[6 + j]) / h[2];
          for (int j = 0; j < 3; ++j) {
            d(i, row, j) = 0;
            d(i, row, 6 + j) = 0;
            for (int z = 0; z < 3; ++z) {
              d(i, row, j) -= projection[z] * dr[3 * z + j];
              d(i, row, 6 + j) += projection[z] * r[3 * z + j];
            }
            d(i, row, 3 + j) = projection[j];
          }
        }
      }
    }
  }
  return py::make_tuple(residual, derivatives);
}

py::tuple graph_evaluate(Array values, Array initial, Indices offsets,
                         Indices edges, Array measured, Array weights,
                         bool jacobian) {
  // Camera-to-world poses. Error is M^-1 T_q^-1 T_p, exactly as in Python.
  // Jacobian columns are source (6) then target (6); fixed offsets are -1.
  shape(values, {-1});
  shape(initial, {-1, 6});
  shape(offsets, {initial.shape(0)});
  shape(edges, {-1, 2});
  auto n = edges.shape(0);
  shape(measured, {n, 4, 4});
  shape(weights, {n, 2});
  auto e = edges.unchecked<2>();
  auto off = offsets.unchecked<1>();
  auto meas = measured.unchecked<3>();
  auto wt = weights.unchecked<2>();
  py::array_t<double> residual({n, py::ssize_t(6)}),
      derivatives({jacobian ? n : 0, py::ssize_t(6), py::ssize_t(12)});
  auto res = residual.mutable_unchecked<2>();
  auto d = derivatives.mutable_unchecked<3>();
  for (py::ssize_t i = 0; i < n; ++i)
    for (int j = 0; j < 2; ++j) {
      if (e(i, j) < 0 || e(i, j) >= initial.shape(0))
        throw std::invalid_argument("invalid graph index");
      if (off(e(i, j)) >= 0 && off(e(i, j)) + 6 > values.size())
        throw std::invalid_argument("invalid graph offset");
    }
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      auto p = e(i, 0), q = e(i, 1);
      const double *cp =
          off(p) < 0 ? initial.data() + 6 * p : values.data() + off(p);
      const double *cq =
          off(q) < 0 ? initial.data() + 6 * q : values.data() + off(q);
      V wp = vector_at(cp), wq = vector_at(cq);
      M rp = rotation(wp), rqt = transpose(rotation(wq)), rm{};
      V tm{};
      for (int a = 0; a < 3; ++a) {
        tm[a] = meas(i, a, 3);
        for (int b = 0; b < 3; ++b)
          rm[3 * a + b] = meas(i, a, b);
      }
      M rmt = transpose(rm);
      V pred = mv(rqt, sub(vector_at(cp + 3), vector_at(cq + 3)));
      V er = logarithm(mm(mm(rmt, rqt), rp)), et = mv(rmt, sub(pred, tm));
      for (int a = 0; a < 3; ++a) {
        res(i, a) = wt(i, 0) * er[a];
        res(i, 3 + a) = wt(i, 1) * et[a];
      }
      if (jacobian) {
        // Differentiate the principal SO(3) log using right/left inverse
        // Jacobians; translation uses R_m^T R_q^T (t_p - t_q).
        M ap = mm(inverse_jacobian(er, true), right_jacobian(wp));
        M aq = mm(mm(inverse_jacobian(er, false), rmt), right_jacobian(wq));
        M bq = mm(mm(rmt, skew(pred)), right_jacobian(wq)), bt = mm(rmt, rqt);
        for (int a = 0; a < 6; ++a)
          for (int b = 0; b < 12; ++b)
            d(i, a, b) = 0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            res(i, a) = wt(i, 0) * er[a];
            d(i, a, b) = wt(i, 0) * ap[3 * a + b];
            d(i, a, 6 + b) = -wt(i, 0) * aq[3 * a + b];
            d(i, 3 + a, 3 + b) = wt(i, 1) * bt[3 * a + b];
            d(i, 3 + a, 6 + b) = wt(i, 1) * bq[3 * a + b];
            d(i, 3 + a, 9 + b) = -wt(i, 1) * bt[3 * a + b];
          }
      }
    }
  }
  return py::make_tuple(residual, derivatives);
}

Array project(Array points, Array rotation_array, Array translation,
              Array intrinsics) {
  shape(points, {-1, 3});
  shape(rotation_array, {3, 3});
  shape(translation, {3});
  shape(intrinsics, {3, 3});
  auto n = points.shape(0);
  py::array_t<double> output({n, py::ssize_t(2)});
  auto out = output.mutable_unchecked<2>();
  M r = matrix_at(rotation_array.data()), k = matrix_at(intrinsics.data());
  V t = vector_at(translation.data());
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      V h = mv(k, add(mv(r, vector_at(points.data() + 3 * i)), t));
      out(i, 0) = h[0] / h[2];
      out(i, 1) = h[1] / h[2];
    }
  }
  return output;
}
Array angles(Array points, Array first, Array second) {
  shape(points, {-1, 3});
  shape(first, {3});
  shape(second, {3});
  auto n = points.shape(0);
  py::array_t<double> output(n);
  auto out = output.mutable_unchecked<1>();
  V c1 = vector_at(first.data()), c2 = vector_at(second.data());
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      V p = vector_at(points.data() + 3 * i), a = sub(p, c1), b = sub(p, c2);
      double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
             cosine = dot / (norm(a) * norm(b));
      out(i) =
          std::acos(std::isnan(cosine) ? cosine : std::clamp(cosine, -1., 1.)) *
          180. / 3.14159265358979323846;
    }
  }
  return output;
}
py::array_t<bool> in_front(Array points, Array rotation_array,
                           Array translation) {
  shape(points, {-1, 3});
  shape(rotation_array, {3, 3});
  shape(translation, {3});
  auto n = points.shape(0);
  py::array_t<bool> output(n);
  auto out = output.mutable_unchecked<1>();
  const double *r = rotation_array.data();
  double tz = translation.data()[2];
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      const double *p = points.data() + 3 * i;
      out(i) = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + tz > 0;
    }
  }
  return output;
}
Array reprojection(Array points, Array pixels, Array rotation_array,
                   Array translation, Array intrinsics) {
  shape(points, {-1, 3});
  shape(pixels, {points.shape(0), 2});
  shape(rotation_array, {3, 3});
  shape(translation, {3});
  shape(intrinsics, {3, 3});
  auto n = points.shape(0);
  py::array_t<double> output(n);
  auto out = output.mutable_unchecked<1>();
  M r = matrix_at(rotation_array.data()), k = matrix_at(intrinsics.data());
  V t = vector_at(translation.data());
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      V h = mv(k, add(mv(r, vector_at(points.data() + 3 * i)), t));
      double dx = h[0] / h[2] - pixels.data()[2 * i],
             dy = h[1] / h[2] - pixels.data()[2 * i + 1];
      out(i) = std::sqrt(dx * dx + dy * dy);
    }
  }
  return output;
}
py::array_t<unsigned int> camera_support(Array points, Array rotations,
                                       Array translations, Array intrinsics,
                                       int width, int height, int threads) {
  shape(points, {-1, 3});
  shape(rotations, {-1, 3, 3});
  shape(translations, {rotations.shape(0), 3});
  shape(intrinsics, {4});
  if (width < 1 || height < 1 || threads < 1)
    throw std::invalid_argument("dimensions and threads must be positive");
  const auto n = points.shape(0), cameras = rotations.shape(0);
  py::array_t<unsigned int> output(n);
  auto *out = output.mutable_data();
  const double *p = points.data(), *r = rotations.data(),
               *t = translations.data(), *k = intrinsics.data();
  {
    py::gil_scoped_release release;
    auto work = [&](py::ssize_t begin, py::ssize_t end) {
      std::fill(out + begin, out + end, 0u);
      for (py::ssize_t c = 0; c < cameras; ++c) {
        const double *rc = r + c * 9, *tc = t + c * 3;
        for (auto i = begin; i < end; ++i) {
          const double *v = p + i * 3;
          const double z = rc[6]*v[0] + rc[7]*v[1] + rc[8]*v[2] + tc[2];
          if (!(z > 1e-6)) continue;
          const double x = k[0]*(rc[0]*v[0]+rc[1]*v[1]+rc[2]*v[2]+tc[0])/z+k[2];
          const double y = k[1]*(rc[3]*v[0]+rc[4]*v[1]+rc[5]*v[2]+tc[1])/z+k[3];
          if (x >= 0 && x < width && y >= 0 && y < height) ++out[i];
        }
      }
    };
    int workers = static_cast<int>(std::min<py::ssize_t>(threads, n));
    std::vector<std::thread> pool;
    try {
      for (int j = 1; j < workers; ++j)
        pool.emplace_back(work, n*j/workers, n*(j+1)/workers);
    } catch (...) {
      for (auto &worker : pool) worker.join();
      throw;
    }
    work(0, workers ? n/workers : 0);
    for (auto &worker : pool) worker.join();
  }
  return output;
}

// 0: unknown/outside; 1: surface agreement; 2: free-space conflict;
// 3: occluded behind the measured surface (not negative evidence).
py::array_t<unsigned char> depth_support(Array points, Array rotation_array,
                                        Array translation, Array intrinsics,
                                        Array depth, double absolute_tolerance,
                                        double relative_tolerance) {
  shape(points, {-1, 3}); shape(rotation_array, {3, 3});
  shape(translation, {3}); shape(intrinsics, {4}); shape(depth, {-1, -1});
  if (!(absolute_tolerance >= 0) || !(relative_tolerance >= 0))
    throw std::invalid_argument("depth tolerances must be nonnegative");
  auto n = points.shape(0);
  py::array_t<unsigned char> output(n);
  auto *out = output.mutable_data();
  const auto width = depth.shape(1), height = depth.shape(0);
  const auto r = matrix_at(rotation_array.data());
  const auto t = vector_at(translation.data());
  const auto *k = intrinsics.data(), *d = depth.data(), *p = points.data();
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      out[i] = 0;
      V v = add(mv(r, vector_at(p + 3*i)), t);
      if (!(v[2] > 1e-6)) continue;
      double x = k[0]*v[0]/v[2]+k[2], y = k[1]*v[1]/v[2]+k[3];
      if (!(x >= 0 && x < width && y >= 0 && y < height)) continue;
      double measured = d[static_cast<py::ssize_t>(y)*width + static_cast<py::ssize_t>(x)];
      if (!(measured > 0) || !std::isfinite(measured)) continue;
      double tolerance = absolute_tolerance + relative_tolerance*measured;
      out[i] = std::abs(v[2]-measured) <= tolerance ? 1 : (v[2] < measured ? 2 : 3);
    }
  }
  return output;
}

PYBIND11_MODULE(_native, m) {
  m.doc() = "Optional C++ residuals, analytic Jacobians, and geometry; SciPy "
            "owns optimization.";
  m.def("ba_evaluate", &ba_evaluate);
  m.def("graph_evaluate", &graph_evaluate);
  m.def("project_points", &project);
  m.def("triangulation_angles", &angles);
  m.def("points_in_front", &in_front);
  m.def("reprojection_errors", &reprojection);
  m.def("camera_support", &camera_support);
  m.def("depth_support", &depth_support);
}
