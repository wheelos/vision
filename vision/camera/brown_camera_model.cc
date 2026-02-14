#include "vision/brown_camera_model.h"

#include <cmath>

namespace wheel {
namespace vision {

namespace {
constexpr float kEpsilon = 1e-6f;
} // namespace

void BrownCameraModel::Init(uint32_t width, uint32_t height,
                            const Eigen::Matrix3f &intrinsics,
                            const Eigen::Matrix<float, 5, 1> &dist_coeffs) {
  width_ = width;
  height_ = height;
  intrinsic_matrix_ = intrinsics;
  distort_params_ = dist_coeffs;
}

bool BrownCameraModel::RayToPixel(const Eigen::Vector3f &ray,
                                  Eigen::Vector2f *pixel) const {
  if (ray.z() < kEpsilon) {
    return false;
  }

  const float x = ray.x() / ray.z();
  const float y = ray.y() / ray.z();

  const float r2 = x * x + y * y;
  const float r4 = r2 * r2;
  const float r6 = r4 * r2;

  const float k1 = distort_params_[0];
  const float k2 = distort_params_[1];
  const float p1 = distort_params_[2];
  const float p2 = distort_params_[3];
  const float k3 = distort_params_[4];

  // Radial distortion factor
  const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

  // Tangential distortion calculation
  const float x_dist =
      x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
  const float y_dist =
      y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;

  // Multiply by intrinsic parameters
  pixel->x() = intrinsic_matrix_(0, 0) * x_dist + intrinsic_matrix_(0, 2);
  pixel->y() = intrinsic_matrix_(1, 1) * y_dist + intrinsic_matrix_(1, 2);

  return true;
}

bool BrownCameraModel::PixelToRay(const Eigen::Vector2f &pixel,
                                  Eigen::Vector3f *ray) const {
  // 1. Convert pixel to normalized image plane with distortion
  const float x_dist =
      (pixel.x() - intrinsic_matrix_(0, 2)) / intrinsic_matrix_(0, 0);
  const float y_dist =
      (pixel.y() - intrinsic_matrix_(1, 2)) / intrinsic_matrix_(1, 1);

  // 2. Fixed-point iteration to solve for undistorted (x, y)
  float x_iter = x_dist;
  float y_iter = y_dist;

  const float k1 = distort_params_[0];
  const float k2 = distort_params_[1];
  const float p1 = distort_params_[2];
  const float p2 = distort_params_[3];
  const float k3 = distort_params_[4];

  const int kMaxIterations =
      5; // Typically 5 iterations are sufficient for convergence in ADAS
  const float kConvergenceEps = 1e-5f; // Convergence condition

  for (int i = 0; i < kMaxIterations; ++i) {
    const float r2 = x_iter * x_iter + y_iter * y_iter;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

    // Tangential distortion calculation
    const float delta_x =
        2.0f * p1 * x_iter * y_iter + p2 * (r2 + 2.0f * x_iter * x_iter);
    const float delta_y =
        p1 * (r2 + 2.0f * y_iter * y_iter) + 2.0f * p2 * x_iter * y_iter;

    // Fixed-point formula: x = (x_dist - delta_x) / radial
    const float x_next = (x_dist - delta_x) / radial;
    const float y_next = (y_dist - delta_y) / radial;

    // Check for convergence
    if (std::abs(x_next - x_iter) < kConvergenceEps &&
        std::abs(y_next - y_iter) < kConvergenceEps) {
      x_iter = x_next;
      y_iter = y_next;
      break;
    }

    x_iter = x_next;
    y_iter = y_next;
  }

  ray->x() = x_iter;
  ray->y() = y_iter;
  ray->z() = 1.0f;
  return true;
}

std::shared_ptr<CameraModel> BrownCameraModel::GetIdealModel() const {
  auto model = std::make_shared<BrownCameraModel>();
  model->Init(width_, height_, intrinsic_matrix_, distort_params_);
  return model;
}

} // namespace vision
} // namespace wheel
