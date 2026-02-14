#include "vision/pinhole_camera_model.h"

#include <cmath>

namespace wheel {
namespace vision {

constexpr float kEpsilon = 1e-6f;

void PinholeCameraModel::Init(uint32_t width, uint32_t height,
                              const Eigen::Matrix3f &intrinsics) {
  width_ = width;
  height_ = height;
  intrinsic_matrix_ = intrinsics;
}

bool PinholeCameraModel::RayToPixel(const Eigen::Vector3f &ray,
                                    Eigen::Vector2f *pixel) const {
  // reject points behind the camera or extremely close to the optical center
  // plane
  if (ray.z() < kEpsilon) {
    return false;
  }

  // Avoid matrix multiplication, directly expand the calculation to improve
  // low-level execution efficiency
  const float inv_z = 1.0f / ray.z();
  const float x_norm = ray.x() * inv_z;
  const float y_norm = ray.y() * inv_z;

  pixel->x() = intrinsic_matrix_(0, 0) * x_norm + intrinsic_matrix_(0, 2);
  pixel->y() = intrinsic_matrix_(1, 1) * y_norm + intrinsic_matrix_(1, 2);

  return true;
}

bool PinholeCameraModel::PixelToRay(const Eigen::Vector2f &pixel,
                                    Eigen::Vector3f *ray) const {
  // Back-project to the normalized plane Z = 1
  ray->x() = (pixel.x() - intrinsic_matrix_(0, 2)) / intrinsic_matrix_(0, 0);
  ray->y() = (pixel.y() - intrinsic_matrix_(1, 2)) / intrinsic_matrix_(1, 1);
  ray->z() = 1.0f;

  // Optional: If the business layer requires unit vectors, this can be done
  // here. ray->normalize(); But usually, keeping the Z=1 form is more
  // beneficial for subsequent multiplication by the depth value to restore the
  // true shape

  // 3D coordinates.
  return true;
}

std::shared_ptr<CameraModel> PinholeCameraModel::GetIdealModel() const {
  auto model = std::make_shared<PinholeCameraModel>();
  model->Init(width_, height_, intrinsic_matrix_);
  return model;
}

} // namespace vision
} // namespace wheel
