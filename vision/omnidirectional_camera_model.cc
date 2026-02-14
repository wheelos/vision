#include "vision/omnidirectional_camera_model.h"

#include <cmath>

namespace wheel {
namespace vision {

namespace {
constexpr float kEpsilon = 1e-8f;

float EvalPoly(const std::vector<float> &coeffs, float x) {
  float result = 0.0f;
  float xn = 1.0f;
  for (float c : coeffs) {
    result += c * xn;
    xn *= x;
  }
  return result;
}
} // namespace

void OmnidirectionalCameraModel::Init(uint32_t width, uint32_t height,
                                      const Eigen::Matrix3f &intrinsics,
                                      const std::vector<float> &cam2world,
                                      const std::vector<float> &world2cam,
                                      const std::array<float, 3> &affine,
                                      const std::array<float, 2> &center) {
  width_ = width;
  height_ = height;
  intrinsic_matrix_ = intrinsics;
  cam2world_ = cam2world;
  world2cam_ = world2cam;
  affine_[0] = affine[0];
  affine_[1] = affine[1];
  affine_[2] = affine[2];
  center_[0] = center[0];
  center_[1] = center[1];
}

bool OmnidirectionalCameraModel::RayToPixel(const Eigen::Vector3f &ray,
                                            Eigen::Vector2f *pixel) const {
  const float x = ray.x();
  const float y = ray.y();
  const float z = ray.z();

  const float norm = std::sqrt(x * x + y * y);
  if (norm < kEpsilon) {
    pixel->x() = center_[0];
    pixel->y() = center_[1];
    return true;
  }

  // Incident angle theta (OCamCalib)
  const float theta = std::atan2(z, norm);
  const float rho = EvalPoly(world2cam_, theta);

  const float x_img = (x / norm) * rho;
  const float y_img = (y / norm) * rho;

  // Affine + principal point
  pixel->x() = x_img * affine_[0] + y_img * affine_[1] + center_[0];
  pixel->y() = x_img * affine_[2] + y_img + center_[1];
  return true;
}

bool OmnidirectionalCameraModel::PixelToRay(const Eigen::Vector2f &pixel,
                                            Eigen::Vector3f *ray) const {
  const float u = pixel.x();
  const float v = pixel.y();

  // Inverse affine transformation
  const float c = affine_[0];
  const float d = affine_[1];
  const float e = affine_[2];
  const float denom = (c - d * e);
  if (std::abs(denom) < kEpsilon) {
    return false;
  }

  const float x_img = (u - center_[0] - d * (v - center_[1])) / denom;
  const float y_img = (v - center_[1] - e * x_img);

  const float rho = std::sqrt(x_img * x_img + y_img * y_img);
  const float z = EvalPoly(cam2world_, rho);

  ray->x() = x_img;
  ray->y() = y_img;
  ray->z() = z;

  // Normalize to unit vectors to avoid scale ambiguity.
  const float norm = std::sqrt(ray->x() * ray->x() + ray->y() * ray->y() +
                               ray->z() * ray->z());
  if (norm < kEpsilon) {
    return false;
  }
  ray->x() /= norm;
  ray->y() /= norm;
  ray->z() /= norm;
  return true;
}

std::shared_ptr<CameraModel> OmnidirectionalCameraModel::GetIdealModel() const {
  auto model = std::make_shared<OmnidirectionalCameraModel>();
  std::array<float, 3> affine_arr = { affine_[0], affine_[1], affine_[2] };
  std::array<float, 2> center_arr = { center_[0], center_[1] };

  model->Init(width_, height_, intrinsic_matrix_,
              cam2world_, world2cam_,
              affine_arr, center_arr);

  return model;
}

} // namespace vision
} // namespace wheel
