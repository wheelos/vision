#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace wheel {
namespace vision {

// Camera model type definition
enum class CameraModelType : uint8_t {
  kUnknown = 0,
  kPinhole = 1,
  kBrown = 2,
  kOmni = 3,
};

// Unified image size structure
struct ImageSize {
  uint32_t width = 0;
  uint32_t height = 0;
};

class CameraModel {
public:
  virtual ~CameraModel() = default;

  // @brief: Project a 3D ray/point in the camera coordinate system to the 2D
  // pixel plane
  // @note: The z value of the input ray must be > 0 (in front of the camera)
  virtual bool RayToPixel(const Eigen::Vector3f &ray,
                          Eigen::Vector2f *pixel) const = 0;

  // @brief: Back-project a 2D pixel to a 3D ray in the camera coordinate system
  // (usually a unit vector or a vector with z=1)
  virtual bool PixelToRay(const Eigen::Vector2f &pixel,
                          Eigen::Vector3f *ray) const = 0;

  // @brief: Return the undistorted (ideal) pinhole model corresponding to this
  // distortion model. This is used for image undistortion, allowing other
  // modules to directly process as Pinhole.
  virtual std::shared_ptr<CameraModel> GetIdealModel() const = 0;

  virtual CameraModelType Type() const = 0;
  virtual std::string Name() const = 0;

  inline ImageSize image_size() const { return {width_, height_}; }
  inline uint32_t width() const { return width_; }
  inline uint32_t height() const { return height_; }
  inline const Eigen::Matrix3f &intrinsic_matrix() const {
    return intrinsic_matrix_;
  }

  // Check if the projected ray falls within the valid image field of view (FOV)
  inline bool IsRayInsideFov(const Eigen::Vector3f &ray) const {
    Eigen::Vector2f pixel;
    if (!RayToPixel(ray, &pixel)) {
      return false;
    }
    return pixel.x() >= 0.0f && pixel.y() >= 0.0f &&
           pixel.x() < static_cast<float>(width_) &&
           pixel.y() < static_cast<float>(height_);
  }

  inline Eigen::Vector2f RayToPixel(const Eigen::Vector3f &ray) const {
    Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
    RayToPixel(ray, &pixel);
    return pixel;
  }

  inline Eigen::Vector3f PixelToRay(const Eigen::Vector2f &pixel) const {
    Eigen::Vector3f ray = Eigen::Vector3f::Zero();
    PixelToRay(pixel, &ray);
    return ray;
  }

protected:
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  Eigen::Matrix3f intrinsic_matrix_ = Eigen::Matrix3f::Identity();
};

using CameraModelPtr = std::shared_ptr<CameraModel>;
using CameraModelConstPtr = std::shared_ptr<const CameraModel>;

} // namespace vision
} // namespace wheel
