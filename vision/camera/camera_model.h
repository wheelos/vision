#pragma once

#include <cmath>
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

  // Core API: 3D (Camera Frame) -> 2D (Pixel)
  // Input ray must have z > 0 (or strictly defined for Omni)
  virtual bool RayToPixel(const Eigen::Vector3f& ray,
                          Eigen::Vector2f* pixel) const = 0;

  // Core API: 2D (Pixel) -> 3D (Camera Frame)
  // Returns a ray (direction vector), not necessarily normalized.
  virtual bool PixelToRay(const Eigen::Vector2f& pixel,
                          Eigen::Vector3f* ray) const = 0;

  // Returns a virtual ideal pinhole model (undistorted) based on current model.
  virtual std::shared_ptr<CameraModel> GetIdealModel() const = 0;

  virtual CameraModelType Type() const = 0;
  virtual std::string Name() const = 0;

  // Handle Image Resize/Crop ROI
  // Updates intrinsic matrix and image size.
  // transform: 3x3 matrix mapping old pixels to new pixels.
  virtual void ApplyImageTransform(const Eigen::Matrix3f& transform,
                                   uint32_t new_width, uint32_t new_height) {
    intrinsic_matrix_ = transform * intrinsic_matrix_;
    width_ = new_width;
    height_ = new_height;
  }

  virtual float GetHorizontalFOV() const {
    // Approximation for general cameras based on fx
    return 2.0f * std::atan(static_cast<float>(width_) /
                            (2.0f * intrinsic_matrix_(0, 0)));
  }

  inline ImageSize image_size() const { return {width_, height_}; }
  inline uint32_t width() const { return width_; }
  inline uint32_t height() const { return height_; }
  inline const Eigen::Matrix3f& intrinsic_matrix() const {
    return intrinsic_matrix_;
  }

  // Helper: Check if ray projects inside image bounds
  inline bool IsRayInsideFov(const Eigen::Vector3f& ray) const {
    Eigen::Vector2f pixel;
    if (!RayToPixel(ray, &pixel)) {
      return false;
    }
    return pixel.x() >= 0.0f && pixel.y() >= 0.0f &&
           pixel.x() <= static_cast<float>(width_ - 1) &&
           pixel.y() <= static_cast<float>(height_ - 1);
  }

  // Overloads for convenience
  inline Eigen::Vector2f RayToPixel(const Eigen::Vector3f& ray) const {
    Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
    RayToPixel(ray, &pixel);
    return pixel;
  }

  inline Eigen::Vector3f PixelToRay(const Eigen::Vector2f& pixel) const {
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

}  // namespace vision
}  // namespace wheel
