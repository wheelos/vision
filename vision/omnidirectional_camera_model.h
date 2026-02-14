#pragma once

#include <array>
#include <memory>
#include <vector>

#include "vision/camera_model.h"

namespace wheel {
namespace vision {

class OmnidirectionalCameraModel : public CameraModel {
public:
  OmnidirectionalCameraModel() = default;
  ~OmnidirectionalCameraModel() override = default;

  // Initialization interface (OCamCalib model)
  // cam2world: Polynomial coefficients from image plane radius rho -> z
  // world2cam: Polynomial coefficients from incident angle theta -> image plane
  // radius rho affine: [c, d, e] corresponding to the affine matrix center:
  // [xc, yc] image principal point
  void Init(uint32_t width, uint32_t height, const Eigen::Matrix3f &intrinsics,
            const std::vector<float> &cam2world,
            const std::vector<float> &world2cam,
            const std::array<float, 3> &affine,
            const std::array<float, 2> &center);

  bool RayToPixel(const Eigen::Vector3f &ray,
                  Eigen::Vector2f *pixel) const override;

  bool PixelToRay(const Eigen::Vector2f &pixel,
                  Eigen::Vector3f *ray) const override;

  std::shared_ptr<CameraModel> GetIdealModel() const override;

  CameraModelType Type() const override { return CameraModelType::kOmni; }
  std::string Name() const override { return "OmnidirectionalCameraModel"; }

private:
  std::vector<float> cam2world_;
  std::vector<float> world2cam_;
  float center_[2] = {0.0f, 0.0f};
  float affine_[3] = {1.0f, 0.0f, 0.0f};
};

} // namespace vision
} // namespace wheel
