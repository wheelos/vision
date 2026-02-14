#pragma once

#include "vision/camera_model.h"

namespace wheel {
namespace vision {

class OmnidirectionalCameraModel : public CameraModel {
 public:
  OmnidirectionalCameraModel() = default;
  ~OmnidirectionalCameraModel() override = default;

  // Init with OCam parameters
  // center: [cx, cy], affine: [c, d, e]
  void Init(uint32_t width, uint32_t height,
            const std::vector<float>& cam2world,
            const std::vector<float>& world2cam, const Eigen::Vector2f& center,
            const Eigen::Vector3f& affine);

  bool RayToPixel(const Eigen::Vector3f& ray,
                  Eigen::Vector2f* pixel) const override;
  bool PixelToRay(const Eigen::Vector2f& pixel,
                  Eigen::Vector3f* ray) const override;
  std::shared_ptr<CameraModel> GetIdealModel() const override;

  CameraModelType Type() const override { return CameraModelType::kOmni; }
  std::string Name() const override { return "OmnidirectionalCameraModel"; }

 private:
  // Polynomial coefficients
  std::vector<float> cam2world_;  // poly(rho) -> z
  std::vector<float> world2cam_;  // poly(theta) -> rho (or z)

  Eigen::Vector2f center_ = Eigen::Vector2f::Zero();
  Eigen::Vector3f affine_ = Eigen::Vector3f::Zero();  // c, d, e
};

}  // namespace vision
}  // namespace wheel
