#pragma once

#include <Eigen/Core>

namespace wheel {
namespace vision {
namespace lidar {

struct RigidTransform3f {
  Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
  Eigen::Vector3f translation = Eigen::Vector3f::Zero();

  inline bool IsFinite() const {
    return rotation.allFinite() && translation.allFinite();
  }
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
