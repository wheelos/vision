#pragma once

#include "vision/lidar/core/point_cloud.h"
#include "vision/lidar/core/rigid_transform.h"
#include "vision/lidar/range_image/range_image.h"
#include "vision/lidar/runtime/cuda_stream.h"

namespace wheel {
namespace vision {
namespace lidar {

class RangeImageBuilder {
 public:
  explicit RangeImageBuilder(const RangeImageConfig& config) : config_(config) {}

  bool Build(const PointCloudView& point_cloud, RangeImageBuffer* output,
             RangeImageWorkspace* workspace,
             const RigidTransform3f* lidar_from_input = nullptr,
             CudaStream stream = kDefaultCudaStream) const;
  bool BuildAsync(const PointCloudView& point_cloud, RangeImageBuffer* output,
                  RangeImageWorkspace* workspace,
                  const RigidTransform3f* lidar_from_input = nullptr,
                  CudaStream stream = kDefaultCudaStream) const;

  inline const RangeImageConfig& config() const { return config_; }

 private:
  RangeImageConfig config_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
