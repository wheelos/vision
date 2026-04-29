#pragma once

#include <cstdint>

#include "vision/lidar/core/point_cloud.h"
#include "vision/lidar/core/point_cloud_buffer.h"
#include "vision/lidar/runtime/cuda_stream.h"

namespace wheel {
namespace vision {
namespace lidar {

struct DecimationFilterConfig {
  uint32_t stride = 1U;
  uint32_t offset = 0U;
  PointLayout output_layout = PointLayout::kPlanar;

  inline bool IsValid() const { return stride > 0U; }
};

class DecimationFilter {
 public:
  explicit DecimationFilter(const DecimationFilterConfig& config)
      : config_(config) {}

  bool Filter(const PointCloudView& source, PointCloudBuffer* output,
              CudaStream stream = kDefaultCudaStream) const;
  bool FilterAsync(const PointCloudView& source, PointCloudBuffer* output,
                   CudaStream stream = kDefaultCudaStream) const;

  inline const DecimationFilterConfig& config() const { return config_; }

 private:
  DecimationFilterConfig config_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
