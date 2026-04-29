#pragma once

#include <cstdint>
#include <limits>

#include "vision/lidar/core/point_cloud.h"
#include "vision/lidar/core/point_cloud_buffer.h"
#include "vision/lidar/memory/buffer.h"
#include "vision/lidar/runtime/cuda_stream.h"

namespace wheel {
namespace vision {
namespace lidar {

enum class CropBoxMode : uint8_t {
  kKeepInside = 0U,
  kRemoveInside = 1U,
};

struct CropBoxFilterConfig {
  float min_x = -std::numeric_limits<float>::infinity();
  float max_x = std::numeric_limits<float>::infinity();
  float min_y = -std::numeric_limits<float>::infinity();
  float max_y = std::numeric_limits<float>::infinity();
  float min_z = -std::numeric_limits<float>::infinity();
  float max_z = std::numeric_limits<float>::infinity();
  bool reject_non_finite = true;
  CropBoxMode mode = CropBoxMode::kKeepInside;
  PointLayout output_layout = PointLayout::kPlanar;

  inline bool IsValid() const {
    return min_x <= max_x && min_y <= max_y && min_z <= max_z;
  }
};

class CropBoxFilterWorkspace {
 public:
  bool Allocate(uint32_t max_points,
                MemoryType index_memory_type = MemoryType::kUnified);
  bool Clear(CudaStream stream = kDefaultCudaStream);
  void Reset();

  inline uint32_t* selected_indices() {
    return selected_indices_.mutable_data<uint32_t>();
  }
  inline const uint32_t* selected_indices() const {
    return selected_indices_.data<uint32_t>();
  }
  inline uint32_t* selected_count() {
    return selected_count_.mutable_data<uint32_t>();
  }
  inline const uint32_t* selected_count() const {
    return selected_count_.data<uint32_t>();
  }
  inline uint32_t max_points() const { return max_points_; }
  inline MemoryType index_memory_type() const {
    return selected_indices_.memory_type();
  }

 private:
  Buffer selected_indices_;
  Buffer selected_count_;
  uint32_t max_points_ = 0U;
};

class CropBoxFilter {
 public:
  explicit CropBoxFilter(const CropBoxFilterConfig& config) : config_(config) {}

  bool Filter(const PointCloudView& source, PointCloudBuffer* output,
              CropBoxFilterWorkspace* workspace,
              CudaStream stream = kDefaultCudaStream) const;

  inline const CropBoxFilterConfig& config() const { return config_; }

 private:
  CropBoxFilterConfig config_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
