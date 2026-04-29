#pragma once

#include <cstdint>

#include "vision/lidar/memory/buffer.h"

namespace wheel {
namespace vision {
namespace lidar {

struct RangeImageConfig {
  uint32_t rows = 0U;
  uint32_t cols = 0U;
  float min_range_m = 0.0f;
  float max_range_m = 0.0f;
  float min_azimuth_rad = 0.0f;
  float max_azimuth_rad = 0.0f;
  float min_elevation_rad = 0.0f;
  float max_elevation_rad = 0.0f;

  inline bool IsValid() const {
    return rows > 0U && cols > 0U && min_range_m >= 0.0f &&
           max_range_m > min_range_m &&
           max_azimuth_rad > min_azimuth_rad &&
           max_elevation_rad > min_elevation_rad;
  }

  inline uint32_t pixel_count() const { return rows * cols; }
};

class RangeImageBuffer {
 public:
  bool Allocate(const RangeImageConfig& config,
                MemoryType memory_type = MemoryType::kUnified);

  inline float* range() { return range_.mutable_data<float>(); }
  inline const float* range() const { return range_.data<float>(); }

  inline float* intensity() { return intensity_.mutable_data<float>(); }
  inline const float* intensity() const { return intensity_.data<float>(); }

  inline uint8_t* valid_mask() { return valid_mask_.mutable_data<uint8_t>(); }
  inline const uint8_t* valid_mask() const { return valid_mask_.data<uint8_t>(); }

  inline uint32_t rows() const { return rows_; }
  inline uint32_t cols() const { return cols_; }
  inline uint32_t pixel_count() const { return rows_ * cols_; }
  inline MemoryType memory_type() const { return range_.memory_type(); }

 private:
  uint32_t rows_ = 0U;
  uint32_t cols_ = 0U;
  Buffer range_;
  Buffer intensity_;
  Buffer valid_mask_;
};

class RangeImageWorkspace {
 public:
  bool Allocate(const RangeImageConfig& config,
                MemoryType memory_type = MemoryType::kUnified);

  inline uint64_t* packed_range_intensity() {
    return packed_range_intensity_.mutable_data<uint64_t>();
  }

  inline const uint64_t* packed_range_intensity() const {
    return packed_range_intensity_.data<uint64_t>();
  }

  inline uint32_t pixel_count() const { return pixel_count_; }
  inline MemoryType memory_type() const {
    return packed_range_intensity_.memory_type();
  }

 private:
  uint32_t pixel_count_ = 0U;
  Buffer packed_range_intensity_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
