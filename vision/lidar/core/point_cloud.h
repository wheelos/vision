#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "vision/lidar/types.h"

namespace wheel {
namespace vision {
namespace lidar {

struct PointFieldDescriptor {
  PointField field = PointField::kX;
  ScalarType scalar_type = ScalarType::kUnknown;
  const void* data = nullptr;
  uint32_t stride_bytes = 0U;

  inline size_t element_size() const { return ScalarTypeSize(scalar_type); }
};

class PointCloudView {
 public:
  PointCloudView() = default;
  PointCloudView(PointLayout layout, MemoryType memory_type,
                 CoordinateFrame frame, uint32_t point_count,
                 std::initializer_list<PointFieldDescriptor> fields);

  bool SetFields(std::initializer_list<PointFieldDescriptor> fields);
  bool AddField(const PointFieldDescriptor& field);

  const PointFieldDescriptor* FindField(PointField field) const;
  bool HasField(PointField field) const;
  bool IsValid() const;
  bool IsRangeImageCompatible() const;

  inline PointLayout layout() const { return layout_; }
  inline MemoryType memory_type() const { return memory_type_; }
  inline CoordinateFrame frame() const { return frame_; }
  inline uint32_t point_count() const { return point_count_; }
  inline uint32_t field_count() const { return field_count_; }
  inline const std::array<PointFieldDescriptor, kMaxPointFields>& fields() const {
    return fields_;
  }

  inline void set_layout(PointLayout layout) { layout_ = layout; }
  inline void set_memory_type(MemoryType memory_type) { memory_type_ = memory_type; }
  inline void set_frame(CoordinateFrame frame) { frame_ = frame; }
  inline void set_point_count(uint32_t point_count) { point_count_ = point_count; }

 private:
  PointLayout layout_ = PointLayout::kInterleaved;
  MemoryType memory_type_ = MemoryType::kHost;
  CoordinateFrame frame_ = CoordinateFrame::kLidar;
  uint32_t point_count_ = 0U;
  std::array<PointFieldDescriptor, kMaxPointFields> fields_{};
  uint32_t field_count_ = 0U;
  bool schema_valid_ = true;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
