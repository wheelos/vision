#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "vision/lidar/core/point_cloud.h"
#include "vision/lidar/memory/buffer.h"

namespace wheel {
namespace vision {
namespace lidar {

struct PointFieldSpecification {
  PointField field = PointField::kX;
  ScalarType scalar_type = ScalarType::kFloat32;
};

class PointCloudBuffer {
 public:
  PointCloudBuffer() = default;

  bool Allocate(uint32_t point_count, PointLayout layout,
                MemoryType memory_type, CoordinateFrame frame,
                std::initializer_list<PointFieldSpecification> fields);
  bool AllocateLike(const PointCloudView& source, uint32_t point_count,
                    PointLayout output_layout = PointLayout::kPlanar);
  void Reset();

  inline const PointCloudView& view() const { return view_; }
  inline PointLayout layout() const { return view_.layout(); }
  inline MemoryType memory_type() const { return view_.memory_type(); }
  inline CoordinateFrame frame() const { return view_.frame(); }
  inline uint32_t point_count() const { return view_.point_count(); }
  inline uint32_t field_count() const { return view_.field_count(); }
  inline uint32_t point_stride_bytes() const { return point_stride_bytes_; }

  const PointFieldDescriptor* FindField(PointField field) const;

  template <typename T>
  inline T* mutable_field_data(PointField field) {
    const PointFieldDescriptor* descriptor = FindField(field);
    if (descriptor == nullptr || descriptor->element_size() != sizeof(T)) {
      return nullptr;
    }
    return reinterpret_cast<T*>(const_cast<void*>(descriptor->data));
  }

  template <typename T>
  inline const T* field_data(PointField field) const {
    const PointFieldDescriptor* descriptor = FindField(field);
    if (descriptor == nullptr || descriptor->element_size() != sizeof(T)) {
      return nullptr;
    }
    return reinterpret_cast<const T*>(descriptor->data);
  }

  template <typename T>
  inline T* mutable_field_element(PointField field, uint32_t index) {
    PointFieldDescriptor const* descriptor = FindField(field);
    if (descriptor == nullptr || descriptor->element_size() != sizeof(T) ||
        index >= point_count()) {
      return nullptr;
    }

    uint8_t* base =
        reinterpret_cast<uint8_t*>(const_cast<void*>(descriptor->data));
    return reinterpret_cast<T*>(base +
                                static_cast<size_t>(index) * descriptor->stride_bytes);
  }

  template <typename T>
  inline const T* field_element(PointField field, uint32_t index) const {
    const PointFieldDescriptor* descriptor = FindField(field);
    if (descriptor == nullptr || descriptor->element_size() != sizeof(T) ||
        index >= point_count()) {
      return nullptr;
    }

    const uint8_t* base = reinterpret_cast<const uint8_t*>(descriptor->data);
    return reinterpret_cast<const T*>(
        base + static_cast<size_t>(index) * descriptor->stride_bytes);
  }

 private:
  bool Allocate(uint32_t point_count, PointLayout layout,
                MemoryType memory_type, CoordinateFrame frame,
                const PointFieldSpecification* fields, uint32_t field_count);
  bool AllocatePlanar(uint32_t point_count, MemoryType memory_type,
                      const PointFieldSpecification* fields,
                      uint32_t field_count);
  bool AllocateInterleaved(uint32_t point_count, MemoryType memory_type,
                           const PointFieldSpecification* fields,
                           uint32_t field_count);

  PointCloudView view_;
  std::array<Buffer, kMaxPointFields> planar_buffers_;
  Buffer interleaved_buffer_;
  uint32_t point_stride_bytes_ = 0U;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
