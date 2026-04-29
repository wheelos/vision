#pragma once

#include <cstdint>

#include "vision/lidar/core/point_cloud.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace internal {

struct DeviceFieldDescriptor {
  const uint8_t* data = nullptr;
  uint32_t stride_bytes = 0U;
  uint32_t element_size = 0U;
};

struct MutableDeviceFieldDescriptor {
  uint8_t* data = nullptr;
  uint32_t stride_bytes = 0U;
  uint32_t element_size = 0U;
};

struct DevicePointCloudView {
  uint32_t point_count = 0U;
  uint32_t field_count = 0U;
  DeviceFieldDescriptor fields[kMaxPointFields]{};
};

struct MutableDevicePointCloudView {
  uint32_t point_count = 0U;
  uint32_t field_count = 0U;
  MutableDeviceFieldDescriptor fields[kMaxPointFields]{};
};

inline DeviceFieldDescriptor MakeDeviceField(const PointFieldDescriptor& field) {
  DeviceFieldDescriptor descriptor;
  descriptor.data = reinterpret_cast<const uint8_t*>(field.data);
  descriptor.stride_bytes = field.stride_bytes;
  descriptor.element_size = static_cast<uint32_t>(field.element_size());
  return descriptor;
}

inline MutableDeviceFieldDescriptor MakeMutableDeviceField(
    const PointFieldDescriptor& field) {
  MutableDeviceFieldDescriptor descriptor;
  descriptor.data = reinterpret_cast<uint8_t*>(const_cast<void*>(field.data));
  descriptor.stride_bytes = field.stride_bytes;
  descriptor.element_size = static_cast<uint32_t>(field.element_size());
  return descriptor;
}

inline DevicePointCloudView MakeDevicePointCloudView(const PointCloudView& view) {
  DevicePointCloudView device_view;
  device_view.point_count = view.point_count();
  device_view.field_count = view.field_count();
  for (uint32_t index = 0U; index < view.field_count(); ++index) {
    device_view.fields[index] = MakeDeviceField(view.fields()[index]);
  }
  return device_view;
}

inline MutableDevicePointCloudView MakeMutableDevicePointCloudView(
    const PointCloudView& view) {
  MutableDevicePointCloudView device_view;
  device_view.point_count = view.point_count();
  device_view.field_count = view.field_count();
  for (uint32_t index = 0U; index < view.field_count(); ++index) {
    device_view.fields[index] = MakeMutableDeviceField(view.fields()[index]);
  }
  return device_view;
}

__device__ inline const uint8_t* FieldElement(const DeviceFieldDescriptor& field,
                                              uint32_t point_index) {
  return field.data + static_cast<size_t>(point_index) * field.stride_bytes;
}

__device__ inline uint8_t* FieldElement(const MutableDeviceFieldDescriptor& field,
                                        uint32_t point_index) {
  return field.data + static_cast<size_t>(point_index) * field.stride_bytes;
}

__device__ inline void CopyFieldElement(const DeviceFieldDescriptor& source,
                                        uint32_t source_index,
                                        const MutableDeviceFieldDescriptor& target,
                                        uint32_t target_index) {
  const uint8_t* source_ptr = FieldElement(source, source_index);
  uint8_t* target_ptr = FieldElement(target, target_index);
  for (uint32_t byte_index = 0U; byte_index < source.element_size; ++byte_index) {
    target_ptr[byte_index] = source_ptr[byte_index];
  }
}

__device__ inline float LoadFloat(const DeviceFieldDescriptor& field,
                                  uint32_t point_index) {
  return *reinterpret_cast<const float*>(FieldElement(field, point_index));
}

}  // namespace internal
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
