#include "vision/lidar/core/point_cloud.h"

namespace wheel {
namespace vision {
namespace lidar {

PointCloudView::PointCloudView(
    PointLayout layout, MemoryType memory_type, CoordinateFrame frame,
    uint32_t point_count, std::initializer_list<PointFieldDescriptor> fields)
    : layout_(layout),
      memory_type_(memory_type),
      frame_(frame),
      point_count_(point_count) {
  SetFields(fields);
}

bool PointCloudView::SetFields(std::initializer_list<PointFieldDescriptor> fields) {
  schema_valid_ = true;
  field_count_ = 0U;
  for (const PointFieldDescriptor& field : fields) {
    if (!AddField(field)) {
      schema_valid_ = false;
      field_count_ = 0U;
      return false;
    }
  }
  return true;
}

bool PointCloudView::AddField(const PointFieldDescriptor& field) {
  if (field_count_ >= kMaxPointFields || field.element_size() == 0U ||
      field.stride_bytes < field.element_size()) {
    schema_valid_ = false;
    return false;
  }

  if (point_count_ > 0U && field.data == nullptr) {
    schema_valid_ = false;
    return false;
  }

  if (HasField(field.field)) {
    schema_valid_ = false;
    return false;
  }

  fields_[field_count_] = field;
  ++field_count_;
  return true;
}

const PointFieldDescriptor* PointCloudView::FindField(PointField field) const {
  for (uint32_t index = 0U; index < field_count_; ++index) {
    if (fields_[index].field == field) {
      return &fields_[index];
    }
  }
  return nullptr;
}

bool PointCloudView::HasField(PointField field) const {
  return FindField(field) != nullptr;
}

bool PointCloudView::IsValid() const {
  if (!schema_valid_) {
    return false;
  }

  for (uint32_t index = 0U; index < field_count_; ++index) {
    const PointFieldDescriptor& field = fields_[index];
    if (field.element_size() == 0U || field.stride_bytes < field.element_size()) {
      return false;
    }

    if (point_count_ > 0U && field.data == nullptr) {
      return false;
    }
  }

  return true;
}

bool PointCloudView::IsRangeImageCompatible() const {
  if (!IsValid()) {
    return false;
  }

  if (point_count_ == 0U) {
    return true;
  }

  const PointFieldDescriptor* x = FindField(PointField::kX);
  const PointFieldDescriptor* y = FindField(PointField::kY);
  const PointFieldDescriptor* z = FindField(PointField::kZ);
  if (x == nullptr || y == nullptr || z == nullptr) {
    return false;
  }

  if (x->scalar_type != ScalarType::kFloat32 ||
      y->scalar_type != ScalarType::kFloat32 ||
      z->scalar_type != ScalarType::kFloat32) {
    return false;
  }

  const PointFieldDescriptor* intensity = FindField(PointField::kIntensity);
  return intensity == nullptr || intensity->scalar_type == ScalarType::kFloat32;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
