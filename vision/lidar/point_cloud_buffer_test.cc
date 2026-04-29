#include "vision/lidar/core/point_cloud_buffer.h"

#include "gtest/gtest.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace {

TEST(PointCloudBufferTest, AllocatesPlanarBufferAndExposesFieldAccessors) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(
      3U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kY, ScalarType::kFloat32},
          {PointField::kZ, ScalarType::kFloat32},
          {PointField::kIntensity, ScalarType::kFloat32},
      }));

  ASSERT_NE(buffer.mutable_field_element<float>(PointField::kX, 0U), nullptr);
  *buffer.mutable_field_element<float>(PointField::kX, 0U) = 1.0f;
  *buffer.mutable_field_element<float>(PointField::kY, 0U) = 2.0f;
  *buffer.mutable_field_element<float>(PointField::kZ, 0U) = 3.0f;
  *buffer.mutable_field_element<float>(PointField::kIntensity, 0U) = 4.0f;

  EXPECT_EQ(buffer.layout(), PointLayout::kPlanar);
  EXPECT_EQ(buffer.point_count(), 3U);
  EXPECT_EQ(buffer.field_count(), 4U);
  EXPECT_EQ(buffer.point_stride_bytes(), 0U);
  EXPECT_TRUE(buffer.view().IsValid());
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 0U), 1.0f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kY, 0U), 2.0f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kZ, 0U), 3.0f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kIntensity, 0U), 4.0f);
}

TEST(PointCloudBufferTest, AllocatesInterleavedBufferWithExpectedStride) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(
      2U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kY, ScalarType::kFloat32},
          {PointField::kZ, ScalarType::kFloat32},
      }));

  ASSERT_NE(buffer.mutable_field_element<float>(PointField::kX, 1U), nullptr);
  *buffer.mutable_field_element<float>(PointField::kX, 1U) = 1.5f;
  *buffer.mutable_field_element<float>(PointField::kY, 1U) = -0.5f;
  *buffer.mutable_field_element<float>(PointField::kZ, 1U) = 0.25f;

  const PointFieldDescriptor* x = buffer.FindField(PointField::kX);
  const PointFieldDescriptor* y = buffer.FindField(PointField::kY);
  const PointFieldDescriptor* z = buffer.FindField(PointField::kZ);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_NE(z, nullptr);
  EXPECT_EQ(x->stride_bytes, sizeof(float) * 3U);
  EXPECT_EQ(y->stride_bytes, sizeof(float) * 3U);
  EXPECT_EQ(z->stride_bytes, sizeof(float) * 3U);
  EXPECT_EQ(buffer.point_stride_bytes(), sizeof(float) * 3U);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 1U), 1.5f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kY, 1U), -0.5f);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kZ, 1U), 0.25f);
}

TEST(PointCloudBufferTest, RejectsInvalidFieldSpecifications) {
  PointCloudBuffer duplicate_fields;
  PointCloudBuffer unknown_scalar_type;

  EXPECT_FALSE(duplicate_fields.Allocate(
      1U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kX, ScalarType::kFloat32},
      }));
  EXPECT_FALSE(unknown_scalar_type.Allocate(
      1U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kUnknown},
      }));
}

TEST(PointCloudBufferTest, SupportsZeroPointSchemasWithoutBackingStorage) {
  PointCloudBuffer planar_buffer;
  PointCloudBuffer interleaved_buffer;

  ASSERT_TRUE(planar_buffer.Allocate(
      0U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kY, ScalarType::kFloat32},
          {PointField::kZ, ScalarType::kFloat32},
      }));
  ASSERT_TRUE(interleaved_buffer.Allocate(
      0U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kIntensity, ScalarType::kUint8},
      }));

  EXPECT_TRUE(planar_buffer.view().IsValid());
  EXPECT_TRUE(interleaved_buffer.view().IsValid());
  EXPECT_EQ(planar_buffer.field_data<float>(PointField::kX), nullptr);
  EXPECT_EQ(interleaved_buffer.field_data<float>(PointField::kX), nullptr);
}

TEST(PointCloudBufferTest, SupportsMixedScalarTypesInInterleavedLayout) {
  PointCloudBuffer buffer;
  ASSERT_TRUE(buffer.Allocate(
      2U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kIntensity, ScalarType::kUint8},
          {PointField::kRing, ScalarType::kUint16},
          {PointField::kY, ScalarType::kFloat32},
      }));

  *buffer.mutable_field_element<float>(PointField::kX, 0U) = 2.0f;
  *buffer.mutable_field_element<uint8_t>(PointField::kIntensity, 0U) = 7U;
  *buffer.mutable_field_element<uint16_t>(PointField::kRing, 0U) = 12U;
  *buffer.mutable_field_element<float>(PointField::kY, 0U) = -1.0f;

  const PointFieldDescriptor* x = buffer.FindField(PointField::kX);
  const PointFieldDescriptor* intensity = buffer.FindField(PointField::kIntensity);
  const PointFieldDescriptor* ring = buffer.FindField(PointField::kRing);
  const PointFieldDescriptor* y = buffer.FindField(PointField::kY);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(intensity, nullptr);
  ASSERT_NE(ring, nullptr);
  ASSERT_NE(y, nullptr);
  EXPECT_EQ(buffer.point_stride_bytes(), 12U);
  EXPECT_EQ(x->stride_bytes, 12U);
  EXPECT_EQ(intensity->stride_bytes, 12U);
  EXPECT_EQ(ring->stride_bytes, 12U);
  EXPECT_EQ(y->stride_bytes, 12U);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kX, 0U), 2.0f);
  EXPECT_EQ(*buffer.field_element<uint8_t>(PointField::kIntensity, 0U), 7U);
  EXPECT_EQ(*buffer.field_element<uint16_t>(PointField::kRing, 0U), 12U);
  EXPECT_FLOAT_EQ(*buffer.field_element<float>(PointField::kY, 0U), -1.0f);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
