#include "vision/lidar/lidar.h"

#include <gtest/gtest.h>

namespace wheel {
namespace vision {
namespace lidar {
namespace {

TEST(DecimationFilterTest, DecimatesPlanarUnifiedPointCloud) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(
      6U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {{PointField::kX, ScalarType::kFloat32},
       {PointField::kY, ScalarType::kFloat32},
       {PointField::kZ, ScalarType::kFloat32},
       {PointField::kIntensity, ScalarType::kFloat32}}));

  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) =
        static_cast<float>(index);
    *source.mutable_field_element<float>(PointField::kY, index) =
        static_cast<float>(index + 10U);
    *source.mutable_field_element<float>(PointField::kZ, index) =
        static_cast<float>(index + 20U);
    *source.mutable_field_element<float>(PointField::kIntensity, index) =
        static_cast<float>(index + 30U);
  }

  DecimationFilter filter({.stride = 2U, .offset = 1U});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output));
  ASSERT_EQ(output.point_count(), 3U);
  ASSERT_EQ(output.layout(), PointLayout::kPlanar);

  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 0U), 1.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 1U), 3.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 2U), 5.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kIntensity, 2U), 35.0F);
}

TEST(DecimationFilterTest, DecimatesInterleavedInputToInterleavedOutput) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(
      5U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle,
      {{PointField::kX, ScalarType::kFloat32},
       {PointField::kY, ScalarType::kFloat32},
       {PointField::kZ, ScalarType::kFloat32},
       {PointField::kRing, ScalarType::kUint16}}));

  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) =
        static_cast<float>(index * 2U);
    *source.mutable_field_element<float>(PointField::kY, index) =
        static_cast<float>(index * 2U + 1U);
    *source.mutable_field_element<float>(PointField::kZ, index) =
        static_cast<float>(100U + index);
    *source.mutable_field_element<uint16_t>(PointField::kRing, index) =
        static_cast<uint16_t>(index + 7U);
  }

  DecimationFilter filter(
      {.stride = 2U, .offset = 0U, .output_layout = PointLayout::kInterleaved});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output));
  ASSERT_EQ(output.point_count(), 3U);
  ASSERT_EQ(output.layout(), PointLayout::kInterleaved);
  ASSERT_EQ(output.frame(), CoordinateFrame::kVehicle);

  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kY, 1U), 5.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kZ, 2U), 104.0F);
  EXPECT_EQ(*output.field_element<uint16_t>(PointField::kRing, 2U), 11U);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
