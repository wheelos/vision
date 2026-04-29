#include "vision/lidar/lidar.h"

#include <cmath>

#include <gtest/gtest.h>

namespace wheel {
namespace vision {
namespace lidar {
namespace {

TEST(CropBoxFilterTest, RemovesPointsInsideBoxAndRejectsNonFinite) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(
      5U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kLidar,
      {{PointField::kX, ScalarType::kFloat32},
       {PointField::kY, ScalarType::kFloat32},
       {PointField::kZ, ScalarType::kFloat32},
       {PointField::kIntensity, ScalarType::kFloat32}}));

  const float xs[5] = {0.0F, 2.0F, 3.0F, NAN, -2.0F};
  const float ys[5] = {0.0F, 0.0F, 0.0F, 0.0F, 1.5F};
  const float zs[5] = {0.0F, 0.5F, 3.0F, 0.0F, 0.2F};
  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) = xs[index];
    *source.mutable_field_element<float>(PointField::kY, index) = ys[index];
    *source.mutable_field_element<float>(PointField::kZ, index) = zs[index];
    *source.mutable_field_element<float>(PointField::kIntensity, index) =
        10.0F + static_cast<float>(index);
  }

  CropBoxFilterWorkspace workspace;
  ASSERT_TRUE(workspace.Allocate(source.point_count()));

  CropBoxFilter filter({.min_x = -1.0F,
                        .max_x = 1.0F,
                        .min_y = -1.0F,
                        .max_y = 1.0F,
                        .min_z = -1.0F,
                        .max_z = 1.0F,
                        .reject_non_finite = true,
                        .mode = CropBoxMode::kRemoveInside});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output, &workspace));

  ASSERT_EQ(output.point_count(), 3U);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 0U), 2.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 1U), 3.0F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kX, 2U), -2.0F);
}

TEST(CropBoxFilterTest, KeepsPointsInsideConfiguredHeightWindow) {
  PointCloudBuffer source;
  ASSERT_TRUE(source.Allocate(
      4U, PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar,
      {{PointField::kX, ScalarType::kFloat32},
       {PointField::kY, ScalarType::kFloat32},
       {PointField::kZ, ScalarType::kFloat32},
       {PointField::kRing, ScalarType::kUint16}}));

  const float zs[4] = {-0.5F, 0.4F, 1.2F, -1.5F};
  for (uint32_t index = 0U; index < source.point_count(); ++index) {
    *source.mutable_field_element<float>(PointField::kX, index) =
        static_cast<float>(index);
    *source.mutable_field_element<float>(PointField::kY, index) = 0.0F;
    *source.mutable_field_element<float>(PointField::kZ, index) = zs[index];
    *source.mutable_field_element<uint16_t>(PointField::kRing, index) =
        static_cast<uint16_t>(index + 1U);
  }

  CropBoxFilterWorkspace workspace;
  ASSERT_TRUE(workspace.Allocate(source.point_count(), MemoryType::kDevice));

  CropBoxFilter filter({.min_z = -1.0F,
                        .max_z = 1.0F,
                        .mode = CropBoxMode::kKeepInside});
  PointCloudBuffer output;
  ASSERT_TRUE(filter.Filter(source.view(), &output, &workspace));

  ASSERT_EQ(output.point_count(), 2U);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kZ, 0U), -0.5F);
  EXPECT_FLOAT_EQ(*output.field_element<float>(PointField::kZ, 1U), 0.4F);
  EXPECT_EQ(*output.field_element<uint16_t>(PointField::kRing, 1U), 2U);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
