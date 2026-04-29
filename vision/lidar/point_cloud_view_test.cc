#include "vision/lidar/core/point_cloud.h"

#include "gtest/gtest.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace {

struct InterleavedPoint {
  float x;
  float y;
  float z;
  float intensity;
};

TEST(PointCloudViewTest, AcceptsInterleavedAndPlanarLayouts) {
  InterleavedPoint interleaved_points[2] = {
      {1.0f, 0.0f, 0.0f, 10.0f},
      {0.0f, 1.0f, 0.0f, 20.0f},
  };

  PointCloudView interleaved(
      PointLayout::kInterleaved, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, &interleaved_points[0].x,
           sizeof(InterleavedPoint)},
          {PointField::kY, ScalarType::kFloat32, &interleaved_points[0].y,
           sizeof(InterleavedPoint)},
          {PointField::kZ, ScalarType::kFloat32, &interleaved_points[0].z,
           sizeof(InterleavedPoint)},
          {PointField::kIntensity, ScalarType::kFloat32,
           &interleaved_points[0].intensity, sizeof(InterleavedPoint)},
      });

  float planar_x[2] = {1.0f, 2.0f};
  float planar_y[2] = {0.0f, 0.0f};
  float planar_z[2] = {0.0f, 1.0f};
  PointCloudView planar(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, planar_x, sizeof(float)},
          {PointField::kY, ScalarType::kFloat32, planar_y, sizeof(float)},
          {PointField::kZ, ScalarType::kFloat32, planar_z, sizeof(float)},
      });

  EXPECT_TRUE(interleaved.IsValid());
  EXPECT_TRUE(interleaved.IsRangeImageCompatible());
  EXPECT_TRUE(planar.IsValid());
  EXPECT_TRUE(planar.IsRangeImageCompatible());
  ASSERT_NE(interleaved.FindField(PointField::kIntensity), nullptr);
  EXPECT_EQ(planar.FindField(PointField::kIntensity), nullptr);
}

TEST(PointCloudViewTest, RejectsDuplicateFieldsAndInvalidStride) {
  float points[2] = {1.0f, 2.0f};

  PointCloudView duplicate(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, points, sizeof(float)},
          {PointField::kX, ScalarType::kFloat32, points, sizeof(float)},
      });
  PointCloudView invalid_stride(
      PointLayout::kPlanar, MemoryType::kHost, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, points, 1U},
      });

  EXPECT_EQ(duplicate.field_count(), 0U);
  EXPECT_FALSE(duplicate.IsValid());
  EXPECT_FALSE(duplicate.IsRangeImageCompatible());
  EXPECT_FALSE(invalid_stride.IsValid());
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
