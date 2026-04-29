#include "vision/lidar/lidar.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

namespace wheel {
namespace vision {
namespace lidar {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kQuarterPi = kPi / 4.0f;
constexpr float kTolerance = 1e-5f;

uint32_t PixelIndex(uint32_t row, uint32_t col, uint32_t cols) {
  return row * cols + col;
}

RangeImageConfig MakeTestConfig() {
  RangeImageConfig config;
  config.rows = 4U;
  config.cols = 8U;
  config.min_range_m = 0.1f;
  config.max_range_m = 10.0f;
  config.min_azimuth_rad = -kPi;
  config.max_azimuth_rad = kPi;
  config.min_elevation_rad = -kQuarterPi;
  config.max_elevation_rad = kQuarterPi;
  return config;
}

TEST(RangeImageTest, ProjectsPlanarPointsAndKeepsNearestHit) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  Buffer x;
  Buffer y;
  Buffer z;
  Buffer intensity;
  ASSERT_TRUE(x.Allocate(4U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(y.Allocate(4U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(z.Allocate(4U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(intensity.Allocate(4U * sizeof(float), MemoryType::kUnified));

  float* x_data = x.mutable_data<float>();
  float* y_data = y.mutable_data<float>();
  float* z_data = z.mutable_data<float>();
  float* intensity_data = intensity.mutable_data<float>();
  x_data[0] = 1.0f;
  y_data[0] = 0.0f;
  z_data[0] = 0.0f;
  intensity_data[0] = 10.0f;

  x_data[1] = 2.0f;
  y_data[1] = 0.0f;
  z_data[1] = 0.0f;
  intensity_data[1] = 20.0f;

  x_data[2] = 0.0f;
  y_data[2] = 1.0f;
  z_data[2] = 0.0f;
  intensity_data[2] = 30.0f;

  x_data[3] = std::numeric_limits<float>::quiet_NaN();
  y_data[3] = 0.0f;
  z_data[3] = 0.0f;
  intensity_data[3] = 40.0f;

  PointCloudView point_cloud(
      PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar, 4U,
      {
          {PointField::kX, ScalarType::kFloat32, x.data(), sizeof(float)},
          {PointField::kY, ScalarType::kFloat32, y.data(), sizeof(float)},
          {PointField::kZ, ScalarType::kFloat32, z.data(), sizeof(float)},
          {PointField::kIntensity, ScalarType::kFloat32, intensity.data(),
           sizeof(float)},
      });

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(builder.Build(point_cloud, &output, &workspace));

  const uint32_t center_pixel = PixelIndex(2U, 4U, config.cols);
  const uint32_t left_pixel = PixelIndex(2U, 6U, config.cols);
  EXPECT_EQ(output.valid_mask()[center_pixel], 1U);
  EXPECT_NEAR(output.range()[center_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[center_pixel], 10.0f, kTolerance);

  EXPECT_EQ(output.valid_mask()[left_pixel], 1U);
  EXPECT_NEAR(output.range()[left_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[left_pixel], 30.0f, kTolerance);

  const uint32_t invalid_pixel = PixelIndex(0U, 0U, config.cols);
  EXPECT_EQ(output.valid_mask()[invalid_pixel], 0U);
  EXPECT_TRUE(std::isinf(output.range()[invalid_pixel]));
  EXPECT_NEAR(output.intensity()[invalid_pixel], 0.0f, kTolerance);
}

TEST(RangeImageTest, ProjectsInterleavedPointsWithRigidTransform) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  struct Point {
    float x;
    float y;
    float z;
  };

  Buffer points;
  ASSERT_TRUE(points.Allocate(sizeof(Point), MemoryType::kUnified));
  Point* point = points.mutable_data<Point>();
  point[0] = {0.0f, 1.0f, 0.0f};

  PointCloudView point_cloud(
      PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kVehicle, 1U,
      {
          {PointField::kX, ScalarType::kFloat32, &point[0].x, sizeof(Point)},
          {PointField::kY, ScalarType::kFloat32, &point[0].y, sizeof(Point)},
          {PointField::kZ, ScalarType::kFloat32, &point[0].z, sizeof(Point)},
      });

  RigidTransform3f lidar_from_vehicle;
  lidar_from_vehicle.rotation << 0.0f, 1.0f, 0.0f,
      -1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f;

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(
      builder.Build(point_cloud, &output, &workspace, &lidar_from_vehicle));

  const uint32_t center_pixel = PixelIndex(2U, 4U, config.cols);
  EXPECT_EQ(output.valid_mask()[center_pixel], 1U);
  EXPECT_NEAR(output.range()[center_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[center_pixel], 0.0f, kTolerance);
}

TEST(RangeImageTest, MapsElevationBoundariesToImageRows) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  Buffer x;
  Buffer y;
  Buffer z;
  ASSERT_TRUE(x.Allocate(2U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(y.Allocate(2U * sizeof(float), MemoryType::kUnified));
  ASSERT_TRUE(z.Allocate(2U * sizeof(float), MemoryType::kUnified));

  float* x_data = x.mutable_data<float>();
  float* y_data = y.mutable_data<float>();
  float* z_data = z.mutable_data<float>();
  x_data[0] = 1.0f;
  y_data[0] = 0.0f;
  z_data[0] = 1.0f;
  x_data[1] = 1.0f;
  y_data[1] = 0.0f;
  z_data[1] = -1.0f;

  PointCloudView point_cloud(
      PointLayout::kPlanar, MemoryType::kUnified, CoordinateFrame::kLidar, 2U,
      {
          {PointField::kX, ScalarType::kFloat32, x.data(), sizeof(float)},
          {PointField::kY, ScalarType::kFloat32, y.data(), sizeof(float)},
          {PointField::kZ, ScalarType::kFloat32, z.data(), sizeof(float)},
      });

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(builder.Build(point_cloud, &output, &workspace));

  const uint32_t top_pixel = PixelIndex(0U, 4U, config.cols);
  const uint32_t bottom_pixel = PixelIndex(3U, 4U, config.cols);
  EXPECT_EQ(output.valid_mask()[top_pixel], 1U);
  EXPECT_EQ(output.valid_mask()[bottom_pixel], 1U);
  EXPECT_NEAR(output.range()[top_pixel], std::sqrt(2.0f), 1e-4f);
  EXPECT_NEAR(output.range()[bottom_pixel], std::sqrt(2.0f), 1e-4f);
}

TEST(RangeImageTest, ProjectsInterleavedPointCloudBufferView) {
  const RangeImageConfig config = MakeTestConfig();
  RangeImageBuilder builder(config);

  PointCloudBuffer point_cloud;
  ASSERT_TRUE(point_cloud.Allocate(
      2U, PointLayout::kInterleaved, MemoryType::kUnified,
      CoordinateFrame::kLidar,
      {
          {PointField::kX, ScalarType::kFloat32},
          {PointField::kY, ScalarType::kFloat32},
          {PointField::kZ, ScalarType::kFloat32},
          {PointField::kIntensity, ScalarType::kFloat32},
      }));

  *point_cloud.mutable_field_element<float>(PointField::kX, 0U) = 1.0f;
  *point_cloud.mutable_field_element<float>(PointField::kY, 0U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kZ, 0U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kIntensity, 0U) = 5.0f;

  *point_cloud.mutable_field_element<float>(PointField::kX, 1U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kY, 1U) = 1.0f;
  *point_cloud.mutable_field_element<float>(PointField::kZ, 1U) = 0.0f;
  *point_cloud.mutable_field_element<float>(PointField::kIntensity, 1U) = 8.0f;

  RangeImageBuffer output;
  RangeImageWorkspace workspace;
  ASSERT_TRUE(output.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(workspace.Allocate(config, MemoryType::kUnified));
  ASSERT_TRUE(builder.Build(point_cloud.view(), &output, &workspace));

  const uint32_t center_pixel = PixelIndex(2U, 4U, config.cols);
  const uint32_t left_pixel = PixelIndex(2U, 6U, config.cols);
  EXPECT_EQ(output.valid_mask()[center_pixel], 1U);
  EXPECT_EQ(output.valid_mask()[left_pixel], 1U);
  EXPECT_NEAR(output.range()[center_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[center_pixel], 5.0f, kTolerance);
  EXPECT_NEAR(output.range()[left_pixel], 1.0f, kTolerance);
  EXPECT_NEAR(output.intensity()[left_pixel], 8.0f, kTolerance);
}

}  // namespace
}  // namespace lidar
}  // namespace vision
}  // namespace wheel
