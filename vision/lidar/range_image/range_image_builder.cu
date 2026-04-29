#include "vision/lidar/range_image/range_image_builder.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

namespace wheel {
namespace vision {
namespace lidar {

namespace {

constexpr float kInvalidRange = std::numeric_limits<float>::infinity();

struct DeviceFieldDescriptor {
  const uint8_t* data = nullptr;
  uint32_t stride_bytes = 0U;
};

struct DevicePointCloudView {
  uint32_t point_count = 0U;
  DeviceFieldDescriptor x;
  DeviceFieldDescriptor y;
  DeviceFieldDescriptor z;
  DeviceFieldDescriptor intensity;
  uint8_t has_intensity = 0U;
};

struct DeviceRigidTransform3f {
  float rotation[9];
  float translation[3];
  uint8_t enabled = 0U;
};

struct DeviceRangeImageConfig {
  uint32_t rows = 0U;
  uint32_t cols = 0U;
  float min_range_m = 0.0f;
  float max_range_m = 0.0f;
  float min_azimuth_rad = 0.0f;
  float max_azimuth_rad = 0.0f;
  float min_elevation_rad = 0.0f;
  float max_elevation_rad = 0.0f;
};

inline bool IsDeviceAccessible(MemoryType memory_type) {
  return memory_type == MemoryType::kUnified || memory_type == MemoryType::kDevice;
}

inline DeviceFieldDescriptor MakeDeviceField(
    const PointFieldDescriptor* descriptor) {
  DeviceFieldDescriptor field;
  if (descriptor != nullptr) {
    field.data = static_cast<const uint8_t*>(descriptor->data);
    field.stride_bytes = descriptor->stride_bytes;
  }
  return field;
}

inline DeviceRigidTransform3f MakeDeviceTransform(
    const RigidTransform3f* lidar_from_input) {
  DeviceRigidTransform3f transform;
  if (lidar_from_input == nullptr) {
    return transform;
  }

  transform.enabled = 1U;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      transform.rotation[row * 3 + col] = lidar_from_input->rotation(row, col);
    }
    transform.translation[row] = lidar_from_input->translation(row);
  }
  return transform;
}

inline DeviceRangeImageConfig MakeDeviceConfig(const RangeImageConfig& config) {
  DeviceRangeImageConfig device_config;
  device_config.rows = config.rows;
  device_config.cols = config.cols;
  device_config.min_range_m = config.min_range_m;
  device_config.max_range_m = config.max_range_m;
  device_config.min_azimuth_rad = config.min_azimuth_rad;
  device_config.max_azimuth_rad = config.max_azimuth_rad;
  device_config.min_elevation_rad = config.min_elevation_rad;
  device_config.max_elevation_rad = config.max_elevation_rad;
  return device_config;
}

__host__ __device__ inline uint32_t FloatToBits(float value) {
  union {
    float f;
    uint32_t u;
  } converter;
  converter.f = value;
  return converter.u;
}

__host__ __device__ inline float BitsToFloat(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } converter;
  converter.u = bits;
  return converter.f;
}

__host__ __device__ inline uint64_t PackRangeIntensity(float range,
                                                       float intensity) {
  return (static_cast<uint64_t>(FloatToBits(range)) << 32U) |
         static_cast<uint64_t>(FloatToBits(intensity));
}

__host__ __device__ inline float UnpackRange(uint64_t packed) {
  return BitsToFloat(static_cast<uint32_t>(packed >> 32U));
}

__host__ __device__ inline float UnpackIntensity(uint64_t packed) {
  return BitsToFloat(static_cast<uint32_t>(packed & 0xffffffffULL));
}

__device__ inline float LoadScalar(const DeviceFieldDescriptor& field,
                                   uint32_t index) {
  const uint8_t* element_address = field.data + static_cast<size_t>(index) * field.stride_bytes;
  return *reinterpret_cast<const float*>(element_address);
}

__device__ inline void ApplyTransform(const DeviceRigidTransform3f& transform,
                                      float* x, float* y, float* z) {
  if (transform.enabled == 0U) {
    return;
  }

  const float input_x = *x;
  const float input_y = *y;
  const float input_z = *z;
  *x = transform.rotation[0] * input_x + transform.rotation[1] * input_y +
       transform.rotation[2] * input_z + transform.translation[0];
  *y = transform.rotation[3] * input_x + transform.rotation[4] * input_y +
       transform.rotation[5] * input_z + transform.translation[1];
  *z = transform.rotation[6] * input_x + transform.rotation[7] * input_y +
       transform.rotation[8] * input_z + transform.translation[2];
}

__device__ inline void AtomicMinPacked(uint64_t* address, float range,
                                       float intensity) {
  const uint64_t candidate = PackRangeIntensity(range, intensity);
  unsigned long long* raw_address =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *raw_address;

  while (range < UnpackRange(static_cast<uint64_t>(observed))) {
    const unsigned long long previous =
        atomicCAS(raw_address, observed, static_cast<unsigned long long>(candidate));
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
}

__global__ void InitializePackedKernel(uint64_t* packed, uint32_t pixel_count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) {
    return;
  }

  packed[index] = PackRangeIntensity(kInvalidRange, 0.0f);
}

__global__ void ProjectRangeImageKernel(DevicePointCloudView point_cloud,
                                        DeviceRigidTransform3f transform,
                                        DeviceRangeImageConfig config,
                                        uint64_t* packed) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= point_cloud.point_count) {
    return;
  }

  float x = LoadScalar(point_cloud.x, index);
  float y = LoadScalar(point_cloud.y, index);
  float z = LoadScalar(point_cloud.z, index);
  ApplyTransform(transform, &x, &y, &z);
  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
    return;
  }

  const float range = sqrtf(x * x + y * y + z * z);
  if (!isfinite(range) || range < config.min_range_m ||
      range > config.max_range_m) {
    return;
  }

  const float azimuth = atan2f(y, x);
  if (!isfinite(azimuth) || azimuth < config.min_azimuth_rad ||
      azimuth > config.max_azimuth_rad) {
    return;
  }

  const float normalized_z = z / range;
  if (normalized_z < -1.0f || normalized_z > 1.0f) {
    return;
  }

  const float elevation = asinf(normalized_z);
  if (!isfinite(elevation) || elevation < config.min_elevation_rad ||
      elevation > config.max_elevation_rad) {
    return;
  }

  const float col_ratio =
      (azimuth - config.min_azimuth_rad) /
      (config.max_azimuth_rad - config.min_azimuth_rad);
  const float row_ratio =
      (config.max_elevation_rad - elevation) /
      (config.max_elevation_rad - config.min_elevation_rad);

  int col = static_cast<int>(floorf(col_ratio * static_cast<float>(config.cols)));
  int row = static_cast<int>(floorf(row_ratio * static_cast<float>(config.rows)));
  if (col == static_cast<int>(config.cols)) {
    col = static_cast<int>(config.cols) - 1;
  }
  if (row == static_cast<int>(config.rows)) {
    row = static_cast<int>(config.rows) - 1;
  }
  if (row < 0 || row >= static_cast<int>(config.rows) || col < 0 ||
      col >= static_cast<int>(config.cols)) {
    return;
  }

  float intensity = 0.0f;
  if (point_cloud.has_intensity != 0U) {
    intensity = LoadScalar(point_cloud.intensity, index);
    if (!isfinite(intensity)) {
      intensity = 0.0f;
    }
  }

  const uint32_t pixel_index = static_cast<uint32_t>(row) * config.cols +
                               static_cast<uint32_t>(col);
  AtomicMinPacked(&packed[pixel_index], range, intensity);
}

__global__ void FinalizeRangeImageKernel(const uint64_t* packed, float* range,
                                         float* intensity, uint8_t* valid_mask,
                                         uint32_t pixel_count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) {
    return;
  }

  const uint64_t packed_value = packed[index];
  const float range_value = UnpackRange(packed_value);
  range[index] = range_value;
  intensity[index] = UnpackIntensity(packed_value);
  valid_mask[index] = isfinite(range_value) ? 1U : 0U;
}

inline bool CheckCudaSuccess(cudaError_t status) {
  return status == cudaSuccess;
}

}  // namespace

bool RangeImageBuffer::Allocate(const RangeImageConfig& config,
                                MemoryType memory_type) {
  if (!config.IsValid()) {
    return false;
  }

  const size_t pixel_count = config.pixel_count();
  if (!range_.Allocate(pixel_count * sizeof(float), memory_type) ||
      !intensity_.Allocate(pixel_count * sizeof(float), memory_type) ||
      !valid_mask_.Allocate(pixel_count * sizeof(uint8_t), memory_type)) {
    return false;
  }

  rows_ = config.rows;
  cols_ = config.cols;
  return true;
}

bool RangeImageWorkspace::Allocate(const RangeImageConfig& config,
                                   MemoryType memory_type) {
  if (!config.IsValid()) {
    return false;
  }

  if (!packed_range_intensity_.Allocate(config.pixel_count() * sizeof(uint64_t),
                                        memory_type)) {
    return false;
  }

  pixel_count_ = config.pixel_count();
  return true;
}

bool RangeImageBuilder::Build(const PointCloudView& point_cloud,
                              RangeImageBuffer* output,
                              RangeImageWorkspace* workspace,
                              const RigidTransform3f* lidar_from_input,
                              CudaStream stream) const {
  if (!BuildAsync(point_cloud, output, workspace, lidar_from_input, stream)) {
    return false;
  }

  return CheckCudaSuccess(cudaStreamSynchronize(stream));
}

bool RangeImageBuilder::BuildAsync(const PointCloudView& point_cloud,
                                   RangeImageBuffer* output,
                                   RangeImageWorkspace* workspace,
                                   const RigidTransform3f* lidar_from_input,
                                   CudaStream stream) const {
  if (!config_.IsValid() || output == nullptr || workspace == nullptr ||
      !point_cloud.IsRangeImageCompatible()) {
    return false;
  }

  if (output->rows() != config_.rows || output->cols() != config_.cols ||
      workspace->pixel_count() != config_.pixel_count()) {
    return false;
  }

  if (!IsDeviceAccessible(point_cloud.memory_type()) ||
      !IsDeviceAccessible(output->memory_type()) ||
      !IsDeviceAccessible(workspace->memory_type())) {
    return false;
  }

  if (point_cloud.frame() != CoordinateFrame::kLidar &&
      lidar_from_input == nullptr) {
    return false;
  }

  if (lidar_from_input != nullptr && !lidar_from_input->IsFinite()) {
    return false;
  }

  const PointFieldDescriptor* x = point_cloud.FindField(PointField::kX);
  const PointFieldDescriptor* y = point_cloud.FindField(PointField::kY);
  const PointFieldDescriptor* z = point_cloud.FindField(PointField::kZ);
  const PointFieldDescriptor* intensity =
      point_cloud.FindField(PointField::kIntensity);

  DevicePointCloudView device_point_cloud;
  device_point_cloud.point_count = point_cloud.point_count();
  device_point_cloud.x = MakeDeviceField(x);
  device_point_cloud.y = MakeDeviceField(y);
  device_point_cloud.z = MakeDeviceField(z);
  device_point_cloud.intensity = MakeDeviceField(intensity);
  device_point_cloud.has_intensity = intensity == nullptr ? 0U : 1U;

  const DeviceRigidTransform3f device_transform =
      MakeDeviceTransform(lidar_from_input);
  const DeviceRangeImageConfig device_config = MakeDeviceConfig(config_);

  constexpr uint32_t kThreadsPerBlock = 256U;
  const uint32_t pixel_blocks =
      (config_.pixel_count() + kThreadsPerBlock - 1U) / kThreadsPerBlock;
  InitializePackedKernel<<<pixel_blocks, kThreadsPerBlock, 0, stream>>>(
      workspace->packed_range_intensity(), config_.pixel_count());
  if (!CheckCudaSuccess(cudaGetLastError())) {
    return false;
  }

  if (point_cloud.point_count() > 0U) {
    const uint32_t point_blocks =
        (point_cloud.point_count() + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    ProjectRangeImageKernel<<<point_blocks, kThreadsPerBlock, 0, stream>>>(
        device_point_cloud, device_transform, device_config,
        workspace->packed_range_intensity());
    if (!CheckCudaSuccess(cudaGetLastError())) {
      return false;
    }
  }

  FinalizeRangeImageKernel<<<pixel_blocks, kThreadsPerBlock, 0, stream>>>(
      workspace->packed_range_intensity(), output->range(), output->intensity(),
      output->valid_mask(), config_.pixel_count());
  if (!CheckCudaSuccess(cudaGetLastError())) {
    return false;
  }

  return true;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
