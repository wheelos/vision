#include "vision/lidar/filters/crop_box_filter.h"

#include <cmath>

#include <cuda_runtime.h>

#include "vision/lidar/filters/internal/device_filter_utils.cuh"

namespace wheel {
namespace vision {
namespace lidar {

namespace {

bool IsDeviceAccessible(MemoryType memory_type) {
  return memory_type == MemoryType::kUnified ||
         memory_type == MemoryType::kDevice;
}

struct DeviceCropBoxFilterConfig {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
  uint8_t reject_non_finite;
  uint8_t keep_inside;
};

DeviceCropBoxFilterConfig MakeDeviceConfig(const CropBoxFilterConfig& config) {
  DeviceCropBoxFilterConfig device_config;
  device_config.min_x = config.min_x;
  device_config.max_x = config.max_x;
  device_config.min_y = config.min_y;
  device_config.max_y = config.max_y;
  device_config.min_z = config.min_z;
  device_config.max_z = config.max_z;
  device_config.reject_non_finite = config.reject_non_finite ? 1U : 0U;
  device_config.keep_inside = config.mode == CropBoxMode::kKeepInside ? 1U : 0U;
  return device_config;
}

__device__ inline bool IsPointSelected(float x, float y, float z,
                                       const DeviceCropBoxFilterConfig& config) {
  if (config.reject_non_finite &&
      (!isfinite(x) || !isfinite(y) || !isfinite(z))) {
    return false;
  }

  const bool inside = x >= config.min_x && x <= config.max_x &&
                      y >= config.min_y && y <= config.max_y &&
                      z >= config.min_z && z <= config.max_z;
  return config.keep_inside != 0U ? inside : !inside;
}

__global__ void SelectCropBoxIndicesKernel(
    internal::DeviceFieldDescriptor x, internal::DeviceFieldDescriptor y,
    internal::DeviceFieldDescriptor z, uint32_t point_count,
    DeviceCropBoxFilterConfig config, uint32_t* selected_indices,
    uint32_t* selected_count) {
  const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= point_count) {
    return;
  }

  const float point_x = internal::LoadFloat(x, point_index);
  const float point_y = internal::LoadFloat(y, point_index);
  const float point_z = internal::LoadFloat(z, point_index);
  if (!IsPointSelected(point_x, point_y, point_z, config)) {
    return;
  }

  const uint32_t output_index = atomicAdd(selected_count, 1U);
  selected_indices[output_index] = point_index;
}

__global__ void GatherSelectedPointsKernel(
    internal::DevicePointCloudView source,
    internal::MutableDevicePointCloudView output,
    const uint32_t* selected_indices) {
  const uint32_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= output.point_count) {
    return;
  }

  const uint32_t source_index = selected_indices[output_index];
  for (uint32_t field_index = 0U; field_index < source.field_count; ++field_index) {
    internal::CopyFieldElement(source.fields[field_index], source_index,
                               output.fields[field_index], output_index);
  }
}

}  // namespace

bool CropBoxFilterWorkspace::Allocate(uint32_t max_points,
                                      MemoryType index_memory_type) {
  Reset();
  if (max_points == 0U || !IsDeviceAccessible(index_memory_type)) {
    return false;
  }

  if (!selected_indices_.Allocate(static_cast<size_t>(max_points) * sizeof(uint32_t),
                                  index_memory_type) ||
      !selected_count_.Allocate(sizeof(uint32_t), MemoryType::kUnified)) {
    Reset();
    return false;
  }

  max_points_ = max_points;
  return Clear(kDefaultCudaStream);
}

bool CropBoxFilterWorkspace::Clear(CudaStream stream) {
  if (selected_count_.size_bytes() != sizeof(uint32_t)) {
    return false;
  }

  return selected_count_.MemsetAsync(0, stream);
}

void CropBoxFilterWorkspace::Reset() {
  selected_indices_.Reset();
  selected_count_.Reset();
  max_points_ = 0U;
}

bool CropBoxFilter::Filter(const PointCloudView& source, PointCloudBuffer* output,
                           CropBoxFilterWorkspace* workspace,
                           CudaStream stream) const {
  if (!config_.IsValid() || output == nullptr || workspace == nullptr ||
      !source.IsValid() || source.point_count() > workspace->max_points() ||
      !IsDeviceAccessible(source.memory_type()) ||
      !IsDeviceAccessible(workspace->index_memory_type())) {
    return false;
  }

  const PointFieldDescriptor* x = source.FindField(PointField::kX);
  const PointFieldDescriptor* y = source.FindField(PointField::kY);
  const PointFieldDescriptor* z = source.FindField(PointField::kZ);
  if (x == nullptr || y == nullptr || z == nullptr ||
      x->scalar_type != ScalarType::kFloat32 ||
      y->scalar_type != ScalarType::kFloat32 ||
      z->scalar_type != ScalarType::kFloat32) {
    return false;
  }

  if (!workspace->Clear(stream)) {
    return false;
  }

  if (source.point_count() > 0U) {
    constexpr uint32_t kThreadsPerBlock = 256U;
    const uint32_t blocks =
        (source.point_count() + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    SelectCropBoxIndicesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        internal::MakeDeviceField(*x), internal::MakeDeviceField(*y),
        internal::MakeDeviceField(*z), source.point_count(),
        MakeDeviceConfig(config_), workspace->selected_indices(),
        workspace->selected_count());
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }
  }

  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }

  const uint32_t selected_count = *workspace->selected_count();
  if (!output->AllocateLike(source, selected_count, config_.output_layout)) {
    return false;
  }

  if (!IsDeviceAccessible(output->memory_type()) || selected_count == 0U) {
    return selected_count == 0U;
  }

  const internal::DevicePointCloudView device_source =
      internal::MakeDevicePointCloudView(source);
  const internal::MutableDevicePointCloudView device_output =
      internal::MakeMutableDevicePointCloudView(output->view());
  constexpr uint32_t kThreadsPerBlock = 256U;
  const uint32_t blocks =
      (selected_count + kThreadsPerBlock - 1U) / kThreadsPerBlock;
  GatherSelectedPointsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      device_source, device_output, workspace->selected_indices());
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
