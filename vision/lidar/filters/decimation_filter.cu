#include "vision/lidar/filters/decimation_filter.h"

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

uint32_t ComputeOutputCount(uint32_t point_count,
                            const DecimationFilterConfig& config) {
  if (config.offset >= point_count) {
    return 0U;
  }

  return 1U + (point_count - 1U - config.offset) / config.stride;
}

__global__ void DecimatePointCloudKernel(internal::DevicePointCloudView source,
                                         internal::MutableDevicePointCloudView output,
                                         uint32_t stride, uint32_t offset) {
  const uint32_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= output.point_count) {
    return;
  }

  const uint32_t source_index = offset + output_index * stride;
  for (uint32_t field_index = 0U; field_index < source.field_count; ++field_index) {
    internal::CopyFieldElement(source.fields[field_index], source_index,
                               output.fields[field_index], output_index);
  }
}

}  // namespace

bool DecimationFilter::Filter(const PointCloudView& source,
                              PointCloudBuffer* output,
                              CudaStream stream) const {
  if (!FilterAsync(source, output, stream)) {
    return false;
  }

  return cudaStreamSynchronize(stream) == cudaSuccess;
}

bool DecimationFilter::FilterAsync(const PointCloudView& source,
                                   PointCloudBuffer* output,
                                   CudaStream stream) const {
  if (!config_.IsValid() || output == nullptr || !source.IsValid() ||
      !IsDeviceAccessible(source.memory_type())) {
    return false;
  }

  const uint32_t output_count = ComputeOutputCount(source.point_count(), config_);
  if (!output->AllocateLike(source, output_count, config_.output_layout)) {
    return false;
  }

  if (!IsDeviceAccessible(output->memory_type()) || output_count == 0U) {
    return output_count == 0U;
  }

  const internal::DevicePointCloudView device_source =
      internal::MakeDevicePointCloudView(source);
  const internal::MutableDevicePointCloudView device_output =
      internal::MakeMutableDevicePointCloudView(output->view());

  constexpr uint32_t kThreadsPerBlock = 256U;
  const uint32_t blocks = (output_count + kThreadsPerBlock - 1U) / kThreadsPerBlock;
  DecimatePointCloudKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      device_source, device_output, config_.stride, config_.offset);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
