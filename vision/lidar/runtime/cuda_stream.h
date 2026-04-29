#pragma once

#include <cuda_runtime_api.h>

namespace wheel {
namespace vision {
namespace lidar {

using CudaStream = cudaStream_t;
constexpr CudaStream kDefaultCudaStream = nullptr;

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
