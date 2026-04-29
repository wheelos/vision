#include "vision/lidar/memory/buffer.h"

#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace wheel {
namespace vision {
namespace lidar {

namespace {

bool AllocateManagedMemory(size_t size_bytes, MemoryType memory_type, void** data) {
  if (size_bytes == 0U || data == nullptr) {
    return false;
  }

  switch (memory_type) {
    case MemoryType::kHost: {
      *data = ::operator new(size_bytes, std::nothrow);
      return *data != nullptr;
    }
    case MemoryType::kPinnedHost:
      return cudaMallocHost(data, size_bytes) == cudaSuccess;
    case MemoryType::kDevice:
      return cudaMalloc(data, size_bytes) == cudaSuccess;
    case MemoryType::kUnified:
      return cudaMallocManaged(data, size_bytes) == cudaSuccess;
    default:
      return false;
  }
}

void FreeManagedMemory(void* data, MemoryType memory_type) {
  if (data == nullptr) {
    return;
  }

  switch (memory_type) {
    case MemoryType::kHost:
      ::operator delete(data);
      return;
    case MemoryType::kPinnedHost:
      cudaFreeHost(data);
      return;
    case MemoryType::kDevice:
    case MemoryType::kUnified:
      cudaFree(data);
      return;
    default:
      return;
  }
}

}  // namespace

Buffer::~Buffer() { Reset(); }

Buffer::Buffer(Buffer&& other) noexcept { *this = std::move(other); }

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  Reset();
  data_ = other.data_;
  size_bytes_ = other.size_bytes_;
  memory_type_ = other.memory_type_;
  other.data_ = nullptr;
  other.size_bytes_ = 0U;
  other.memory_type_ = MemoryType::kHost;
  return *this;
}

bool Buffer::Allocate(size_t size_bytes, MemoryType memory_type) {
  Reset();
  if (!AllocateManagedMemory(size_bytes, memory_type, &data_)) {
    return false;
  }

  size_bytes_ = size_bytes;
  memory_type_ = memory_type;
  return true;
}

bool Buffer::Memset(int value) {
  if (!MemsetAsync(value, kDefaultCudaStream)) {
    return false;
  }

  return memory_type_ == MemoryType::kHost ||
         cudaStreamSynchronize(kDefaultCudaStream) == cudaSuccess;
}

bool Buffer::MemsetAsync(int value, CudaStream stream) {
  if (data_ == nullptr || size_bytes_ == 0U) {
    return false;
  }

  if (memory_type_ == MemoryType::kHost) {
    std::memset(data_, value, size_bytes_);
    return true;
  }

  return cudaMemsetAsync(data_, value, size_bytes_, stream) == cudaSuccess;
}

void Buffer::Reset() {
  FreeManagedMemory(data_, memory_type_);
  data_ = nullptr;
  size_bytes_ = 0U;
  memory_type_ = MemoryType::kHost;
}

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
