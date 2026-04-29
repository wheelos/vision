#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "vision/lidar/runtime/cuda_stream.h"
#include "vision/lidar/types.h"

namespace wheel {
namespace vision {
namespace lidar {

class Buffer {
 public:
  Buffer() = default;
  ~Buffer();

  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  bool Allocate(size_t size_bytes, MemoryType memory_type);
  bool Memset(int value);
  bool MemsetAsync(int value, CudaStream stream = kDefaultCudaStream);
  void Reset();

  inline void* data() { return data_; }
  inline const void* data() const { return data_; }

  template <typename T>
  inline T* mutable_data() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  inline const T* data() const {
    return static_cast<const T*>(data_);
  }

  inline size_t size_bytes() const { return size_bytes_; }
  inline MemoryType memory_type() const { return memory_type_; }

 private:
  void* data_ = nullptr;
  size_t size_bytes_ = 0U;
  MemoryType memory_type_ = MemoryType::kHost;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
