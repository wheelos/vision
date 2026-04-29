#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vision/lidar/memory/buffer.h"

namespace wheel {
namespace vision {
namespace lidar {

class BufferPool {
 public:
  BufferPool() = default;

  bool Init(size_t buffer_size_bytes, uint32_t capacity,
            MemoryType memory_type = MemoryType::kUnified);
  Buffer* Acquire();
  bool Release(Buffer* buffer);
  void Reset();

  inline size_t buffer_size_bytes() const { return buffer_size_bytes_; }
  inline uint32_t capacity() const { return static_cast<uint32_t>(buffers_.size()); }
  inline uint32_t available_count() const {
    return static_cast<uint32_t>(available_indices_.size());
  }

 private:
  size_t buffer_size_bytes_ = 0U;
  MemoryType memory_type_ = MemoryType::kUnified;
  std::vector<std::unique_ptr<Buffer>> buffers_;
  std::vector<size_t> available_indices_;
};

}  // namespace lidar
}  // namespace vision
}  // namespace wheel
