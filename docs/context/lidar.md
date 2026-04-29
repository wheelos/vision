# Lidar Module Context

This document defines the current public architecture for `vision/lidar/`, whose
goal is to grow into an open source replacement for the foundational lidar
capabilities commonly consumed from DriveWorks.

## Purpose

The target is not a one-off Range Image kernel. The target is a reusable Lidar
foundation layer with stable data interfaces, explicit memory ownership, GPU
execution, and testable module boundaries so later features can be added without
rewriting the core:

- point cloud data representation
- frame transforms and coordinate normalization
- preallocated buffers and buffer pools
- GPU-native Range Image generation
- future filters, registration, clustering, and fusion modules

## Replacement strategy relative to DriveWorks

The implementation is intentionally staged around the same capability layers that
production Lidar stacks rely on:

1. **Data representation and memory management**
   - support interleaved and planar point layouts
   - keep public interfaces zero-copy friendly
   - avoid large hidden allocations in algorithms
2. **Preprocessing**
   - start with Range Image because it is a common downstream representation
   - keep output channels explicit and stable
3. **Transformation**
   - normalize point clouds into the lidar frame through rigid transforms
4. **Future parity layers**
   - downsampling, outlier removal, clustering, registration, and accumulation

The first implementation goes directly to **GPU**, not CPU fallback, so the
public API is shaped around device-accessible memory from day one.

## Canonical package boundary

The lidar code lives under:

```text
vision/lidar/
```

The camera code now lives separately under `vision/camera/`. This split is
intentional and should remain stable because the lidar package is expected to
grow around GPU memory management, stream-aware kernels, and DriveWorks-style
throughput concerns.

## Current module layout

```text
vision/lidar/
  BUILD.bazel
  lidar.h
  types.h
  memory/
    buffer.h
    buffer.cu
    buffer_pool.h
    buffer_pool.cc
  core/
    point_cloud.h
    point_cloud.cc
    point_cloud_buffer.h
    point_cloud_buffer.cc
    rigid_transform.h
  runtime/
    cuda_stream.h
  filters/
    crop_box_filter.h
    crop_box_filter.cu
    decimation_filter.h
    decimation_filter.cu
  range_image/
    range_image.h
    range_image_builder.h
    range_image_builder.cu
  decimation_filter_test.cc
  crop_box_filter_test.cc
  point_cloud_view_test.cc
  point_cloud_buffer_test.cc
  range_image_test.cc
```

Planned next foundational subdirectories:

```text
vision/lidar/
  filters/
  registration/
  segmentation/
  fusion/
```

## Public API boundaries

The current public targets are:

- `@vision//vision/lidar:types`
- `@vision//vision/lidar:buffer`
- `@vision//vision/lidar:buffer_pool`
- `@vision//vision/lidar:core`
- `@vision//vision/lidar:runtime`
- `@vision//vision/lidar:filters`
- `@vision//vision/lidar:range_image`
- `@vision//vision/lidar:lidar`

The umbrella header for external consumers is:

```cpp
#include "vision/lidar/lidar.h"
```

## Data model invariants

### Point cloud input

`PointCloudView` is the zero-copy read interface for algorithms.

Invariants:

- point layout may be `interleaved` or `planar`
- `x`, `y`, and `z` are required for Range Image generation and crop-box
  filtering
- `intensity` is optional
- Range Image GPU build currently accepts **unified** or **device** memory
- non-finite samples are ignored during projection
- planar/SoA-friendly access remains the preferred layout for future kernels

### Frame semantics

Input clouds declare their source frame:

- `kLidar`
- `kVehicle`
- `kWorld`

If the input is not already in `kLidar`, callers must provide
`lidar_from_input` so projection is always performed in the lidar frame.

### Range Image output

The MVP output channels are fixed to:

- `range`
- `intensity`
- `valid_mask`

Pixel conflict policy is also fixed:

- when multiple points map to one pixel, keep the **nearest range**

Invalid pixels are normalized as:

- `range = +inf`
- `intensity = 0`
- `valid_mask = 0`

## GPU-only implementation choice

The first backend is CUDA-only for two reasons:

1. the project goal is a practical replacement path for DriveWorks-class Lidar
   processing rather than a low-performance reference port
2. forcing the first implementation through GPU-safe memory contracts avoids a
   future breaking redesign when additional kernels are added

This does **not** mean every future helper must be CUDA code, but it does mean
the production Range Image path should not depend on a CPU fallback.

## Current implementation status

Implemented today:

- `Buffer`
- `BufferPool`
- `PointCloudView`
- `PointCloudBuffer`
- `runtime/cuda_stream.h`
- `RigidTransform3f`
- spherical GPU `RangeImageBuilder`
- stream-aware `RangeImageBuilder::BuildAsync`
- `DecimationFilter`
- `CropBoxFilter` and `CropBoxFilterWorkspace`
- regression tests for layout validation, owned point-cloud storage, and
  range-image projection/filter semantics

Not implemented yet, but expected in the base library roadmap:

- cylindrical Range Image
- voxel/outlier filter modules
- registration, segmentation, and fusion layers

## Validation commands

Inside a CUDA-enabled environment:

```bash
bazel test --config=gpu --@rules_cuda//cuda:archs=compute_89:sm_89 //vision/lidar:all
```

Or via the repository container entrypoint:

```bash
docker/scripts/whl.sh 'bazel test //vision/lidar:all --config=gpu --@rules_cuda//cuda:archs=compute_89:sm_89'
```

## Next parity steps after the initial Range Image milestone

- add reusable workspaces that can be shared by range-image and future filters
- add ring and timestamp aware projection policies
- add configurable occlusion tie-breakers
- add voxel and outlier filters
- add accumulation and multi-lidar stitching interfaces
- add CUDA performance and memory regression tests

## Apollo-lite replacement relevance

The current filter layer is intentionally chosen to cover the first parts of an
Apollo-lite replacement path:

- `CropBoxFilter` can express finite-value rejection, ego-box removal, and
  height-window style preprocessing
- `DecimationFilter` provides the simplest deterministic downsample stage
- `RangeImageBuilder::BuildAsync` establishes the stream-aware execution pattern
  later filter and export modules should follow

The remaining high-value gaps for Apollo-lite replacement are voxel filtering,
feature-array export, and fusion/accumulation modules.
