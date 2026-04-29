# DriveWorks Replacement Roadmap

This document tracks the architectural path for turning `vision/lidar/` into an
open source lidar base library that can replace the foundational lidar building
blocks commonly consumed from DriveWorks.

## Scope

The goal is not feature-for-feature branding parity. The goal is **production
capability parity** for the base lidar layer:

- point cloud representation and memory ownership
- rigid transforms and coordinate normalization
- preallocated workspaces and pools
- high-throughput GPU preprocessing
- reusable interfaces for registration, segmentation, and fusion

## Package boundaries

The repository is now intentionally split into separate domains:

```text
vision/camera/   # camera models and camera-only math
vision/lidar/    # lidar data, memory, preprocessing, and future fusion stack
```

This split matters because DriveWorks-style lidar processing optimizes for GPU
execution and explicit memory control in ways that are structurally different
from the camera package.

## Performance-first design rules

To be a practical replacement rather than a demo implementation, the lidar base
library should follow these rules:

1. **GPU-first public API**
   - no mandatory CPU fallback in production paths
   - kernel-facing memory types are explicit in interfaces
2. **Zero-copy friendly data flow**
   - prefer unified/device-accessible memory
   - avoid hidden host-device transfers
3. **SoA-friendly access patterns**
   - planar field layout should be first-class because it maps naturally to
     coalesced GPU loads
   - interleaved input remains supported for compatibility with upstream
     sensors and middleware
4. **Explicit workspaces**
   - large temporary allocations must be caller-owned or pool-owned
   - algorithms should accept reusable workspace objects
5. **Composable module boundaries**
   - filters, registration, segmentation, and fusion layers should share the
     same `PointCloudView`/buffer abstractions instead of inventing local types
6. **Stream-aware execution**
   - future kernels should be able to run on user-provided CUDA streams and
     avoid device-wide synchronization in hot paths

## Current status

Already implemented:

- separate `vision/lidar/` package
- `PointCloudView` with planar/interleaved metadata
- `PointCloudBuffer` with owned planar/interleaved allocation
- `Buffer` and `BufferPool`
- `runtime/cuda_stream.h`
- `RigidTransform3f`
- GPU-native spherical Range Image generation
- stream-aware `RangeImageBuilder::BuildAsync`
- GPU `DecimationFilter`
- GPU `CropBoxFilter`
- containerized GPU build entrypoint

Still required for serious DriveWorks-class replacement:

- voxel grid and outlier filters
- radius/statistical outlier filters
- registration interfaces and GPU ICP implementation
- ground detection and clustering modules
- accumulation, ego-motion compensation, and multi-lidar stitching
- performance regression tests and throughput benchmarks

## Foundation roadmap

### Phase 1: Base library completion

Focus on the reusable substrate every later module depends on:

- field accessor helpers for host/device code
- configurable workspace objects
- broader stream-aware kernel launch surfaces
- deterministic buffer-pool reuse rules
- ring/timestamp support in the common point schema

### Phase 2: Preprocessing parity

- spherical and cylindrical Range Image builders
- voxel grid downsampling
- radius outlier removal
- statistical outlier removal

### Apollo-lite replacement note

For Apollo-lite specifically, the current `CropBoxFilter` and
`DecimationFilter` are the first reusable substitutes for the mixed host-loop /
PCL-centric preprocessing currently spread across
`modules/perception/lidar/lib/pointcloud_preprocessor` and
`pointcloud_detection_preprocessor`.

### Phase 3: Transformation and feature extraction

- ICP interfaces and CUDA kernels
- ground plane extraction
- Euclidean clustering
- DBSCAN-style clustering

### Phase 4: Accumulation and fusion

- ego-motion compensated accumulation
- multi-frame map stitching
- multi-lidar stitching into a shared frame

### Phase 5: Production hardening

- CUDA architecture matrix coverage
- throughput/latency baselines
- memory pressure regression tests
- external-consumer integration tests

## Acceptance bar for “can replace DriveWorks”

The library should not claim replacement-readiness until it can satisfy all of
the following in the target deployment environment:

1. ingest common lidar layouts without extra copies
2. keep hot preprocessing on GPU
3. reuse preallocated buffers/workspaces across frames
4. expose stable, documented public APIs and package boundaries
5. provide regression and performance coverage for supported GPU targets
