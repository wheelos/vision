# Apollo-lite Replacement Mapping

This document records how `vision/lidar/` is being shaped so it can eventually
replace the current lidar preprocessing substrate used in
`apollo-lite/modules/perception/lidar/lib/` while also staying aligned with the
broader DriveWorks-replacement goal.

## Why this matters

Apollo-lite already contains useful lidar functionality, but its current
preprocessing path mixes several concerns into a small number of modules:

- point filtering in host-side loops
- PCL-based voxel downsampling
- frame fusion logic
- point shuffling
- point-cloud-to-array export for downstream models

That works as an application stack, but it is not the shape we want for a
reusable, high-throughput base library. A DriveWorks-class replacement should
separate these concerns into reusable GPU-native modules with explicit memory
ownership and stable interfaces.

## Current Apollo-lite responsibilities and target mapping

| Apollo-lite responsibility | Current or planned `vision/lidar` module |
| --- | --- |
| finite-point filtering | `filters/CropBoxFilter` with `reject_non_finite=true` |
| nearby ego-box removal | `filters/CropBoxFilter` with `mode=kRemoveInside` |
| height-window filtering | `filters/CropBoxFilter` with a configured `z` interval |
| simple deterministic downsample | `filters/DecimationFilter` |
| voxel grid downsample | planned `filters/VoxelGridFilter` |
| frame fusion | planned `fusion/` accumulation and stitching modules |
| cloud-to-array export | planned tensor/feature export surface |
| stage-local temporary storage | explicit workspaces and `BufferPool` reuse |

## Design differences that should make this library stronger

To outperform the current Apollo-lite substrate, the base library is following
these rules:

1. **GPU-first interfaces**
   - no PCL dependency in hot preprocessing paths
   - kernels operate on `PointCloudView` / `PointCloudBuffer`
2. **Explicit memory ownership**
   - algorithms do not hide large internal allocations
   - reusable workspaces are exposed to callers
3. **Stable data contracts**
   - all preprocessing stages share the same point-cloud substrate
   - later fusion/export modules do not need bespoke cloud types
4. **Stream-aware execution**
   - hot kernels can run on caller-provided CUDA streams
   - library APIs should avoid device-wide synchronization unless output sizing
     or compatibility forces it
5. **Layout-aware performance**
   - planar layout remains first-class for coalesced GPU reads
   - interleaved input is supported for middleware compatibility

## What is already in place

The current repository now has the first replacement-oriented building blocks:

- `PointCloudView` and `PointCloudBuffer`
- `Buffer` and `BufferPool`
- `runtime/cuda_stream.h`
- `RangeImageBuilder::BuildAsync`
- `filters/DecimationFilter`
- `filters/CropBoxFilter`
- `filters/CropBoxFilterWorkspace`

That means the library can already express the same broad category of work as
Apollo-lite's early preprocessing stages, but with more reusable module
boundaries and better GPU affinity.

## Highest-value remaining gaps for Apollo-lite replacement

1. `VoxelGridFilter` to remove the remaining PCL-centric dependency pattern
2. feature-array export for downstream model ingestion
3. frame-fusion and ego-motion-aware accumulation
4. ring/timestamp-aware policies for sensor-specific preprocessing
5. throughput benchmarks against representative Apollo-lite workloads
