# Engineering TODO

This is the working backlog for turning the current camera model module and the
wider project into a more production-ready package.

## High priority

1. **Add CI gates**
   - Run `bazel test //vision/camera:all` and the CUDA lidar test entrypoints in
     a clean environment on every change.
   - Add formatting and static analysis checks as mandatory quality gates.

2. **Broaden numerical validation**
   - Add randomized round-trip tests for pinhole, Brown, and omnidirectional
     models.
   - Add boundary tests for extreme intrinsics, large distortion, and near-FOV
     limits.
   - Add explicit tests for non-finite inputs.

3. **Define public compatibility policy**
   - Clarify which headers and Bazel targets are public.
   - Define how semantic/API-breaking changes are recorded and communicated.
   - Add versioning expectations for bzlmod consumers.

4. **Complete the lidar base substrate**
   - Keep expanding stream-aware GPU APIs beyond Range Image so hot paths stay
     free of device-wide synchronization.
   - Add reusable workspaces for future filters and registration kernels.
   - Extend the common schema with first-class ring/timestamp support.

5. **Deliver preprocessing parity**
   - Add voxel grid on top of the shared point cloud abstractions.
   - Add radius and statistical outlier removal with explicit workspace reuse.
   - Add cylindrical Range Image beside the current spherical builder.
   - Add GPU feature-array export to replace Apollo-lite style cloud-to-array
     staging.

## Medium priority

1. **Add external consumption tests**
   - Create a small example workspace that imports `@vision` through bzlmod.
   - Build and test that example as part of validation.

2. **Measure performance**
   - Add reproducible microbenchmarks for projection and back-projection.
   - Track latency and allocation regressions.

3. **Strengthen diagnostics**
   - Document expected error behavior for invalid intrinsics and invalid rays.
   - Add debug-friendly utilities or logging hooks if future integration needs
     them.

4. **Harden the GPU lidar module**
   - Add a CUDA toolchain matrix for at least one datacenter GPU and one
     workstation GPU architecture.
   - Add regression coverage for unified-memory pressure and device-only output
     buffers.
   - Extend Range Image inputs to cover ring and timestamp driven layouts.

5. **Start DriveWorks-class feature layers**
    - Define GPU ICP interfaces and convergence/termination policies.
    - Add ground-plane extraction and obstacle clustering interfaces.
    - Add accumulation, ego-motion compensation, and multi-lidar stitching APIs.

6. **Close the Apollo-lite replacement gap**
   - Add `VoxelGridFilter` so downsampling no longer depends on PCL-style host
     components.
   - Add fusion/export layers that map cleanly onto Apollo-lite
     `pointcloud_detection_preprocessor` responsibilities.
   - Benchmark the new GPU path against representative Apollo-lite workloads.

## Context maturity TODO

1. Add ADRs for:
   - choosing OpenCV-compatible parameter conventions
   - choosing self-implemented math over an OpenCV runtime dependency
   - public header and bzlmod export policy

2. Add how-to docs for:
   - initializing each camera model
   - selecting the right model for a sensor
   - consuming the library from another Bazel module

3. Add reference docs for:
   - coefficient ordering
   - valid input ranges and failure conditions
   - test coverage boundaries and known limitations
