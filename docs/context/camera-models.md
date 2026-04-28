# Camera Models Context

## Scope

The `vision/` module currently provides self-implemented camera projection and
back-projection models intended for low-level vision use:

- `PinholeCameraModel`
- `BrownCameraModel`
- `OmnidirectionalCameraModel`
- umbrella header: `vision/camera_models.h`

The implementation is designed to be consumed from Bazel/bzlmod as:

```cpp
#include "vision/camera_models.h"
```

with:

```bzl
deps = ["@vision//vision:camera_models"]
```

## External references

These references shape the current implementation and should be treated as the
first source of truth when evolving projection behavior or parameter semantics.

| Source | Why it matters | Current alignment |
| --- | --- | --- |
| [NVIDIA DriveWorks camera model use case](https://docs.nvidia.com/drive/driveworks-4.0/cameramodel_usecase0.html) | Defines the expected workflow around `pixel2Ray` / `ray2Pixel` style usage. | The module exposes `PixelToRay` / `RayToPixel` with the same conceptual flow. |
| [OpenCV calib3d camera model docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) | Defines the standard pinhole and Brown-Conrady intrinsic/distortion conventions. | `PinholeCameraModel` and `BrownCameraModel` use OpenCV-style `K` and coefficient ordering. |
| [OpenCV omnidir / ccalib docs](https://docs.opencv.org/4.x/d3/ddc/group__ccalib.html) | Defines Mei omnidirectional model semantics and parameter ordering. | `OmnidirectionalCameraModel` uses `K + xi + D(k1,k2,p1,p2)` semantics. |

## Current design

### Base API

`vision/camera_model.h` is the common surface:

- validates initialization state
- exposes `RayToPixel` and `PixelToRay`
- stores image size and intrinsic matrix
- rejects unusable inputs explicitly instead of failing silently

### Model-specific behavior

#### 1. Pinhole

- OpenCV-style intrinsic matrix:
  `K = [fx, skew, cx; 0, fy, cy; 0, 0, 1]`
- `PixelToRay` returns a ray in normalized `z = 1` form
- supports non-zero skew

#### 2. Brown

- coefficient order:
  `(k1, k2, p1, p2, k3)`
- forward projection applies Brown-Conrady radial + tangential distortion
- inverse projection uses fixed-point iteration
- `GetIdealModel()` returns a distortion-free `PinholeCameraModel`

#### 3. Omnidirectional

- uses a self-implemented Mei-style omnidirectional model
- parameter set:
  - `K = [fx, skew, cx; 0, fy, cy; 0, 0, 1]`
  - `xi`
  - `D = (k1, k2, p1, p2)`
- inverse projection:
  1. remove pixel-space distortion iteratively
  2. solve the Mei model sphere intersection
  3. normalize to a unit ray
- `GetIdealModel()` returns a `PinholeCameraModel`

## Important invariants

These assumptions should stay stable unless an ADR explicitly changes them:

- Intrinsic matrices must be finite and keep the last row equal to `[0, 0, 1]`.
- `fx` and `fy` must be strictly positive.
- Invalid, singular, or non-finite inputs must return `false`.
- The library should remain **self-implemented** and must not depend on OpenCV
  at runtime for projection math.
- Public include paths should remain stable under `vision/` for bzlmod users.

## File map

- `vision/camera_model.h`
- `vision/pinhole_camera_model.h`
- `vision/pinhole_camera_model.cc`
- `vision/brown_camera_model.h`
- `vision/brown_camera_model.cc`
- `vision/omnidirectional_camera_model.h`
- `vision/omnidirectional_camera_model.cc`
- `vision/camera_models.h`
- `vision/camera_models_test.cc`
- `vision/BUILD.bazel`

## Validation command

Use the repository test entry point for this module:

```bash
bazel test //vision:all
```

## Open questions

The following are not blockers for the current implementation, but should be
reviewed before the project is called production-ready end-to-end:

- randomized numerical stress testing
- broader edge-case coverage at extreme FOV and distortion values
- performance characterization and regression thresholds
- external consumer tests from a separate Bazel workspace
