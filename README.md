# vision
Low-level camera and lidar foundation libraries

# Quick Start

## Install

```
sudo bash scripts/deps/install_bazel.sh
```

## Build

```
bazel build //vision/camera:all
```

## Test

```
bazel test //vision/camera:all
```

## GPU Lidar Build

For the GPU-native lidar package:

```bash
bazel test --config=gpu --@rules_cuda//cuda:archs=compute_89:sm_89 //vision/lidar:all
```

Or build in the provided CUDA container:

```bash
docker/scripts/whl.sh 'bazel test //vision/lidar:all --config=gpu --@rules_cuda//cuda:archs=compute_89:sm_89'
```

## Public C++ API

### Camera

Canonical dependency and include:

```cpp
#include "vision/camera/camera.h"
```

```text
@vision//vision/camera:camera
```

The camera module provides self-implemented projection and back-projection
models:

- `PinholeCameraModel`
- `BrownCameraModel` with OpenCV-compatible `(k1, k2, p1, p2, k3)` ordering
- `OmnidirectionalCameraModel` with OpenCV `omnidir`-compatible `xi` and
  `(k1, k2, p1, p2)` ordering

### Lidar

Canonical dependency and include:

```cpp
#include "vision/lidar/lidar.h"
```

```text
@vision//vision/lidar:lidar
```

The first lidar release exposes:

- point cloud data interfaces with planar/interleaved layout support
- unified/device buffer ownership and fixed-size buffer pooling
- stream-aware GPU-native spherical Range Image generation
- GPU `DecimationFilter`
- GPU `CropBoxFilter` for finite-value, box, and height-window style filtering
- a path toward DriveWorks-class lidar base-library parity documented under
  `docs/context/`

## Context Docs

Project context, references, and engineering TODOs live under:

```text
docs/context/
```
