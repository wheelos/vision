# vision
Low-level vision library functions of the camera

# Quick Start

## Install

```
sudo bash scripts/deps/install_bazel.sh
```

## Build

```
bazel build //vision:all
```

## Test

```
bazel test //vision:all
```

## Public C++ API

Depend on `@vision//vision:camera_models` and include:

```cpp
#include "vision/camera_models.h"
```

The module provides self-implemented projection and back-projection models:

- `PinholeCameraModel`
- `BrownCameraModel` with OpenCV-compatible `(k1, k2, p1, p2, k3)` ordering
- `OmnidirectionalCameraModel` with OpenCV `omnidir`-compatible `xi` and
  `(k1, k2, p1, p2)` ordering

## Context Docs

Project context, references, and engineering TODOs live under:

```text
docs/context/
```
