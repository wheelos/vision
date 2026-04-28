# Engineering TODO

This is the working backlog for turning the current camera model module and the
wider project into a more production-ready package.

## High priority

1. **Add CI gates**
   - Run `bazel build //vision:all` and `bazel test //vision:all` in a clean
     environment on every change.
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
