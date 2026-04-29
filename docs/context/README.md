# Project Context

This directory is the living project context for `vision`. It is intended to
capture the information that is usually lost in code-only changes: why an
implementation exists, which external references it follows, what assumptions
must stay true, and what should be improved next.

## Why this directory exists

The goal is to make future implementation work faster and safer by keeping the
most useful context close to the codebase:

- **Reference**: stable API shapes, parameter conventions, file ownership.
- **Explanation**: design rationale, trade-offs, and alignment with upstream
  references such as DriveWorks and OpenCV.
- **Roadmap**: actionable TODOs for engineering and productization work.
- **Decision memory**: a lightweight place for ADR-style records when a change
  is architecturally significant.

## Structure

- `camera.md`: canonical package boundary and compatibility policy for camera.
- `camera-models.md`: model-specific behavior and parameter conventions.
- `lidar.md`: current lidar architecture, public API, and GPU implementation
  status.
- `driveworks-replacement.md`: staged roadmap for DriveWorks-class lidar base
  library parity.
- `apollo-lite-replacement.md`: mapping from Apollo-lite lidar preprocessing
  responsibilities to the modular `vision/lidar` replacement plan.
- `todo.md`: near-term and medium-term backlog for production hardening.
- `context-practices.md`: how to keep this directory useful over time.
- `adr-template.md`: template for future Architecture Decision Records.

## Operating principles

This directory follows two widely used documentation practices:

1. **ADR / decision-log thinking**: record architecturally significant
   decisions and their rationale, not just the final code.
2. **Divio documentation split**: keep reference, explanation, how-to, and
   tutorials separate so readers can quickly find the right level of detail.

## Update checklist

When a meaningful feature or refactor lands, update this directory in the same
change whenever possible:

1. Update `camera.md`, `lidar.md`, or the relevant context file with new
   invariants, assumptions, public API changes, or validation commands.
2. Update `todo.md` if the change closes or creates meaningful follow-up work.
3. Add an ADR using `adr-template.md` if the change affects architecture,
   module boundaries, parameter conventions, or compatibility expectations.
4. Keep links to external references precise and stable.
