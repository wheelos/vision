# Context Maintenance Practices

This file describes how to keep `docs/context` useful as the project grows.

## Principles

### 1. Keep context close to the code

If a change introduces a new invariant, compatibility rule, parameter ordering,
or architectural trade-off, document it in the same change as the code.

### 2. Prefer short, actionable context over long narrative

Good context answers questions future contributors actually have:

- What is public?
- Which external references define the behavior?
- What assumptions must not be broken?
- What remains unfinished?

### 3. Separate documentation by purpose

Following a Divio-style split:

- **Reference**: exact API facts, coefficient order, file map, commands
- **Explanation**: why a model or convention was chosen
- **How-to**: concrete usage steps
- **Tutorial**: end-to-end examples for new users

`docs/context` should primarily hold **reference** and **explanation**. If the
project later adds user onboarding examples, those can live under `docs/`
separately as how-to or tutorial material.

### 4. Record significant decisions explicitly

Use ADRs for decisions that would otherwise be rediscovered repeatedly. Good
ADR candidates include:

- replacing one camera model with another
- changing public targets or include paths
- altering distortion conventions
- adding or removing external runtime dependencies

## Suggested workflow after each feature

1. Update or add the relevant context file.
2. Update TODOs if the feature creates follow-up work.
3. If the decision is architecturally significant, add an ADR.
4. Link the exact validation command that was used for the changed area.

## Suggested future structure

As the project grows, this directory can evolve toward:

- `docs/context/<module>.md` for module-level explanation/reference
- `docs/context/adr/NNN-title.md` for decision history
- `docs/how-to/` for consumer-focused instructions
- `docs/reference/` for generated or hand-maintained API detail

## Anti-patterns to avoid

- dumping raw meeting notes without decisions or action items
- copying large external documentation instead of linking to it
- mixing unstable TODOs with stable reference material in the same file
- leaving stale assumptions after behavior changes
