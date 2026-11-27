# Step docs

Every build step gets a folder here, named after its milestone and step, for example `m1-6c-ingest/`. It holds three short files, written **in this order, before any code**:

| File | Who writes it | What it holds |
|---|---|---|
| `intent.md` | The owner, in their own words | Problem, outcome, users, constraints, open questions. 5 to 10 lines. |
| `spec.md` | Owner and Claude together, from the intent | What the step does and doesn't do, inputs and outputs, edge cases, and **done when** (checks anyone can run). |
| `plan.md` | Claude drafts, the owner reviews | Files to create or change, in order, with the tests for each. |

Rules:
- The spec is agreed before the plan, and the plan before the code. Each one is committed.
- If the code has to differ from the plan, update `plan.md` in the same commit.
- If a decision changes what the step is for, update `spec.md` and say why.
- At the end of the step, add a short **Result** section to `plan.md`: what was built, what the second check on the real data showed, and anything left for later.

## Templates

### intent.md

```markdown
# Intent: <milestone> <step>, <name>

Problem:
Outcome:
Users:
Constraints:
Open questions:
```

### spec.md

```markdown
# Spec: <milestone> <step>, <name>

## Goal
One paragraph: what this step does and why.

## In scope / out of scope

## Inputs and outputs
Files, formats, commands.

## Behaviour
Normal path, then each edge case and what happens.

## Done when
- [ ] Checks anyone can run (commands, counts, expected results), including a run on the full real data.
- [ ] The four CI checks pass.
```

### plan.md

```markdown
# Plan: <milestone> <step>, <name>

| # | Change | Files | Tests |
|---|---|---|---|
| 1 | | | |

## Result
(filled in at the end of the step)
