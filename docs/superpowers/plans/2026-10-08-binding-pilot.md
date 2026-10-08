# AVL Binding Pilot Implementation Plan

**Goal:** Execute the fixed two-subject learned communication pilot and preserve
failures and limits.
**Architecture:** Separate synthetic data generator, GRU sender/MLP receiver
runner using existing AVL1 codec, and regression tests. Existing inference stays
unchanged.
**Tech Stack:** Python3.12, PyTorch2.6.0CPU, unittest, GitHub Actions.
**Spec:** docs/AVL_BINDING_PILOT_PROTOCOL.md
**Execution:** Direct implementation in the existing AVL research branch under
the user's standing independent-development instruction.

## Global Constraints
Only AVL files; no ACT code or personal content. Freeze protocol before training.
600second total budget,2threads,3seeds, no retuning against test outcomes.
Record canonical overlap and explicit literal channel.
No default inference changes or main merge.

## Task1: Dataset and leakage tests
- [ ] Add antlab/tests/test_semantic_binding.py covering inverse equivalence,
group split isolation, balance, canonical overlap and malformed tables.
- [ ] Observe failure before implementation.
- [ ] Implement antlab/semantic_binding_data.py: make_data()->dict of row lists;
canonicalize(text,table)->list[str]; inverse/transform helpers.
- [ ] Verify tests and counts; target leakage absent in receiver metadata.

## Task2: Training and wire receiver
- [ ] Test actual codec roundtrip, finite inputs, correct intervention targets,
all-class metric support and wire totals.
- [ ] Implement antlab/semantic_binding_pilot.py with the fixed architecture,
training and constant-input baseline. Receive only deserialized vectors.
- [ ] Execute all seeds once; save checkpoints, source/data hashes, metrics and
cross-play matrix. No success-dependent training changes.

## Task3: CI and evidence
- [ ] Add isolated binding workflow; first demonstrate test failure for missing
implementation, then run complete regression and full pilot.
- [ ] Review code and reports; save durable summary and result analysis.
- [ ] Publish only explicitly listed AVL paths to existing draft PR.

## Review Focus
Inverse paraphrase leaks: semantic split audit.
Metadata relation leaks: per-context balance.
Copied names misread as generalization: canonical overlap counts.
Zero-vector distribution shift: valid counterfactual replacement and trained
no-message control.
False intervention success: require both paired predictions correct.
Missing classes: metrics reject absent support.
