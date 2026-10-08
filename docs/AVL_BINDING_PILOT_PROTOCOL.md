# AVL two-subject binding pilot1
Declared2026-10-08 before model execution. Implements the preceding candidate
binding review as a bounded pilot, with explicit changes below.

## Fixed data
16 training identifiers obj00..obj15;8 test identifiers new00..new07.
Sort all120 unordered training pairs by SHA256 of
'avl-binding-v1:' + first + ':' + second. First30 reserved, remaining90 train.
Relations left/right/above/below; inverse indices0<->1 and2<->3.
Enumerate4 physical scenes,2 equivalent grammatical orientations,2 table orders.
Templates:
0 '{A} is {relation} {B}.'
1 '{A} is {relation} {B} now.'
2 'Now {A} is {relation} {B}.'
3 '{A} now is {relation} {B}.'
left/right phrases include 'of'; above/below do not.
Counts: train2880, validation1440, pair960, joint480, new_names448, phrasing1440.
Keep inverse descriptions in their physical scene group and unordered pair split.
Metadata-only targets are exactly balanced within each table context.

Lexical adapter maps the two literal names to slot0/slot1 according to the
explicit name table. It MUST NOT parse relation words, reorder grammar or
compute a target. Other tokens come from training-only vocabulary.
Because this erases identity, pair/new-name inputs can duplicate training
canonical token sequences. Report those overlaps; these partitions test adapter
coverage, not novel semantic composition. Heldout template3 is a surface test.

## Fixed model and training
Fresh embedding16 -> GRU48 -> linear16 tanh sender; receiver16->32 GELU->4.
End-to-end class-supervised learning, not unsupervised language emergence.
CPU PyTorch2.6.0;2threads; seeds44/55/66. AdamW lr0.003 weight_decay0.0001,
2000steps, batch128, gradient clip1. No early stopping or validation selection.
600second hard total training/audit budget; output must be fresh.
Train a separate no-message receiver (same16->32->4 architecture, constant zero
input) from scratch for2000steps and identical label batches, each seed.
This is a capacity-matched constant-input control, not a full expressive
identity-metadata model. Exact per-table class balance proves25% table-only
Bayes accuracy in this synthetic design.
No discrete-channel comparison in pilot1; defer until this baseline is measured.

## Wire and inference
Inference transmits actual per-example AVL1 float32 packet (76bytes), plus
a uint16-length-prefixed two-name table. No labels/source text to receiver.
Each packet decoded separately with existing codec before receiver inference.
Receiver predicts4 scene classes relative to entity-table order.
Header/model identity remain experimental; same-checkpoint required.
Report source UTF8 and all packet/table bytes. Structured oracle comparison is
an explicit one-byte class + same table; text transport is source bytes + table.
These are cost/fidelity references with gold parsing, not learned competitors.

## Evaluation and fixed gates
On every test partition (pair,joint,new_names,phrasing), each seed must achieve
accuracy>=0.95 AND macro_recall>=0.95. All classes support reported.
Gate equivalence, scene-reversal and coherent table-order change correctness
jointly at>=0.95 per test partition. Pairs count only if BOTH outputs correct.
Replace each received message with valid message for opposite scene of SAME
pair/table/template/orientation. Report original accuracy and replacement-scene
accuracy. Require replacement agreement>=0.95. These tests establish bounded
semantic response, not a formal universal causal identification.
Also report zero/shuffled accuracy as out-of-distribution diagnostics, not proof.
No-message baseline must be<=0.30, transmitted gap>=0.60.
Run all9 sender/receiver cross-play pairs descriptively; no alignment gate.
Do not tune pilot1 after observing final data. Any failure remains recorded;
architecture/curriculum changes require a separately declared revision.
Save dataset/source hashes, weights hashes, metrics, costs and raw artifacts.
