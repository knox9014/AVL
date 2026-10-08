# AVL binding-space calibration pilot1
Declared2026-10-08 before execution. No test-dependent selection or tuning.

Reuse the binding pilot1 architecture/data and reproduce its fixed seeds44/55/66
unchanged in CI. Freeze and safe-load its weights. Verify checkpoint hashes
against the reproduced report and source/data hashes against the prior durable
binding summary. Reproduction is explicitly reported, not silent new training.

For each of six ordered mismatched sender/receiver pairs, use4/8/16 UNIQUE
canonical TRAIN strings. Three fixed calibration-selection seeds101/202/303.
Select equally per target class, SHA256-order canonical strings by seed within
class, take1/2/4 examples each, then interleave classes0,1,2,3. Nested selections.
One source row per canonical string; no duplicate-name inflation.
Thus54 fitted adapters. Final test templates3 never enter calibration.
Record selected strings/labels/indices and dataset hashes.

Fit an affine16->16 ridge bridge on paired sender/receiver vectors for the
SAME calibration messages. Solve float64 normal equations with lambda0.01 on
weights and no bias regularization.17x16=272 parameters. No optimizer or tuning.
Convert mapped vectors to float32, serialize/deserialize existing AVL1 bytes,
then invoke the frozen receiver. Sender/receiver weights never change.
Bridge needs paired target-model representations; this is supervised calibration,
not zero-shot language discovery, despite fitting without explicit class targets.

Controls: unadapted cross-play; wrong-pair bridge trained on target vectors
cyclically shifted by one row (interleaved classes guarantee WRONG class);
fresh affine16->4 ridge classifier on sender vectors and explicit one-hot class
labels,68parameters; wrong-label classifier using same cyclic shift.
Fresh-classifier supervision and representation-pair supervision differ, so
report both without claiming identical learning tasks.

Evaluate all four existing test partitions; accuracy, macro recall, all-class
support/confusion. Same16/8 canonical-string limits remain. These are fresh
surface tests of already taught classes, not new semantic structures.
Gate for EACH calibration seed and EACH model pair at16 examples: every test
partition accuracy and macro recall>=0.95; wrong-pair accuracy<=0.30 and gap>=0.60.
Smaller calibration counts and fresh classifier are descriptive learning curves.
Do not change protocol after final inspection; failures remain failures.
No causal universality or model-independence claim.

Budget:CPU2threads,600seconds excluding unchanged upstream reproduction.
Fresh output folder; finite tensors; safe weights-only loading.
Save reports, adapter tensors, source/checkpoint/data hashes and selected examples.
Report bridge272 float32 parameters=1088 raw bytes; classifier68=272 raw bytes,
separate torch archive sizes. Per-message wire remains90bytes including table.
Calibration payload counts two76byte vector packets plus14byte shared name table
per example=166n bytes; source bytes/model access, computation, network and model
distribution excluded and stated. Label-classifier payload one76byte vector,
14byte table and one-byte target=91n. These are distinct assumed provisioning
schemes, not observed network benchmarks. Model/adapter identity negotiation
remains unimplemented.
