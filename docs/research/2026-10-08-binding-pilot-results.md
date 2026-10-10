# AVL spatial binding pilot1: results
Date2026-10-08. [Protocol](../AVL_BINDING_PILOT_PROTOCOL.md) was committed before
training at c47dfc1a3e8fa40bc5d3009957b804a6365cd4e7.
This is a new, class-supervised research model; the existing nine-field/default
AVL interface and its checkpoints are unchanged.

## Outcome
All three seeds passed the fixed within-checkpoint binding gates. Independent
checkpoint cross-play failed to establish a common language.

| Evaluation | Seed44 | Seed55 | Seed66 |
|---|---:|---:|---:|
| Reserved pairs, familiar surface |100%|100%|100%|
| Reserved pairs and new surface |100%|100%|100%|
| New names and new surface |100%|100%|100%|
| Familiar pairs, new surface |100%|100%|100%|
| Separately trained constant-input receiver |25%|25%|25%|

Accuracy and four-class macro recall are100% in every transmitted test partition.
The test partitions have960/480/448/1440 sources respectively, with equal class
support. Each model also scores100% when requiring BOTH original and transformed
examples correct for inverse paraphrases, opposite scenes and coherent table
reordering. Replacing a message with the valid opposite-scene message yields
100% agreement with the replacement scene and0% with the original scene.

Zero vectors produce25%; cyclic one-position vector shuffling produces6.25%
on the familiar-template pair partition and12.5% on the new-template partitions.
These shuffles are order-dependent corruption diagnostics, not random baselines
or standalone causal proof. The separately trained no-message head uses identical
label batches and the same receiver architecture, with constant zeros.

## Independently trained codebooks

Rows are senders, columns receivers; evaluated on the480-source joint partition.

| Sender / Receiver |44|55|66|
|---|---:|---:|---:|
|44|100%|25%|25%|
|55|0%|100%|25%|
|66|37.5%|0%|100%|

The six off-diagonal pairs average18.75%, below the25% four-class chance level.
This describes these six pairs, not a statistically established population
difference from chance. No alignment adapter was trained and no universal
interoperability is demonstrated. Same-checkpoint inference remains necessary.

## What this actually establishes

The11,124-parameter model learns a bounded4-relation scene classifier through a
16-dimensional continuous channel. Its receiver has676 parameters; the separately
trained constant-input baseline also has676. Each seed used2,000 fixed steps,
batch128, CPU2threads. Full training/evaluation took27.485seconds in this CI run,
excluding dependency installation and regression tests.

The literal adapter replaces arbitrary two-name strings with slot0/slot1.
It does not parse relations or labels, but it provides the name binding scaffold.
The receiver's neural head sees only deserialized vectors and predicts relative
to that table; literal names are not learned concepts.

There are only16 UNIQUE canonical training strings despite2,880 train sources.
All960 reserved-pair familiar-surface sources have canonical training overlap.
The template3 test partitions have no exact training-string overlap, but share
only eight canonical forms with one another. Their many names/pairs are therefore
not independent new semantic tasks. Development uses a separately reserved
surface without selecting training steps/checkpoints.

This pilot demonstrates limited word-order/role binding and response to valid
meaning changes. It does not establish general compositional language, learned
name grounding, action execution in an external world, cross-model alignment,
or superiority to text/structured representations. A lookup over finite
canonical strings remains an alternative explanation for much of the behavior.

## Actual byte costs

Each example uses one76-byte AVL1 packet plus14bytes for the two5-character
identifiers and their uint16 length prefixes:90bytes total. Vectors are actually
serialized and deserialized before receiver inference. Literal-table byte
buffers are built for accounting; no network service was deployed.
Training uses the differentiable pre-serialization vector; exact float32 wire
fidelity is tested separately because byte packing intentionally detaches it.

| Partition | Vector plus table | Text plus same table | One-byte oracle plus table |
|---|---:|---:|---:|
| Reserved pair |86400|36720|14400|
| Joint pair/surface |43200|19320|7200|
| New names/surface |40320|18032|6720|
| Familiar pair/new surface |129600|57960|21600|

The joint vector format costs2.24 times the text reference and6 times the
one-byte structured oracle. The oracle carries gold labels and is a cost/fidelity
reference, not a trained competing system; one byte is a convenient encoding,
not an optimal information-theoretic bound. No communication efficiency gain
has been demonstrated. Network overhead and model-distribution costs excluded.

## Verification and provenance

[Successful binding run37720958084](https://github.com/knox9014/AVL/actions/runs/37720958084)
at tested head17bbc80fa215cdc6cd7604cb431d8e11be4efbbe:
8 focused tests and128 repository tests passed on Python3.12/PyTorch2.6.0CPU.
[Existing query workflow37720958028](https://github.com/knox9014/AVL/actions/runs/37720958028)
also passed its original studies and Python3.10/3.12 fixtures.

An additional local, independent surface-label check parsed every7,648 synthetic
source and matched its table-relative target. Local dataset tests passed5/5.
The publication pattern scanner inspected six newly staged code/test/protocol/
workflow files with no findings or binary exclusions; this is not proof of
absence of every possible private detail. Synthetic outputs and result records
were separately reviewed.

Test development first observed missing-module failures. The first implemented
run stopped at a mistaken test expecting gradients through byte serialization;
the test was corrected to verify differentiable training and byte inference
separately. NO model training ran in either failed workflow, and no model/data/
optimizer/gate was changed after observing final results.

[Durable summary](../../antlab/runs/binding-pilot-summary-20261008.json) includes
full class confusion/support/recall, source/data/checkpoint hashes, gates, controls
and cross-play results.
[Artifact11526091591](https://github.com/knox9014/AVL/actions/runs/37720958084/artifacts/11526091591)
contains the complete synthetic dataset, reports and model/baseline weights;
retention30days. Zip SHA256:
a38eff9ea05ed8ebb01fde06f7dfbc9245a8f394a50085f729b2a8082ae29e95.
Checkpoint metadata is limited to format, seed, synthetic vocabulary and tensor
state dictionaries. Use weights_only=True for future loading.

Reproduce with a fresh output directory:
```sh
python -m unittest discover -s antlab/tests
python -m antlab.semantic_binding_pilot --output binding-pilot-output
```

## Review and next decision
Author self-review checked class balance, semantic partitions, lexical adapter
access, byte receiver fidelity, paired evaluation, controls and source hashes.
No independent reviewer was used; this remains a draft research PR.

Next prioritize a declared alignment/sample-efficiency study: freeze one
receiver, fit a small vector translator on explicitly counted calibration
examples, compare heldout surfaces and wrong/shuffled calibration controls, and
report labels/parameters/bytes. This would be supervised compatibility,
not independently emerging universal semantics.

For broader language, separately introduce three entities and multiple relations
with genuinely reserved canonical structures. Do not count renamed copies of
the same eight/16 slot strings as compositional generalization.
