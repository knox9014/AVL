# Three-entity AVL composition transport — 2026-10-08

An experimental AVC1 envelope now carries two frozen learned relation vectors
and explicit binding topology for three named entities. All three matched
sender/receiver checkpoints pass the preregistered gates on each final partition.
This is a working designed composition interface, not a model that learned
composition from withheld combinations.

## Evidence

[Protocol](../AVL_COMPOSITION_PILOT_PROTOCOL.md) declared before implementation
at e60c006e07f15f72828e92aec6272159eef9e001.
[CI run37733118560](https://github.com/knox9014/AVL/actions/runs/37733118560)
tested head ca3871a3be52801d8de53b9c92718f6106f57227:
6 focused tests and136 full tests pass; fixed primitive training reproduction and
the20.070-second CPU2thread composition audit complete successfully.
Existing query, binding and alignment workflows also pass this source head.
Initial test-first run37732962269 failed on the absent composition implementation;
reproduction and audit did not run. No final-test tuning followed.

The old11,124-parameter primitive model is reproduced separately for each of
seeds44/55/66. Neural weights remain frozen during this audit; tensor-state and
checkpoint hashes are checked before/after. No new neural parameters or adapter
training are added. Source/data configuration matches the old binding record,
but all three checkpoint file hashes differ from the previous run; old tensor
identity is not established.

## Reconstruction and interventions

| Partition | Rows per seed | Supported relation tuples | Exact two-edge / observed-tuple macro recall |
| --- | --- | --- | --- |
| development | 1152 | 12 | 100% / 100% |
| joint | 384 | 4 withheld | 100% / 100% |
| phrasing | 1152 | 12 | 100% / 100% |
| new_names | 1536 | all16 | 100% / 100% |

Every supported tuple has96 source examples. Final partitions total3072 base
messages per seed,9216 across three seeds. Zero-support classes remain explicit
in the report and are excluded from observed-class macro recall.

For every seed and partition, inverse descriptions, coherent name-table
permutations and reversed clause order have100% paired accuracy.
Replacing the first relation by its inverse gives100% new-meaning accuracy.
Swapping vector rows with topology fixed gives100% expected changed-meaning
accuracy and0% original agreement on eligible rows:384 joint,768 phrasing and
1152 new-name rows per seed (768 development). Eligibility means the two
canonical edge labels differ.

Zero-vector exact accuracy is0% joint,8.33% phrasing and6.25% new_names.
Cyclic row-shuffling gives18.75%,16.67%,16.67% respectively.
Wrong-checkpoint controls use three cyclic pairings, not all six ordered pairs:
44->55 and55->66 score0%/8.33%/6.25% across those partitions;66->44 scores
8.33%/15.97%/14.06%. Matched checkpoints remain necessary.
The constant modal tuple from development is a descriptive no-message control.

## Meaning and scope

A source such as "obj00 is left of obj01. And obj01 is above obj02." is split
at an explicit delimiter. Lexical name matching supplies each edge's endpoints;
the learned primitive sender supplies its relation vector. The packet transmits
the name table and endpoints, so binding is not learned from vectors alone.
No clause parser supplies relation labels to the neural receiver.

Withheld joint tuples are(0,2),(2,0),(1,3),(3,1), using0:left,1:right,2:above,3:below.
These combinations are absent from the multi-clause development fixture.
No model trains on that fixture: all primitive relations were already supervised.
The result therefore establishes preservation of designed combinations, not
end-to-end learned compositional generalization or new semantic concepts.

A separate hand-written A->C rule is transitive only for two identical
directions; other cases are undetermined because orthogonal coordinates remain
unconstrained. Derived query accuracy is100%, but every joint query is the
undetermined class (support[0,0,0,0,384]); it cannot establish novel transitive
reasoning. Phrasing supports[96,96,96,96,768], and new_names supports
[96,96,96,96,1152]. This rule is not neural reasoning.

Only three nodes, two distinct connected edges, the fixed four-relation grammar
and existing primitive forms are supported. No general graph, conjunction
learning, cross-model alignment or AI network is implemented here.

## Packet and costs

AVC1 has a6byte header, three length-prefixed ASCII names,4byte endpoint topology
and one140byte AVL1 packet holding two vectors. The five-character fixture names
make the full envelope171bytes. Strict checks reject wrong headers/counts,
duplicate/self edges, invalid names, truncation/trailing bytes and nonfinite
vectors. Default AVL1 inference is unchanged.

| Final partition | Actual transmitted | Source UTF8 + same table | Symbolic gold oracle |
| --- | --- | --- | --- |
| joint | 65664 | 30144 | 12672 |
| phrasing | 196992 | 90432 | 38016 |
| new_names | 262656 | 120576 | 50688 |

Actual transport is2.1783x text-plus-table and5.1818x the33byte explicit symbolic
oracle with the same envelope, names and endpoints plus two one-byte relations.
That oracle is a gold representation comparison, not a trained text parser;
100% oracle accuracy is by construction. No learned advantage or wire saving
is claimed. Model identity negotiation/distribution and network framing are
excluded from costs and remain unimplemented.

## Reproduction and records

With the repository's pinned CPU runtime, use fresh output directories:

```sh
python -m antlab.semantic_binding_pilot --output binding-reproduction
python -m antlab.semantic_composition_pilot --study binding-reproduction --output composition-output
```

[Complete durable report](../../antlab/runs/composition-pilot-report-20261008.json)
retains all metrics/confusions, support, gates, source/data/model hashes and run
provenance. JSON is compact to retain the full report without a large pretty
printed file. Dataset SHA256 is
c1cb65a594d71b709531861c72402a625af6ab4a024eaa38688ca2ce1be7f59a.
Full synthetic datasets, weights and reports are also in30-day artifact11530812704;
ZIP SHA25640df6033fb10f25b952f38158285ca9d5759a9a561d11b58a47e8417ec642287.
Author review checked topology leakage, sparse query support, frozen tensors,
real byte round-trips, intervention eligibility and claim limits.

Next research should compare a learned composition receiver against this
designed scaffold and a symbolic baseline, with a preregistered split that
includes supported directional queries among heldout structures. Do not expand
the network or claim AI-language completion from this finite audit.
