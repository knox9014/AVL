# Shared AVL relation operator and four-node transfer — 2026-10-08

All preregistered research gates passed for all three operators. A shared
learned channel operation repairs the observed three-node failure cohort and
answers fresh four-node/three-edge chain queries at100% accuracy and observed
macro recall. This success relies on designed routing, known inverse channels
and the old supervised primitive decoder. It is not autonomous rule discovery
or a universal AI language.

## What changed

The historical raw50feature MLP reached63.89–70.31% on the observed joint cohort.
The new receiver first uses the frozen supervised primitive decoder to obtain
four directional probabilities. Explicit transmitted topology chooses the query
path; known inverse-channel permutation handles reverse traversal.

One2->16 GELU->1 channel network is applied identically to each of the four
relations. A learned unknown logit completes the five-class output. There are66
new trainable parameters and676 frozen decoder parameters,742 active receiver
parameters. A one-edge query calls g(p,p); two edges call g(p,q). Three edges
recursively reuse the same operator after softmax, retaining directional
probability mass without renormalizing away the unknown class. Learned
inference contains no hard relation-equality or absorbing-unknown branch.

Direct-query examples already supervise positive channel agreement for all
directions, and the two-edge data teach mixed cases. Binary operator patterns
are therefore supervised/observed. Sharing these computations is a strong
architectural prior. The comparison with the old MLP changes both structure and
access to pretrained class semantics, so this is not an equal-information,
capacity-matched ablation or proof that parameter sharing alone caused gains.

## Protocol and data

[Protocol](../AVL_SHARED_OPERATOR_PROTOCOL.md) was declared before implementation
at57d3039d79ec90682fe53d5ab6d4e9b61aa23be9.
Operator seeds244/255/266 use frozen primitive seeds44/55/66,3000steps,128-example
class-balanced batches, AdamW0.003/weight_decay0.0001/clip1, with no selection,
extra curriculum or post-test tuning. Separate no-vector operators train on
the same batches, zeroing vectors before the primitive decoder.

Three-node train is the prior12-tuple fixture with6912queries. The36 train and12
joint semantic signatures remain disjoint, but old joint results are already
observed and are openly reused as a regression comparison.

The fresh structural transfer cohorts have four nodes/three edges and all64
relation triples, two orientation masks, four of24 table permutations and two
synthetic name quadruples. Each cohort has1024scenes and12288directed queries.
Training contains no four-node scene or three-edge route. known4/new4 share
semantic structures under renaming and are not independent concept discoveries.
Only the finite four primitive relations are supported.

## Results

All values below hold separately for all three seeds.

| Cohort | Queries per seed | Accuracy | Observed-class macro recall |
| --- | --- | --- | --- |
| old joint | 2304 | 100% | 100% |
| old phrasing | 6912 | 100% | 100% |
| old new_joint | 2304 | 100% | 100% |
| fresh known4 | 12288 | 100% | 100% |
| fresh new4 | 12288 | 100% | 100% |

Four-node full support is[1824,1824,1824,1824,4992].
Of2048 distance-three queries, support is[32,32,32,32,1920]; all128 supported
positive distance-three queries score100%, as do undetermined cases.
Thus the result does not merely predict the majority unknown class.

Eligible directed-query reversal scores100% over all base queries. Paired
inverse wording, coherent table permutation, clause reversal and valid
first-edge inverse replacement also score100% on the declared subsamples:
16scenes/96queries in old joint and new_joint;48/288 in phrasing;64/768 in each
four-node cohort. Samples are SHA256 selected per relation tuple; they do not
cover every base row or all24 four-node table permutations. Sample hashes and
complete intervention confusion matrices are retained.

All rule baselines score100%. Independently trained no-vector macro recall is
20–23.33% on old joint and20% on each four-node cohort. Four-node zero-vector
accuracy is40.625%, matching the unknown-class prevalence; shuffled-vector
accuracy is63.54%. A post-hoc gold-label diagnostic shifts canonical edge labels by one scene,
traverses the current topology and applies the reference rule. It reproduces
exactly63.54%, confirming that the cyclic shuffle preserves substantial
answer information. It is not a chance-level negative control or proof of a
stronger causal effect; this diagnostic did not alter training or gates. Three cyclic mismatched decoder-checkpoint controls
score36.91%,38.74%,52.51% on four-node cohorts. Compatibility remains required;
this is not an all-six-pair alignment study.

## Reproduction and verification

[CI run37787142732](https://github.com/knox9014/AVL/actions/runs/37787142732)
atc261178b3f8a400adcfcf87fa541c84249ee78c7 passed6 focused and149 full tests,
primitive reproduction and the105.505-second CPU2thread operator audit.
All earlier query/binding/alignment/composition/learned-query workflows pass
this source head. The earlier learned-query experiment still retains failed
research gates; a successful workflow does not erase those failures.

The initial test-first run37786646290 failed on the intentionally absent
implementation, without training or final evaluation. No protocol retuning
followed. Tests include actual packet round-trips/invalid frames, connected
chain validation, routing over all24 table permutations, channel-equivariant
outputs and recursive operator reuse with finite gradients.

Primitive checkpoint FILE hashes differ from original binding artifacts.
A post-hoc audit compares tensor-state fingerprints with the immediately prior
flat-query study and finds exact agreement for all three primitive models.
This verifies primitive tensor identity for that historical comparison, not
with every earlier artifact. File and tensor hashes remain unchanged within
the current experiment. Operator state dicts are safe-loaded and verified.

## Message interface and costs

Experimental AVC2 accepts3/4 nodes and2/3 ordered relation vectors with explicit
name table/endpoints. ACQ2 adds a directed query wrapper. The original
AVC1/ACQ1/default AVL1 inference remains unchanged.

| Cohort | Actual query bytes | Total actual | Source + same table/query overhead | Symbolic gold |
| --- | --- | --- | --- | --- |
| old joint | 181 | 417024 | 206784 | 99072 |
| old phrasing | 181 | 1251072 | 608832 | 297216 |
| known4 | 254 | 3121152 | 1557504 | 651264 |
| new4 | 254 | 3121152 | 1557504 | 651264 |

Four-node queries are2.0039x text-plus-table/query framing and4.7925x the53byte
explicit symbolic gold representation. The gold representation's relations are
given, not learned from text. Graph bytes repeat per query; no caching,
compression or network efficiency gain is claimed. Model identity negotiation,
distribution and network framing are excluded and unimplemented.

[Interface guide](../AVL_CHAIN_QUERY_GUIDE.md) documents semantics, packet fields
and a label-free source/query example. Its source grammar was checked locally;
the example uses the same functions exercised by the audit. It is an
experimental interface, not a production compatibility guarantee.

## Records and limits

[Complete durable JSON](../../antlab/runs/shared-operator-report-20261008.json)
retains every gate, support/confusion, subsample hash, source/data/model hash,
historical reference and provenance. Dataset SHA256:
6b73e795b7e23f196cb7b5ef46443b6f7830762c4816101ae9318154d1dd4d9b.
Full synthetic data, weights and reports are also in30-day artifact11554812276,
ZIP SHA2563328d5170091ff1e3f50204689b0ddbf30e996b78cba46ec8821a8569c7a66d2.

Only chains of at most four nodes, explicit English clause segmentation,
literal-name slots and four known coordinate relations are covered.
Branching graphs, negation, contradiction handling, novel concept meanings,
independent model negotiation and AI network expansion are not established.
Author review checked supervised operator cases, routing priors, sparse
three-step positives, intervention sampling, tensor provenance and full costs.

A next useful language milestone is explicit typed assertions with a defined
contract for negation/conflicts and unsupported inputs, with new preregistered
semantic tests. Bigger networks should wait for stronger language coverage.
