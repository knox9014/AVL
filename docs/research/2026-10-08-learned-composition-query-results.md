# Learned AVL composition query receiver — 2026-10-08

**Research gates failed.** Three raw-vector MLP receivers fit every training query,
but reach only63.89–70.31% accuracy on withheld semantic structures. The frozen
primitive decoder plus graph/transitivity rules answers100% from the same
packets. This demonstrates a generalization failure of this receiver design,
not missing relation information in the tested packet format.

## Preregistered design

[Protocol](../AVL_LEARNED_COMPOSITION_QUERY_PROTOCOL.md) was committed at
c5995ed045e73acd40aca4b5654b0f202cf981fc before implementation/final evaluation.
A new7,749-parameter MLP receives32 raw vector coordinates,12 one-hot edge
endpoint features and6 query-index features. The50->64->64->5 network uses
GELU activations. It receives neither oracle labels nor frozen decoder outputs.
Three sender seeds44/55/66 are paired with receiver seeds144/155/166.
Each receiver and its separate no-vector control train3000steps on the same
class-balanced128-example batches, AdamW0.003, weight_decay0.0001, clip1.
There is no validation selection, extra curriculum or post-test tuning.

The primitive model is reproduced from the unchanged binding source/data
configuration and frozen during the experiment. Newly reproduced checkpoint
file hashes differ from earlier runs; old tensor identity is not established.
Within-run tensor-state/file hashes are verified before and after.

Six directed queries are asked for every three-node scene:AB,BA,BC,CB,AC,CA.
Training excludes(0,0),(1,1),(0,2),(3,1), where0:left,1:right,2:above,3:below.
This set is closed under chain reversal with inverted edge relations.
Training demonstrates positive above/below transitivity; heldout positive path
questions require left/right. All five answer classes occur in training via
direct queries or path questions.

Query-rooted graph signatures remove name, table-order, edge-order and equivalent
inverse-description distinctions. Train has36 unique semantic signatures; joint
has12, with zero overlap. Phrasing has the same36 as train; new_names mixes all48.
new_joint uses the same12 heldout signatures with copied new identifiers and
is a subset of new_names, not an independent semantic confirmation.

## Results

| Sender / receiver seed | Train accuracy | Phrasing accuracy | Joint accuracy / macro recall | Joint direct | Joint positive path | Joint undetermined |
| --- | --- | --- | --- | --- | --- | --- |
| 44 /144 | 100% | 100% | 70.31% /70.83% | 82.29% | 21.88% | 70.83% |
| 55 /155 | 100% | 96.93% | 68.92% /66.46% | 71.35% | 83.33% | 44.79% |
| 66 /166 | 100% | 100% | 63.89% /58.65% | 64.58% | 81.25% | 43.75% |

Joint contains2304 queries per receiver. Full class support is
[768,768,192,192,384]; two-hop support is[192,192,0,0,384].
It therefore includes supported positive left/right queries and genuine
undetermined queries. Zero-support classes are never silently scored perfect.
new_joint metrics equal joint after the literal adapter, as expected.
All16-structure new_names accuracy is92.58%,89.93%,90.97%; its mixture of known
and heldout structures makes that aggregate easier than joint.

All matching rule controls achieve100% in every base partition. Independently
trained no-vector joint macro recalls are28.33%,30%,29.58%.
Zero-vector main-receiver accuracy is24.31%,20.14%,20.14%.
Shuffled-vector accuracy is36.46%,38.37%,37.67%; removed topology/query accuracy
31.42%,28.82%,35.42%. Three cyclic wrong-sender-checkpoint controls score26.74%,
23.26%,14.06%; no all-six-pair study or checkpoint compatibility is claimed.

Joint paired table-permutation accuracy is54.51%,48.61%,39.06%.
Equivalent-description paired accuracy is68.40%,67.01%,61.81%; reversed-clause
paired accuracy61.11%,61.46%,46.18%. Eligible directed-query reversal is61.67%,
65%,56.67%, with1920 eligible directed queries per receiver. Thus failures extend
to binding/context robustness and are not confined to two-hop answers.

First-edge inverse replacement on joint scores100%,98.96%,100%, but paired
original-and-replacement correctness remains70.31%,68.58%,63.89%.
That replacement maps all heldout path questions to undetermined; its positive
path subset has zero support and records null accuracy/macro recall. A high
replacement aggregate does not repair failed original-structure generalization.
On phrasing, replacement accuracy is also below the95% gate for all receivers.
See complete report for every intervention, support and confusion matrix.

Every receiver fails the preregistered all-partition gates. Ordinary tests and
successful workflow completion indicate that the experiment executed; they do
not mean the research hypothesis passed. The report retains pilot_passed=false.

## Interpretation and next experiment

Training accuracy100% with strong seen-structure phrasing scores and weaker
heldout scores is consistent with context-specific fitting by this flat MLP.
This is an interpretation, not a proof of its internal mechanism or a limit
on all neural receivers. Rule success shows the tested primitive vectors remain
decodable. Merely adding more text forms or reporting easier renamed aggregates
would not answer the heldout-structure question.

A defensible next design shares relation-processing weights and uses graph
endpoints explicitly, so node/edge reorderings preserve the relevant computation.
Compare that architecture against this frozen failure baseline and the rule
receiver under a new preregistered protocol. These test results are now observed;
any reuse must be disclosed rather than described as a fresh blind test.
No new feature, hyperparameter or training change was selected from these scores.

## Cost, verification and records

The ACQ1 query wrapper has8bytes of framing plus2 query indices around an
unchanged171byte AVC1 graph:181bytes/query. The graph is repeated for each query;
no caching or network performance benefit is claimed.

| Partition | Queries | Actual transmitted | Source + same table/query overhead | Symbolic gold |
| --- | --- | --- | --- | --- |
| joint | 2304 | 417024 | 206784 | 99072 |
| phrasing | 6912 | 1251072 | 608832 | 297216 |
| new_names | 9216 | 1668096 | 815616 | 396288 |
| new_joint | 2304 | 417024 | 206784 | 99072 |

The symbolic gold representation uses43bytes/query with the same framing,
names, topology and query indices;181/43=4.2093x. Its relation labels are gold,
not the output of a trained text parser. Model distribution/identity negotiation
and network framing are excluded. Default inference and AVL1 remain unchanged.

[CI run37760159484](https://github.com/knox9014/AVL/actions/runs/37760159484)
at9eeeb7ad143a552e39d92b81efb9196cafb3bc45 passed7 focused and143 full tests.
Original query, binding, alignment and composition workflows also passed this
head. The127.054-second CPU2thread experiment includes about9seconds training
per receiver/control pair, excluding upstream primitive reproduction.

Initial test-first run37759414052 failed on the absent implementation, with no
model execution. Run37759791443 subsequently exposed an empty intervention
metric subset; a local function-level regression reproduced the exception.
The fix explicitly records zero support/null scores, with a new regression test.
Model architecture, optimizer, steps, split and gates were unchanged.
A pre-evaluation sampler-validation optimization only replaced scalar Python
validation with the equivalent tensor range check. No final performance tuning.

[Complete durable JSON](../../antlab/runs/learned-composition-query-report-20261008.json)
retains all failed gates, full confusion/support, semantic signatures, source/data/
weight hashes and provenance. Dataset SHA256:
d142b193f8a267217ebb54cba8abb4d994e6140c4f7544b56afa2d0ebc7050c4.
Synthetic data, receiver/control weights and reports are also in30-day
artifact11542111414; ZIP SHA256
38be63433e7ff5047548bdc26b6f25da8bf2ce67525eeba653f1dd9de31c2685.
Author review checked feature provenance, semantic split, sparse subsets,
frozen states, controls and the distinction between CI and research success.

Reproduce with the pinned CPU runtime and fresh directories:

```sh
python -m antlab.semantic_binding_pilot --output binding-reproduction
python -m antlab.semantic_learned_query_pilot --study binding-reproduction --output learned-query-output
```

Only synthetic fixtures are used; this is an AVL research branch, not ACT
publication, autonomous language completion or an AI internet deployment.
