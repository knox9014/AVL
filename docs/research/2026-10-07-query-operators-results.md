# AVL revision 3: learned semantic operators

2026-10-07. [Protocol](../AVL_QUERY_PILOT_V3_PROTOCOL.md) was committed before
execution. [Executed run](https://github.com/knox9014/AVL/actions/runs/37600152482)
used source head7b6b217a2569c88a99ea4163135ad89db7271f79.
All three seeds passed the unchanged >=95% accuracy AND macro-recall gates
for every query family. This is a bounded receiver improvement, not completion
of a universal AI language.

## What changed

The frozen v2 sender and field receiver remain unchanged. Instead of a flat
query MLP, two learned neural operators consume predicted semantic factors.
One combines scope, certainty, polarity and query matching; the other interprets
numeric constraints using predicted amount/comparator and the query candidate.
Metadata answers reuse the existing learned decoder.

The operators are trained on64 abstract Boolean patterns and1,218 numeric
patterns, with class-balanced sampling. No v2 texts/row IDs/gold frames enter
their training or inference. However, the abstract semantic patterns overlap
with test meanings, and the frozen factor receiver was already supervised.
Do not call all evaluation semantics unseen throughout training. No handwritten
answer oracle chooses inference answers; hard classifier decisions, typed factor
selection and arithmetic input features are designed inductive biases.

Both architecture and curriculum changed from the failed pilots. The comparison
does not identify which change caused improvement. This is not new codebook
learning, independent-model alignment, or a rule-free emergent-language claim.

## Results

| Sender seed | Kind | Certainty | Condition | Proposition | Numeric constraint |
|---|---|---|---|---|---|
| 44 | 100% | 100% | 100% | 100% | 100% |
| 55 | 100% | 100% | 100% | 100% | 100% |
| 66 | 100% | 100% | 100% | 100% | 100% |

Every table entry is both accuracy and represented-class macro recall on the
previously observed joint set of202 segments and24 questions per segment.
For proposition queries, all4 entailed,10 contradicted and1,602 undetermined
instances were correct. These include only2 and5 distinct supported/contradicted
frames, each with two paraphrases; they are not independent observations.
The previous revision-1 head had0% recall on both supported classes.

A separate coverage audit over all5,376 permitted grammar surfaces also reached
100% accuracy and macro recall in every family for each seed. This includes
sender-training examples and is not new generalization evidence. All three
operators' abstract training curricula were classified100% correctly; that is a
training metric, not a test result.

## Controls

| Seed | Zero-vector proposition macro recall | Shuffled proposition macro recall | Zero numeric macro recall | Shuffled numeric macro recall |
|---|---|---|---|---|
| 44 | 33.3333% | 33.0420% | 33.7327% | 36.1534% |
| 55 | 33.3333% | 33.0420% | 33.3333% | 27.5067% |
| 66 | 33.3333% | 33.0420% | 33.7327% | 20.9384% |

The query-only majority remains99.1337% proposition accuracy but33.3333% macro
recall. Unlike that baseline, the transmitted-vector receiver recovers both
supported answer classes. Corruption happens before the frozen receiver.
This supports use of message information within the bounded task, not arbitrary
reasoning. Shuffled rows can retain some identical meanings.

## Resources, verification and replay

Two1,379-parameter operators add2,758 learned parameters. With the reused frozen
sender54,897 and receiver45,080 parameters, inference totals102,735 parameters.
Six operator trainings, inference and coverage took about21.34seconds in this
CI run, excluding installation, unit tests and upload. This is not a network
latency benchmark. Existing field-decoder/rule answers were already100%, so this
does not demonstrate superiority to that simpler deterministic baseline.

The full repository suite passed112tests on Python3.12/PyTorch2.6.0CPU.
The stdlib query/renderer/data suites passed18tests on Python3.10 and3.12.
Original v2 sources, protocol, checkpoints and primary gates were preserved.

[Durable class-level summary](../../antlab/runs/query-pilot-r3-summary-20261007.json)
contains all controls, class support/recall, training metrics, source/data hashes,
operator hashes and resource counts. The run artifact contains raw reports,
curriculum data and operator weight files, retained for30days:
artifact11472870331, archive SHA256
7c97eb96b8ad1e2e9fecf003ed178bc1c232293dfe5d0452a8cdf252c2f8d087.

Reproduce from the research branch, using a fresh output directory:

~~~bash
python -m unittest discover -s antlab/tests
python -m antlab.semantic_query_operators --output antlab/runs/my-operator-study
~~~

The script never overwrites an existing output. It verifies the saved primary
data/checkpoint hashes, records unchanged checkpoint hashes and saves learned
operator states. The manifest provides audit consistency, not authentication.

## Remaining work

The current achievement is a learned semantic-use path for existing bounded
messages. Broader language content, fresh reserved tasks and entity/numeric
ranges, sample-efficient acquisition of a frozen codebook, and interoperability
between independent senders/receivers remain unproven. Extend those in separate
studies rather than treating nine typed factors as the final language ceiling.
The [earlier failed studies](2026-10-07-query-pilot-results.md) remain preserved.
