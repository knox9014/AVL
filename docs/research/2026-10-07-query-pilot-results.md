# Frozen-vector query pilot: first result

2026-10-07. Revision-1 training and gates were committed before execution in
[the pilot protocol](../AVL_QUERY_PILOT_PROTOCOL.md).
[Executed CI run](https://github.com/knox9014/AVL/actions/runs/37591546190)
used source head 42938e37745ff0fb6bab254dcdf0c54d3cde0af8.
The code ran successfully; the research success gate FAILED.

## Observations

| Sender seed | Kind / certainty / condition accuracy | Proposition accuracy | Proposition macro recall | Numeric accuracy | Numeric macro recall |
|---|---|---|---|---|---|
| 44 | 100% each | 98.8243% | 33.2293% | 96.3062% | 94.4262% |
| 55 | 100% each | 98.8243% | 33.2293% | 96.3442% | 93.4416% |
| 66 | 100% each | 98.4530% | 33.1045% | 96.1538% | 94.4708% |

The receiver was newly initialized and trained on serialized-vector-derived
inputs, question IDs and numeric candidate features. Sender weights stayed
fixed. The full three-seed pilot took approximately 7.0 seconds in this CPU CI
run, excluding installation, unit tests and upload. This is not a network or
end-to-end application benchmark.

Every seed failed the >=95% per-family macro-recall gate. The high proposition
accuracy must not be presented as strong semantic use: the label distribution
is dominated by undetermined answers. Class recalls and controls require
inspection before attributing this failure to a particular mechanism.
No inference that the vector lacks the information follows from this head's
failure. The existing field decoder is a different, already trained receiver.

## Validation and artifacts

The initial rule/renderer/data suites passed 18 tests on Python 3.10 and 3.12.
The learned-head/semantic-rule/codec/v2-model suites passed 22 tests on Python
3.12 with PyTorch 2.6.0 CPU. These are selected suites, not a full-repository
test result and not an exact replay of the original Windows/PyTorch environment.

The CI artifact named frozen-vector-query-pilot contains a JSON report and
three head weight files; retention is 30 days. Artifact ID 11468383945;
archive SHA256 5a59b08c871dc452913a8bc54d8ccfa22fbce35cabb20e8efcc5b801ab91ee09.
The report includes per-class support/confusion, all declared controls, model
and source hashes, and training metadata. Download from the linked run while
retained. This table is a durable summary, not a replacement for the raw report.

## Post-hoc follow-up

A separately marked diagnostic adds evaluation on head-training examples and
prints class-level controls. It does not change questions, sampling, head
architecture, steps, seeds, test partitions or gates. It is a diagnostic rerun,
not an independent replication. Subsequent architectural or objective changes
must be recorded as a new study revision rather than overwrite this failure.

The goal remains a language AI can learn and use. Rule answers and human-defined
fields are evaluation aids. Network expansion and a universal-language claim
are outside this pilot.

## Diagnostic result: information retained, use did not generalize

[Diagnostic run](https://github.com/knox9014/AVL/actions/runs/37591754075)
retained all original training choices. In every seed, training recall was 100%
for entailed and contradicted proposition answers, but joint-test recall was
0% for both. Decoded-field rules scored 100% on every query family. This is
evidence that useful information remains accessible through the existing
receiver, while the newly trained query head failed the heldout combinations.
It does not isolate a unique architectural cause.

The joint proposition queries contain 4 entailed, 10 contradicted and 1,602
undetermined labels. Each heldout frame has two paraphrases: positive/negative
claims therefore have only 2 and 5 distinct supporting frames respectively.
The query-only majority scores 99.1337% overall, higher than the learned head.
The 1,616 question instances must not be described as 1,616 independent facts.
Future studies need more varied supported claims and balanced outcome reports.

[Durable class-level summary](../../antlab/runs/query-pilot-r1-summary-20261007.json)
preserves supports, recalls and controls from the diagnostic log.

## Revision 2: frozen learned-feature reuse did not fix the failure

[Protocol](../AVL_QUERY_PILOT_V2_PROTOCOL.md) was committed after the first
failure and before running the new comparison.
[Run](https://github.com/knox9014/AVL/actions/runs/37592141280)
used source head e0cd5874c1478100838a7b315df0f748cab38b9f.
Both arms use the same 10,349-parameter query head and training budget.
One receives raw vectors padded to 72 features; the other receives the frozen
field receiver's 72 soft probabilities, requiring 45,080 reused pretrained
decoder parameters. This is an auxiliary-structure study on reused observed
data, not an independent confirmation or cost-equivalent inference comparison.

| Seed | Raw proposition macro recall | Feature-reuse proposition macro recall | Raw numeric macro recall | Feature-reuse numeric macro recall |
|---|---|---|---|---|
| 44 | 33.2501% | 32.7507% | 94.2586% | 94.6957% |
| 55 | 33.2293% | 32.7923% | 93.0366% | 92.1004% |
| 66 | 58.1669% | 32.0433% | 93.5677% | 92.4767% |

Kind, certainty and condition were 100% in both arms for all three seeds.
Both arms FAILED their all-family, all-seed gates. There is no demonstrated
general improvement from the feature-reuse change. Do not select the better
single seed or revise gates to make the comparison pass.
The six head trainings and evaluation took approximately12.1seconds in this
run, excluding setup and tests. Selected receiver/transport suites passed25tests.

[Durable revision-2 summary](../../antlab/runs/query-pilot-r2-summary-20261007.json)
preserves each family and control's support/recall. Full reports and six head
weights are retained in the run artifact for30days.

## Next research decision

Stop treating a larger flat query MLP or decoder-feature reuse as a demonstrated
solution. The next separately declared study should distinguish:

1. insufficient grounded-claim diversity in training and evaluation;
2. failure to bind query subjects/predicates to message content;
3. failure to compose certainty, polarity and conditional scope.

A compositional receiver with auxiliary factor learning is a candidate, not a
proven improvement. Preserve raw-vector access, explicit resource accounting
and unseen combinations. Design a richer reserved test before tuning further.
No additional architecture has been trained on the basis of these results.
The final objective remains an AI-learned language beyond nine fixed fields.
