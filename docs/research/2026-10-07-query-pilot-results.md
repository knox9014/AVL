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
