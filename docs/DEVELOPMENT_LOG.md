# AVL development log

Entries record implemented behavior, measured evidence, limitations and next
questions. Use synthetic examples only. Never copy user conversations, account
identifiers, personal paths, credentials or raw exception logs into this log.

## 2026-10-08

Implemented and executed the separately supervised two-subject spatial binding
pilot. All3 seeds passed within-checkpoint accuracy, macro recall and paired
meaning-change gates at100%; the separately trained constant-input receiver
scored25%. Independent checkpoint cross-play averaged18.75% across6 pairs.
This is bounded four-relation learning with a designed name-to-slot adapter:
only16 unique canonical training strings. Repeated identity pairs do not count
as new semantic structures. Joint vector packets plus literal tables cost2.24x
the text reference and6x the one-byte gold oracle. No efficiency or universal
compatibility gain is claimed.

Validation:128 repository tests and8 focused tests passed in
[run37720958084](https://github.com/knox9014/AVL/actions/runs/37720958084).
[Protocol](AVL_BINDING_PILOT_PROTOCOL.md),
[results and durable evidence](research/2026-10-08-binding-pilot-results.md).
Original inference remains unchanged. Next: explicitly supervised codebook
alignment/sample-efficiency, then richer canonical structures.

## 2026-10-07

Named literal extension: added optional AVN1 packets with explicit identifiers
and uint64 values around one frozen learned relation vector. All3 seeds retained
100% exact full meanings across2560 sources each;69 numeric boundary queries per
seed also scored100%. Full suite now passes120 tests. Complete wire bytes270336
versus223296 UTF-8 source bytes:21.07% larger. This is exact literal transport,
not learned new entity concepts or number semantics. Same-checkpoint and finite
known-role grammar limits remain. [Protocol](AVL_NAMED_VALUES_PROTOCOL.md),
[usage](AVL_NAMED_VALUES_USAGE.md),
[results and immutable evidence](research/2026-10-07-named-values-results.md).


1. Reaffirmed AVL as a learned language for AI use. Human-readable frames,
   English renderers and answer rules remain diagnostic scaffolding.
2. Added a conservative semantic-query baseline for requests, uncertainty,
   condition scope, unknown answers and numeric boundaries.
3. Declared and executed a three-seed frozen-vector learned query pilot.
   Metadata questions reached100%, but proposition macro recall was about33%.
   Its success gate failed despite roughly98% headline accuracy.
4. Diagnosed outcome imbalance: the query-only majority reached99.1337%
   proposition accuracy. The learned head recovered supported claims in
   training but missed heldout supported claims. Existing field-decoder rules
   recovered all query answers; loss of all relevant vector information is
   not supported by these observations.
5. Declared a matched-capacity follow-up comparing raw vectors with frozen
   decoder probability features. Both arms failed their per-family gates;
   feature reuse did not demonstrate a general improvement.
6. Preserved primary references, protocols, class-level summaries and failure
   analysis in [the research record](research/2026-10-07-semantic-use.md) and
   [results](research/2026-10-07-query-pilot-results.md). CI artifacts contain
   full reports and head weights, with30-day retention. Durable summaries
   retain per-class supports, recalls and controls in the repository.

7. Revision3 introduced factorized learned operators with an abstract semantic
   curriculum. All three frozen sender/receiver checkpoints passed every
   unchanged query-family gate:100% accuracy and macro recall on the observed
   joint set, plus100% in the separate all-grammar coverage audit.
   [Results and limitations](research/2026-10-07-query-operators-results.md).
   This reuses the supervised field receiver and fully teaches the abstract
   operator patterns; it is not new codebook learning or universal language.
   Added2,758operator parameters; total inference102,735parameters.

Validation: [112 repository tests passed](https://github.com/knox9014/AVL/actions/runs/37600152482)
on Python3.12/PyTorch2.6.0CPU; the stdlib query/renderer/data suites also passed
on Python3.10 and3.12. Original v2 protocol, source files, checkpoints and gates
were preserved. Passing implementation tests does not make failed research
gates pass. Revisions1/2 remain failed; revision3 separately passed. New findings
are bounded and reuse previously observed v2 data.
No network expansion or universal AI-language completion is claimed.

Next: declare a richer heldout semantic-use study, with more distinct grounded
claims, explicit equally available world context where needed, and controlled
tests of query binding and factor composition. Do not tune repeatedly against
the same observed joint set.

## 2026-10-06

1. Frozen semantic v2: 99,977 parameters, controlled English, 16-dimensional
   float32 vectors. Three checkpoints passed primary finite-grammar gates.
   [Results](AVL_V2_RESULTS.md). General language understanding remains unproven.
2. Added optional local M2M100 translation around AVL. Eight-language probe
   generated 24 translations; none entered AVL's grammar, and some translation
   errors were observed. [Evidence](MULTILINGUAL_RESULTS.md).
3. Cross-play diagnostic: matching checkpoints scored 100%; six independent
   checkpoint pairs averaged 0.1650% exact frame recovery on 202 joint inputs.
   [Evidence](AI_COMMUNICATION_REFERENCES.md#avl-cross-play-diagnostic).
4. Batched local translation and request-local duplicate reuse: 36 synthetic
   inputs took 94.020 s individually, mean 38.284 s in two batch runs, with
   identical outputs and approximately 2.1% more peak process memory.
   [Evidence](TRANSLATION_EFFICIENCY.md). Meaning coverage did not improve.
5. Added an explicit experimental int8 wire codec: vector payload 64 → 20 bytes.
   All three seeds retained 100% exact frames on all 5,376 admitted surfaces;
   no predictions changed. Conversion added CPU work and remained lossy.
   [Evidence](INT8_WIRE_RESULTS.md). No cross-model alignment or universal meaning claim.
6. Added [publication privacy checks](PUBLICATION_PRIVACY.md): inspect Git index
   text, redact matched values, flag recognizable identifiers/secrets, explicitly
   list binary exclusions and require separate metadata/commit review.
   Pattern scanning does not guarantee absence of every personal detail.

Validation for this addition: 87 tests passed; staged and full-index text scans
found no recognized private identifiers or secrets. Seven existing checkpoints
were loaded safely and their string metadata had no findings. Commit identities
remain the neutral project identity. No new model weights were published.

Next research questions: align independently trained vector spaces; broaden
semantic coverage with reserved evaluation data; evaluate direct multilingual
representations; measure communication efficiency under realistic network
latency. Connection scaling and general intelligence remain tested hypotheses.

한국어: 개발 과정은 합성 입력·측정 결과·한계·다음 연구 질문 중심으로
기록합니다. 사용자의 대화와 개인정보는 공개 기록에 옮기지 않습니다.


## 2026-10-08: frozen binding-space calibration

Preregistered 54 trials across six model pairs, three calibration sizes and three
selection seeds. All declared16-example gates passed; eight examples descriptively
reached100% on every partition. Four-example bridge joint mean99.31%, minimum87.5%.
Fresh label classifiers also reach100% at8/16; no bridge superiority is established.
Wrong-pair and wrong-label controls score0% at8/16. The task retains only four known
classes and16 canonical training strings. Reproduced checkpoint file hashes differ
from the original run; tensor identity has not been established. Weights remained
frozen throughout calibration. No default inference or network expansion.

Run37724409724 at441ccba1508116a5e46f4bed99824933dff8ced2 passed2 focused and130 full tests;
existing query and binding workflows also passed this head. Durable summary and
results: docs/research/2026-10-08-alignment-pilot-results.md. Synthetic-only review.
Local tracked-content scan inspected15 AVL files and flagged one false email match
at semantic_alignment_pilot.py:39: Python matrix multiplication design.T@y.double().
Manual source review confirms operators/identifiers, not an email or private data.
The text-pattern scanner itself did not return a clean pass. ACT content excluded.
