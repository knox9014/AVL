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


## 2026-10-08: three-entity designed composition transport

Added experimental AVC1 envelope with three explicit names, two endpoint pairs
and two frozen learned AVL1 relation vectors. Preregistered four partitions and
interventions before implementation. All three matched checkpoints reconstruct
both edges at100% on all final partitions; invariant descriptions/table/clause
order and inverse-relation replacement pass100%. Eligible vector swaps follow
new meaning at100% and retain original meaning0%.136 full tests and6 focused
tests pass at ca3871a3be52801d8de53b9c92718f6106f57227,
run37733118560. Existing query/binding/alignment workflows pass the same head.

No neural model trains on multi-clause fixtures: segmentation/topology and the
derived transitivity rule are designed. All joint derived queries have only
undetermined support. Do not interpret heldout-fixture results as learned
composition or reasoning. Newly reproduced checkpoint file hashes differ from
prior runs; frozen tensor-state hashes are verified within this run.
Messages171bytes are2.1783x final text-plus-table and5.1818x symbolic gold
representation. No compression or model-independent compatibility is established.

Complete report including confusion matrices and source/data/model hashes:
antlab/runs/composition-pilot-report-20261008.json.
Results/usage/limitations: docs/research/2026-10-08-composition-pilot-results.md.
Local staged and tracked scans inspect23 AVL files each; no new composition
findings or binary skips. The earlier alignment Python matrix operator at
semantic_alignment_pilot.py:39 remains the sole email-pattern false positive,
manually reviewed; whole scans still return blocked=true, not a clean pass.
All source fixtures are synthetic; no ACT content or private conversations.
Default AVL1 inference and main are preserved.


## 2026-10-08: learned composition query generalization — failed gates

Preregistered7,749-parameter raw-vector MLP with explicit endpoints/query slots,
three frozen primitive senders and paired separately trained no-vector controls.
Symmetry-closed heldout tuples(0,0),(1,1),(0,2),(3,1) yield12 joint semantic
signatures disjoint from36 train signatures. Heldout positive two-hop answers
include left/right; every base query partition has all five answer classes.

All three receivers fit train100%, but joint accuracy70.31%,68.92%,63.89%
and macro recall70.83%,66.46%,58.65% fail95% gates. Matching primitive-decoder
plus graph-rule baseline scores100%. Direct-query and paired binding robustness
also degrade on new structures. This is a failed generalization pilot, not a
successful learned-language claim. Record pilot_passed=false; no model retuning.
New-name aggregates include known structures and are not the primary holdout.

Seven focused and143 full tests pass at9eeeb7ad143a552e39d92b81efb9196cafb3bc45,
run37760159484, along with existing query/binding/alignment/composition workflows.
A real empty intervention subset previously crashed metrics in run37759791443;
regression-tested fix now records zero support/null scores. Architecture,
training, split and research gates remain unchanged. CPU2thread audit127.054s.
Query wrapper makes181bytes/query versus43byte symbolic gold representation;
graph bytes repeat per question and no compression benefit is claimed.

Full report: antlab/runs/learned-composition-query-report-20261008.json.
Results/limitations: docs/research/2026-10-08-learned-composition-query-results.md.
Local staged/tracked scans inspect31 explicit AVL research files each: no new
learned-query findings or binary skips. The sole earlier alignment Python
matrix-operator false email match remains manually reviewed; whole scans still
return blocked=true, not a clean pass. Synthetic fixtures only; ACT/private
content excluded. Main/default inference unchanged; existing draft PR updated.
These joint results are now observed and must be disclosed if reused.


## 2026-10-08: shared operator and fresh four-node transfer — gates passed

Added experimental AVC2/ACQ2 bounded3/4node chain interface and66parameter
shared learned channel operator over the frozen676parameter relation decoder.
Explicit routing and known inverse-channel permutation are designed, not
learned. Reuse of supervised class semantics differs from the historical raw
MLP and is not an equal-information/capacity ablation.

All three operators pass every preregistered gate: old observed joint/phrasing/
new_joint and fresh known4/new4 accuracy and macro recall100%. Each four-node
cohort has12288queries over64primitive relation triples; all128 positive
distance-three queries also score100%. These three-edge routes never enter
training. Binary operator cases remain supervised/observed. Rule baselines
also score100%; no superiority over rules or novel concept discovery claimed.

149 full tests and6 focused tests pass atc261178b3f8a400adcfcf87fa541c84249ee78c7,
run37787142732. Earlier workflows pass the same head; old raw-MLP research
failure is preserved. Frozen primitive tensor fingerprints match the prior
flat-query experiment for all3seeds despite different checkpoint FILE hashes;
this match is a post-hoc provenance audit. Paired invariance/replacement tests
use declared16/48/64scene subsamples, with hashes/counts retained.

Fresh queries254bytes are2.0039x text-plus-table/query framing and4.7925x
symbolic gold. Graph repeats per query. No compression, network deployment or
independent model negotiation. Learned operator states are safe-loaded and
roundtrip-verified;105.505s CPU2thread audit excluding primitive reproduction.
A post-hoc gold-label shift diagnostic exactly reproduces63.54% shuffle
accuracy, confirming residual answer information in that control.

Complete report: antlab/runs/shared-operator-report-20261008.json.
Results/limitations: docs/research/2026-10-08-shared-operator-results.md.
Language semantics/wire/API example: docs/AVL_CHAIN_QUERY_GUIDE.md.
Local staged/tracked scans inspect40 explicit AVL research files each, with no
new shared-operator/guide findings or binary skips. The earlier alignment
Python @ operator remains the sole manually reviewed false email finding;
whole scans still report blocked=true, not a clean pass. Only synthetic data,
no ACT or private conversation content. Main/default inference preserved.
