# AVL development log

Entries record implemented behavior, measured evidence, limitations and next
questions. Use synthetic examples only. Never copy user conversations, account
identifiers, personal paths, credentials or raw exception logs into this log.

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
