# AVL v2 measured results

[English overview](../README.md) | [한국어 개요](../README.ko.md)

2026-10-06. **The frozen v2 study passed.** This result concerns a finite synthetic English grammar, not general English understanding or lossless compression of arbitrary content.

## Method and scope

The [protocol](AVL_V2_PROTOCOL.md) was committed before full training. Historical v1 failure motivated this new study; v2 is not an independent replication. All three final checkpoints were used, without tuning after v2 heldout scores.

The model has 99,977 parameters: 54,897 in the sender and 45,080 in the receiver. Training-derived lexical vocabulary has 41 words. A bidirectional GRU reads word order; masked attention and mean pooling summarize the whole segment into 16 float32 numbers. A vector-only receiver predicts all nine fields. Two training-only surfaces of each frame receive cross-entropy supervision and normalized-vector agreement with weight 0.05.

There are 448 allowed frames. Train has 2,776 rows from 347 frames and 8 templates; validation and phrasing each have 694 rows from these frames and two unseen templates. Combination has 202 rows from 101 reserved frames and two training templates. Joint has 202 rows from reserved frames and unseen phrasing templates, withholding meaning combinations and surface forms together. Every heldout token occurs in training, but no training token sequence occurs in a heldout split after punctuation removal. These templates remain a narrow surface test.

Each seed ran 1,800 optimizer steps; total study time was 488.5 seconds. See the [sanitized numerical environment](../antlab/runs/semantic-v2-environment-20261006.json).

## Exact-frame accuracy

Correct means all nine meaning fields match. Every heldout split must reach at least 95% exact-frame and 95% polarity, certainty and condition accuracy in every seed.

| Seed | Train | Validation | Combination | Phrasing | Joint | All gates |
|---:|---:|---:|---:|---:|---:|---|
| 44 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | PASS |
| 55 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | PASS |
| 66 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | PASS |

## Critical fields

| Seed / split | Polarity | Certainty | Condition | Macro class accuracy |
|---|---:|---:|---:|---:|
| 44 / validation | 100.00% | 100.00% | 100.00% | 100.00% |
| 44 / combination | 100.00% | 100.00% | 100.00% | 100.00% |
| 44 / phrasing | 100.00% | 100.00% | 100.00% | 100.00% |
| 44 / joint | 100.00% | 100.00% | 100.00% | 100.00% |
| 55 / validation | 100.00% | 100.00% | 100.00% | 100.00% |
| 55 / combination | 100.00% | 100.00% | 100.00% | 100.00% |
| 55 / phrasing | 100.00% | 100.00% | 100.00% | 100.00% |
| 55 / joint | 100.00% | 100.00% | 100.00% | 100.00% |
| 66 / validation | 100.00% | 100.00% | 100.00% | 100.00% |
| 66 / combination | 100.00% | 100.00% | 100.00% | 100.00% |
| 66 / phrasing | 100.00% | 100.00% | 100.00% | 100.00% |
| 66 / joint | 100.00% | 100.00% | 100.00% | 100.00% |

Macro class accuracy averages class recalls present in each gold field, then averages the nine fields. Per-class supports, predictions and confusion matrices are saved in the full report.

## Controls: exact-frame accuracy

| Seed / split | Byte transport | Zero vector | Global circular shuffle | Query-only majority | Gold oracle | Direct numerical vector |
|---|---:|---:|---:|---:|---:|---:|
| 44 / train | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 44 / validation | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 44 / combination | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 44 / phrasing | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 44 / joint | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 55 / train | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 55 / validation | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 55 / combination | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 55 / phrasing | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 55 / joint | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 66 / train | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 66 / validation | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 66 / combination | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 66 / phrasing | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| 66 / joint | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |

Shuffle operates on the entire split using one deterministic nonzero circular offset, with no original row retained at its index. Ordered data and multiple surfaces of the same frame can still share features or meanings, so it is not an unconstrained random null. Query-only uses independent training-majority labels, which may form an absent frame. The oracle returns gold targets by construction and does not validate semantic annotations independently. Direct numerical vectors use the same trained model and only check codec equality; they are not a separately trained raw-text classifier baseline. No advantage over such a classifier is established.

## Transport costs per seed

| Split | Raw ASCII source bytes | Payload bytes | Header bytes | Total wire bytes | Change from source |
|---|---:|---:|---:|---:|---:|
| train | 168,432 | 177,664 | 264 | 177,928 | +5.64% |
| validation | 45,139 | 44,416 | 72 | 44,488 | -1.44% |
| combination | 11,747 | 12,928 | 24 | 12,952 | +10.26% |
| phrasing | 44,819 | 44,416 | 72 | 44,488 | -0.74% |
| joint | 12,861 | 12,928 | 24 | 12,952 | +0.71% |

Each segment carries 64 payload bytes; each batched packet adds 12 header bytes. Totals are identical across seeds. These measurements exclude network framing, authentication, checkpoint distribution, storage and retransmission. Float32 transport retains encoder vectors bit-for-bit; vector meaning accuracy is a separate question.

The finite ontology contains only 448 frames: an already-known gold frame could theoretically be indexed in 9 bits using a shared codebook. This is not a tested parser or a fair learned baseline, but shows that 512-bit latent payloads are not an efficient encoding of this small ontology. Useful open-ended meaning and practical compression remain future work.

## Interpretation and remaining work

A separate [post-hoc coverage audit](../antlab/runs/semantic-v2-coverage-20261006.json)
enumerates every admitted canonical grammar form: 448 frames × 12 templates
= 5,376 examples. This set includes training examples and is not a new heldout
benchmark. The 808 forms outside all primary splits were evaluated separately.
All three checkpoints scored 100% exact-frame on both sets. This audit never
changes the primary gates and does not cover new words or arbitrary English.

The [actual vector demonstration](../antlab/runs/semantic-v2-demo-20261006.json)
shows two received predictions, their 16-number vectors, packet bytes and
canonical English reconstructions. The conditional negative possibility about
a door and a budget request match separate audit targets. A 17 USD input is
retained unchanged as unsupported. This chosen example is a post-hoc
illustration, not a success estimate.

V1 failed on rearranged wording. V2 changes architecture, data and objective together; its scores are on a different frozen dataset and cannot isolate which change caused an improvement. No higher model capacity or learned generic knowledge is implied by this finite success criterion.

Model outputs remain predictions. Admission is finite grammar membership, not correctness verification. The caller defines segment boundaries, and originals are retained locally for unsupported fallback. New words, arbitrary amounts, compound propositions and nested scope remain unsupported. Automatic semantic organization and general AI-to-AI reasoning are unfinished.

The deterministic English renderer uses only received predicted fields, rejects contradictory field combinations, and does not recover original wording or establish model correctness. Sender and receiver require the same checkpoint: the AVL1 codec version is not learned-model identity or authentication.

## Reproduce

[Full report](../antlab/runs/semantic-v2-20261006/report.json) · [Configuration](../antlab/runs/semantic-v2-20261006/config.json) · [Manifest](../antlab/runs/semantic-v2-20261006/manifest.json) · [Source hashes](../antlab/runs/semantic-v2-20261006/source_hashes.json) · [English usage](USAGE_V2.md) · [한국어 사용법](USAGE_V2.ko.md)

Verification regenerates data, checks frozen configuration, source/artifact hashes and finite checkpoint metadata, checks training logs and parameter movement from seeded initialization, then replays every split and control. It replays inference, not optimization. Historical v0/v1 failure artifacts remain unchanged.

The full code suite passed 60 tests. That check includes packet validation,
ordered-input/padding behavior, training-only pair selection, joint gates,
tampering rejection, all 448 canonical rendered frames and coverage partition
integrity. Code correctness and the finite model benchmark are separate claims.

한국어: 시드 세 개의 고정 v2 실험은 성공 기준을 통과했습니다. 문장 표현과 의미 조합을 동시에 바꾼 joint 시험도 포함합니다. 결과는 정해진 영어 문법에만 해당하며, 범용 의미 보존이나 실용적인 압축을 완성한 것은 아닙니다.
