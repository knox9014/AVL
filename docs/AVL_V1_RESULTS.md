# AVL v1 measured results

[English overview](../README.md) | [한국어 개요](../README.ko.md)

2026-10-06. **The preregistered study failed.** The local encode/packet/decode interface runs, but meaning retention is unreliable on unseen phrasing. This is a research prototype, not a completed general AI language.

## Method

Frozen [protocol](AVL_V1_PROTOCOL.md): 98,511 total parameters (29,104 sender; 69,407 receiver), one 16-dimensional float32 vector per caller-defined segment, nine predictions from that vector, final checkpoints for seeds 11/22/33 and 1,500 AdamW steps each. No setting or checkpoint was selected using heldout scores. Evaluation crosses actual serialized bytes. The receiver receives neither source text nor gold labels.

Synthetic finite grammar: 448 frames (256 device, 192 budget). Train has 1,101 rows covering 367 frames with three templates; validation has 367 rows using template 3; combination has 81 entirely reserved frames using training template 0; phrasing has 367 seen frames using template 4. Normalized full text is disjoint across splits. New templates mostly rearrange common words and punctuation; this is a narrow surface test, not broad paraphrase understanding. Admission is only a boolean whitelist and supplies no semantic labels to the trained encoder.

Training and evaluation took 760.8 seconds in the [sanitized recorded environment](../antlab/runs/semantic-v1-environment-20261006.json).

## Exact-frame accuracy

An example counts as correct only when all nine fields match. Each heldout split must reach 95% exact-frame and 95% polarity, certainty and condition accuracy in every seed.

| Seed | Train | Validation | Unseen combinations | Unseen phrasing | All heldout gates |
|---:|---:|---:|---:|---:|---|
| 11 | 100.00% | 64.03% | 100.00% | 92.37% | FAIL |
| 22 | 100.00% | 72.21% | 100.00% | 100.00% | FAIL |
| 33 | 100.00% | 50.14% | 100.00% | 89.37% | FAIL |

## Important heldout fields

| Seed / split | Polarity | Certainty | Condition | Macro class accuracy | Gate |
|---|---:|---:|---:|---:|---|
| 11 / validation | 88.28% | 99.46% | 83.92% | 81.52% | FAIL |
| 11 / combination | 100.00% | 100.00% | 100.00% | 100.00% | PASS |
| 11 / phrasing | 100.00% | 100.00% | 92.92% | 98.43% | FAIL |
| 22 / validation | 99.73% | 95.91% | 97.55% | 88.24% | FAIL |
| 22 / combination | 100.00% | 100.00% | 100.00% | 100.00% | PASS |
| 22 / phrasing | 100.00% | 100.00% | 100.00% | 100.00% | PASS |
| 33 / validation | 87.19% | 89.10% | 83.11% | 79.67% | FAIL |
| 33 / combination | 100.00% | 100.00% | 100.00% | 100.00% | PASS |
| 33 / phrasing | 100.00% | 100.00% | 89.37% | 98.81% | FAIL |

Macro accuracy averages each field’s mean accuracy over classes present in its gold split, then averages the nine fields. Full supports and confusion matrices are saved in the report.

## Controls: exact-frame accuracy

| Seed / split | Transmitted | Zero vector | Shuffled vector | Query-only majority | Gold-label oracle |
|---|---:|---:|---:|---:|---:|
| 11 / train | 100.00% | 0.00% | 2.45% | 0.00% | 100.00% |
| 11 / validation | 64.03% | 0.00% | 0.27% | 0.00% | 100.00% |
| 11 / combination | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| 11 / phrasing | 92.37% | 0.00% | 0.54% | 0.00% | 100.00% |
| 22 / train | 100.00% | 0.00% | 2.27% | 0.00% | 100.00% |
| 22 / validation | 72.21% | 0.00% | 1.63% | 0.00% | 100.00% |
| 22 / combination | 100.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| 22 / phrasing | 100.00% | 0.00% | 1.63% | 0.00% | 100.00% |
| 33 / train | 100.00% | 0.00% | 1.91% | 0.00% | 100.00% |
| 33 / validation | 50.14% | 0.00% | 0.82% | 0.00% | 100.00% |
| 33 / combination | 100.00% | 0.00% | 2.47% | 0.00% | 100.00% |
| 33 / phrasing | 89.37% | 0.00% | 0.82% | 0.00% | 100.00% |

Shuffle permutes vectors within each evaluation batch (up to 128 examples), retaining field selectors. Ordered data can make batches homogeneous on some fields, and permutations can retain original rows; this is not a global derangement. Query-only uses independent training-majority labels, which can form a frame absent from the split. The gold-label oracle returns targets by construction; it is not learned performance or independent validation of semantic annotations. No raw-text classifier baseline was run, so no efficiency or superiority claim over one is supported.

## Actual transport cost per seed

| Split | Original source bytes | Payload bytes | Header bytes | Total wire bytes | Change from source |
|---|---:|---:|---:|---:|---:|
| train | 67,695 | 70,464 | 108 | 70,572 | +4.25% |
| validation | 23,325 | 23,488 | 36 | 23,524 | +0.85% |
| combination | 4,582 | 5,184 | 12 | 5,196 | +13.40% |
| phrasing | 23,418 | 23,488 | 36 | 23,524 | +0.45% |

Wire costs are identical across seeds: 64 payload bytes per segment plus 12 bytes per batched packet. These short texts do not become smaller in this study. Network framing, authentication, model distribution, storage and retransmission are excluded. Representation size and preserved meaning must be established separately.

## Demonstration and verification

The [post-hoc demonstration](../antlab/runs/semantic-v1-demo-20261006.json) includes two synthetic sources, actual 16-number vectors, packet hex, predicted frames and separate audit targets. Seed 11 correctly recovered a conditional negative possibility about a door and a budget request; a 17 USD source was rejected unchanged. The two vectors occupy 140 wire bytes. This chosen example is not a heldout success estimate. Targets were compared after reception, never given to the receiver.

The full test suite passed 36 tests. Artifact replay checks all three checkpoints, every split/control, data regeneration, source and artifact hashes, configuration, finite weights, training-log schedule and parameter movement from seeded initialization. It replays inference, not optimization.

[Report](../antlab/runs/semantic-v1-20261006/report.json) · [Configuration](../antlab/runs/semantic-v1-20261006/config.json) · [Manifest](../antlab/runs/semantic-v1-20261006/manifest.json) · [Source hashes](../antlab/runs/semantic-v1-20261006/source_hashes.json)

## Interpretation and remaining work

The vectors support reserved semantic combinations in this finite grammar, while new wording exposes failures despite 100% training accuracy. A working codec is not evidence of a general meaning-preserving language. All output remains a model prediction, including accepted grammar input.

General vocabulary, arbitrary numbers, compositional scope, automatic segmentation, open-ended semantic queries and robust paraphrase understanding remain unfinished. A subsequent study needs a separately frozen, broader dataset and relevant text baselines before any practical semantic-compression claim. This release preserves the failed v0 and v1 studies.

한국어: 실행 가능한 영어 → 벡터 패킷 → 의미 항목 예측 인터페이스를 구현했지만, 세 시드 전체의 사전 성공 관문은 실패했습니다. 정해진 문법에서도 새로운 표현에서 오류가 납니다. 범용 의미 보존과 실용적인 압축은 아직 미완성이며, 코드 테스트 통과와 모델 성능은 별개입니다.
