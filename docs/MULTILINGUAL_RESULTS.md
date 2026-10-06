# Local multilingual gateway: illustrative results

2026-10-06. This is a post-hoc engineering probe, not a multilingual benchmark.
The original gateway source snapshot for this probe is commit
`767016a9a9182101122e720303fc182659702280`; later batching optimizations have
separate measurements. Historical recorded source hashes are preserved.
Inputs are three manually authored synthetic parallel examples in each of eight
foreign languages: Korean, Japanese, Chinese, Spanish, French, German, Russian
and Arabic. They cover a lamp fact, a conditional negative door fact, and a
20 USD budget request. Native speakers have not audited the complete fixture.

The frozen AVL v2 seed-44 checkpoint is unchanged. The reference frames are
used only in the audit comparison; they are never supplied to translation,
the encoder, or the packet-only receiver.

| Measurement | Observed result |
|---|---|
| Foreign inputs producing nonempty English translations | 24 / 24 |
| Foreign translations accepted by the frozen AVL grammar | 0 / 24 |
| Foreign inputs retained with `out_of_scope` | 24 / 24 |
| Foreign semantic accuracy conditional on AVL admission | Not measurable: no admitted inputs |
| English controls accepted and recovered exactly | 3 / 3 |
| Reverse English-to-Korean/Japanese/Spanish outputs | 9 / 9 nonempty predictions; semantic quality not formally scored |
| Latest probe runtime | 95.593 seconds, CPU, two threads; includes model loading and all probe calls |

Generating a translation is not evidence that it preserves the source meaning.
For example, the Japanese lamp example became “I am sure that the lamp is
there.” and the Arabic example became “The lamp is clear.” These outputs fail
to express the intended `on` fact. Many other translations express plausible
English outside AVL's controlled grammar, such as “When it rains, the door may
not be closed.” This probe does not diagnose all rejected translations as
wrong. It diagnoses both a finite-grammar coverage gap and observed translation
errors. No normalization rules were added after seeing these outputs.

The first optional backend is the publicly named M2M100 418M checkpoint at
revision `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`. Its loaded model contains
483,905,536 unique parameter elements as counted by PyTorch; the public model
name is not our measurement of parameter count. Downloaded required files total
1,941,931,012 bytes, separately from the 99,977-parameter AVL model. The actual
tokenizer declares the expected set of 100 language codes. This validates its
declared routes, not quality for 100 languages or support for every language.

The machine ran PyTorch 2.13.0+xpu in CPU mode, Transformers 4.57.6 and
SentencePiece 0.2.2. Runtime inference uses local files only and sends no input
text to a remote translation service. The optional backend and model weights
are not committed to this repository.

An earlier engineering run took 38.757 seconds with the same outcomes; the
final provenance-recorded run took 95.593 seconds. These wall-clock measurements
include loading and may depend on concurrent host activity. They are not a
controlled latency comparison or a guarantee of translation speed.

Raw inputs, translations, statuses, comparisons, file hashes and runtime:
[gateway probe](../antlab/runs/multilingual-gateway-20261006.json).
The earlier [direct-input probe](../antlab/runs/semantic-v2-multilingual-probe-20261006.json)
showed rejection before translation was added. Reproduce the new engineering
probe with `python -m antlab.multilingual_probe` after optional setup described
in [gateway usage](MULTILINGUAL_GATEWAY.md).

The local translator and fallback interface work, but this release does not
establish successful foreign-language-to-AVL meaning transmission. Future
work needs a broader semantic encoder or a separately validated adapter,
alongside independent tests of translation meaning. SONAR-style multilingual
representation alignment is a research reference, not implemented here.

한국어: 8개 언어 24문장 모두 번역 결과를 생성했지만 현재 AVL 문법에는 하나도
들어맞지 않아 원문과 번역문을 보존했습니다. 번역 일부에는 의미 오류도
관찰됐습니다. 번역 계층과 실패 처리 구현은 검증했지만 다국어 AVL 의미 전달의
성공이나 모든 언어 지원을 증명한 결과는 아닙니다.
