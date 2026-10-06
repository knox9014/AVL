# Experimental int8 wire format

2026-10-06. Post-hoc codec experiment; frozen model, data and primary gates
are unchanged. Each width-16 vector uses one positive float32 scale plus
16 signed int8 values in -127..127. The 12-byte header identifies `AVQ1`,
version 1, dtype 2, width 16 and segment count. Payload order is preserved.

| Size | Frozen float32 | Experimental int8 |
|---|---|---|
| Vector payload | 64 bytes | 20 bytes |
| One-vector packet | 76 bytes | 32 bytes |
| Two-vector packet | 140 bytes | 52 bytes |
| 5,376 examples in batches of 128 | 344,568 bytes | 108,024 bytes |

Payload reduction is 68.75%; batched packet reduction is about 68.65%.
This excludes network overhead. A short source string or a known finite
semantic frame may still have a much smaller representation.

For each frozen seed (44, 55, 66), nine-field exact recovery was 100% through
both codecs on the joint heldout set (202), all admitted grammar surfaces
(5,376), and previously unmeasured surfaces (808). No frame predictions
changed. These sets overlap; all grammar includes training examples. There
is no new preregistered gate, no additional training, and no general-language
or cross-model compatibility claim. The receiver sees reconstructed vectors
from actual bytes and a fixed field-vocabulary mask, without texts or labels.

The codec is lossy: maximum observed absolute vector error across all grammar
was 0.01393 / 0.01491 / 0.01491 for the three seeds. Unchanged finite-frame
predictions do not imply exact vector recovery or preservation of arbitrary
information. Model weights and translation resource costs are unchanged.

Packing plus unpacking on all 5,376 inputs took approximately 0.015–0.026 s
for float32 and 0.045–0.048 s for int8. These are small, noisy CPU timings,
not a controlled end-to-end network benchmark. Int8 conversion adds work;
the measured benefit is fewer transmitted bytes. Choose it explicitly for
experimentation; the existing float32 pipeline remains the default.

## Run and inspect

```bash
python -m antlab.semantic_int8_demo --text "It is certain that the lamp is on." --text "Please make sure that the budget limit is at most 20 USD."
python -m antlab.semantic_int8_audit --output antlab/runs/my-int8-audit.json
```

The demo supports the frozen controlled English grammar and same-checkpoint
communication. It preserves unsupported sources locally. `AVQ1` is a separate
format and the frozen `AVL1` decoder rejects it. Sender and receiver must
explicitly agree on the format and matching checkpoint; the header does not
authenticate or identify model weights. No automatic gateway format switch
or network delivery has been added.

[Raw comparison, metrics and hashes](../antlab/runs/semantic-int8-audit-20261006.json).
The standalone codec rejects malformed headers, invalid scales, -128,
overflow, nonfinite inputs and incorrect packet lengths.

한국어: 전송 벡터에 int8 양자화를 적용해 payload를 64바이트에서 20바이트로
줄였습니다. 같은 체크포인트끼리 제한된 문법의 5,376개 입력을 비교했을 때
세 학습 실행 모두 의미 복원 100%였지만 벡터 값은 근사됩니다. 일반 문장,
독립 모델 사이의 의미 보존을 보장하지 않습니다. 변환 연산 시간은 늘었고
전송 바이트 수가 줄었습니다. 기존 float32 기본 경로는 유지합니다.
