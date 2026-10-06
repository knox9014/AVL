# Translation throughput and request-local reuse

2026-10-06. A bounded engineering improvement to the existing local translator:
batch same-language segments and reuse identical source translations within
one request. No new training, semantic canonicalization or vector alignment.

The comparison uses the same pinned M2M100 checkpoint, CPU, two threads,
three-beam deterministic decoding and fixed source fixture. There are 24 unique
examples across eight languages, then a 12-segment Korean request repeating
three examples four times. Each mode starts in a separate process. Model
loading and one common warm-up are excluded from inference timing.

| Measurement | Individual processing | Batch size 4 and request-local reuse |
|---|---|---|
| Inference time | 94.020 s, one final-code run | 38.261 s / 38.307 s, two runs |
| Model generation calls | 36 | 9 in each run |
| Nonempty output segments | 36 / 36 | 36 / 36 |
| Exact output differences from individual mode | Reference | 0 / 36 in each run |
| Peak process working set | 2,675,208,192 bytes | 2,732,228,608 / 2,731,601,920 bytes |
| Translations admitted by current AVL grammar | 0 / 36 | 0 / 36 |

Mean batch inference time is 38.284 s, an observed 2.456x throughput ratio.
Generation calls fell 75%; this is not a 75% reduction in all neural computation,
because each call now processes multiple unique examples. Mean peak working
set increased approximately 2.1%. Memory counters include loading and the
entire process; they do not isolate tensor/activation memory. Disk weight size
is unchanged. Larger batches trade throughput for memory.

Only one individual-mode and two batch-mode final-code runs were measured.
Other host activity is uncontrolled. The results do not establish statistical
significance, universal speed gains, or general translation accuracy. Exact
output equality covers this fixture only. All translated texts still remained
outside AVL's finite grammar, so no foreign semantic transmission gain is
claimed. Existing translation errors remain.

## Behavior and limits

- Deduplication is exact-string and local to an invocation. No persistent text
  cache is created; original order and result count remain unchanged.
- Sender and receiver gateway calls can use an optional `translate_many` method.
  Scalar-only backends continue to work. Metadata distinguishes input segments,
  unique/reused segments and backend API calls; API calls are not generation calls.
- Tokenization and generation are split into at most four unique segments by
  default. A single very long input still incurs tokenization before rejection;
  total request bytes and retained source records are not memory-bounded.
- Empty/over-limit/failed rows remain unavailable. A long row does not prevent
  other rows from translating. Output-limit checks ignore padding belonging to
  shorter rows. No silent source truncation or unsupported meaning repair.
- Explicit PyTorch memory exhaustion preserves unavailable rows; unexpected
  programming errors still propagate. No source text is sent remotely.

See [usage](MULTILINGUAL_GATEWAY.md) for `--batch-size`, API and failure statuses.
An English-only path still bypasses translation altogether.

## Reproduction and raw data

Use the optional local setup and pinned revision from the usage document.
Run sequentially in separate processes, with new output paths:

```bash
python -m antlab.translation_efficiency --mode serial --output antlab/runs/my-serial.json
python -m antlab.translation_efficiency --mode batch --output antlab/runs/my-batch.json
```

[Summary and raw file hashes](../antlab/runs/translation-efficiency-summary-20261006.json),
[individual run](../antlab/runs/translation-efficiency-serial-2-20261006.json),
[batch run 1](../antlab/runs/translation-efficiency-batch-1-20261006.json),
[batch run 2](../antlab/runs/translation-efficiency-batch-2-20261006.json).
Raw reports include input/output text, per-group times, generation counts,
model revision, dependency versions and source hashes without private paths.

한국어: 동일한 합성 입력 36개를 개별 처리하면 약 94초, 묶음 처리하면 평균
약 38초였습니다. 이번 시험에서는 번역 결과가 모두 같았고 생성 호출은
36회에서 9회로 감소했습니다. 최대 메모리는 약 2.1% 증가했습니다. 번역
정확도나 AVL 의미 전달 범위가 개선됐다는 결과는 아닙니다.
