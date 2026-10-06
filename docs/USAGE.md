# AVL v1 usage

[English](USAGE.md) | [한국어](USAGE.ko.md)

AVL v1 is a controlled English semantic transport interface, not a general
natural-language parser. See [protocol](AVL_V1_PROTOCOL.md) for exact scope.

## Demo

Run from the repository root with PyTorch installed:

```bash
python -m antlab.semantic_demo --text "When it rains, it is possible that the door is not closed." --text "Please make sure that the budget limit is at most 20 USD."
```

Each `--text` is one caller-defined segment. Automatic splitting or
summarization of arbitrary prose is not performed. The sender keeps originals
locally and performs boolean grammar admission, which supplies no meaning
labels. Case/whitespace normalization matches training.

The receiver takes only packet bytes and predicts nine fields: kind, subject,
predicate, polarity, certainty, condition, amount, comparator and unit.
Original text, gold labels and admission indices never reach the receiver.
Two segments use128 payload bytes plus12 header bytes:140 total.
Unsupported text retains its original and `unsupported` status locally;
no fabricated semantic vector is produced. Vectors are not anonymization.

## Python API

```python
from antlab.semantic_demo import load_checkpoint, transmit_segments, receive_packet
model = load_checkpoint("antlab/runs/semantic-v1-20261006", seed=11)
sent = transmit_segments(model, ["It is certain that the lamp is on."])
if sent["packet"] is not None:
    print({"status": "model_prediction", "predictions": receive_packet(model, sent["packet"])})
```

Transmit only the packet; sender sources stay local. Authentication and
network failure handling are not provided by this local codec.

## Study and replay

```bash
python -m unittest discover -s antlab/tests
python -m antlab.semantic_run --output antlab/runs/semantic-v1-20261006 --verify
python -m antlab.semantic_run --output antlab/runs/my-new-study
```

New output folders are required. Full training has three seeds and1,500steps
per seed. Verification reloads checkpoints and replays actual byte transport,
data regeneration, source/artifact hashes and training metadata; no optimizer
steps are repeated. The demo seed11 was chosen before evaluation, not selected
from test scores. Exact replay targets the recorded numerical environment.

## Limits

Devices are lamp/heater/fan/door; budgets use10/20/50/100 USD. Examples follow
the frozen templates, e.g. `It is certain that the lamp is on.` or
`Consider making sure that the budget limit is exactly 50 USD.` New entities,
17 USD, nested conditions and unsupported paraphrases are not accepted.
Negative polarity negates the stated predicate without inferring its opposite;
possible and certain remain separate. One condition governs one segment.

A vector may exceed a short original's size. Meaning retention and compression
efficiency are separate measurements. Previous v0 failure artifacts remain.
The v1 preregistered study failed its gates; accepted grammar inputs can still
be decoded incorrectly, especially unseen phrasing. See [results](AVL_V1_RESULTS.md).
