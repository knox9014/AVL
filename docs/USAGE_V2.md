# AVL v2 usage

[English](USAGE_V2.md) | [한국어](USAGE_V2.ko.md)

V2 is a finite English semantic channel. Its 99,977-parameter model uses an
ordered word encoder and a vector-only receiver. Acceptance by the grammar
does not verify the model prediction. See [protocol](AVL_V2_PROTOCOL.md).

## Run

From the repository root with PyTorch installed:

```bash
python -m antlab.semantic_v2_demo --text "When it rains, that the door is not closed is now possible." --text "Please, now ensure that the budget limit is at most 20 USD."
python -m antlab.semantic_v2_run --output antlab/runs/semantic-v2-20261006 --verify
python -m unittest discover -s antlab/tests
```

Each repeated text argument is a caller-defined independent segment. The
sender retains original sources and admits only saved grammar templates.
The encoder uses a training-derived vocabulary of 41 words, retaining order;
it normalizes case/whitespace and ignores punctuation. The receiver takes
only the serialized vectors and predicts nine fields: kind, subject,
predicate, polarity, certainty, condition, amount, comparator and unit.

All output is marked `model_prediction`. Unsupported text, including a new
entity or 17 USD, retains its exact original with `unsupported` status and
produces no vector. Automatic segmentation, summarization and arbitrary
English understanding are not implemented. One condition governs one segment;
negation is retained without inferring its opposite state.

## Python interface

```python
from antlab.semantic_v2_demo import load_checkpoint, transmit_segments, receive_packet
model = load_checkpoint("antlab/runs/semantic-v2-20261006", seed=44)
sent = transmit_segments(model, ["It is certain that the lamp is on."])
if sent["packet"] is not None:
    print({"status": "model_prediction", "predictions": receive_packet(model, sent["packet"])})
```

Transmit only packet bytes; sources, admission indices and audit targets stay
local. Sender and receiver must use the same v2 checkpoint. The codec's AVL1
identifier describes float32 framing, not learned-model identity; it does not
detect checkpoint mismatch or authenticate a sender. Vectors are not
anonymization. This local prototype supplies no network reliability layer.

A vector uses 64 payload bytes and a packet adds 12 header bytes. Two
segments use 140 bytes, excluding network overhead. A short original can be
smaller than its vector. Meaning retention and compression are separate.

## Repeat the frozen study

```bash
python -m antlab.semantic_v2_run --output antlab/runs/my-new-v2-study
```

Use a new output folder. The study trains final checkpoints for seeds
44/55/66 with 1,800 optimizer steps per seed and training-only paraphrase
pairs. Default demo seed 44 was specified before evaluation. Verification
reloads checkpoints, regenerates data and replays all controls through actual
bytes; it does not repeat optimization. Exact floating-point replay targets
the recorded numerical environment. A passed finite benchmark does not
establish general English meaning preservation.
