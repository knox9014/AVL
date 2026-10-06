# AVL — AI Vector Language

[English](README.md) | [한국어](README.ko.md)

AVL researches learned vector communication that preserves meaning between AI units. The long-term goal is a useful AI communication language; this release provides a runnable, bounded experiment.

**Status: v1 research prototype.** A 98,511-parameter model encodes one controlled English segment into a 16-dimensional float32 vector and predicts nine meaning fields from that vector alone. Ordered multi-segment packets, unsupported-input handling, three trained checkpoints and replayable evaluations are included. **The preregistered study failed: unseen phrasing remains unreliable.** General English understanding and lossless semantic compression have not been demonstrated. See the [measured results](docs/AVL_V1_RESULTS.md).

## Run

Use Python 3.10 or newer with PyTorch installed. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m antlab.semantic_demo --text "When it rains, it is possible that the door is not closed." --text "Please make sure that the budget limit is at most 20 USD."
python -m unittest discover -s antlab/tests
python -m antlab.semantic_run --output antlab/runs/semantic-v1-20261006 --verify
```

The receiver predicts kind, subject, predicate, polarity, certainty, condition, amount, comparator and unit. It receives actual serialized vectors, without source text or answer labels. A vector carries 64 payload bytes; a packet adds a 12-byte header. Two segments use 140 bytes, excluding network overhead. This can exceed the original text size.

The demo uses seed 11, chosen before evaluation. Output is marked `model_prediction`, which can be wrong even for accepted text. Unsupported input stays unchanged locally with `unsupported` status. Acceptance means the text belongs to the finite grammar, not that its prediction is verified.

## Scope and documentation

- [English usage and Python API](docs/USAGE.md), [한국어 사용법](docs/USAGE.ko.md).
- [Frozen v1 protocol](docs/AVL_V1_PROTOCOL.md) and [v1 results](docs/AVL_V1_RESULTS.md).
- [Semantic input research direction](docs/SEMANTIC_INPUT.md).

Only the saved English templates are accepted, with lamp/heater/fan/door/budget subjects and amounts 10/20/50/100 USD. The caller supplies segment boundaries. Arbitrary prose, new entities or numbers, nested scope and automatic summarization remain unsupported. Vectors do not provide anonymization, authentication or network delivery.

To repeat training, use a new output folder:

```bash
python -m antlab.semantic_run --output antlab/runs/my-new-study
```

The fixed study uses three seeds, 1,500 steps each, final checkpoints and predeclared heldout gates. Verification regenerates data, checks source/artifact hashes and training metadata, and replays checkpoint inference through the byte codec. It does not retrain. Exact numerical replay targets the [recorded environment](antlab/runs/semantic-v1-environment-20261006.json).

## Earlier v0 experiment

The original 98,928-parameter connected character model always answered `YES` and scored 50% on its YES/NO smoke task. Its sources, checkpoint and failure reports remain intact: [body design](docs/research/2026-10-05-english-connected-body.md), [smoke results](docs/research/2026-10-05-english-body-results.md), [long-input measurements](docs/research/2026-10-06-long-vector-size.md).

The earlier 64-byte vectors were smaller than long English inputs, but did not establish meaning retention. Adding AI units has not been shown to improve general intelligence by this repository.

`antlab` is the historical Python module name retained for recorded hashes. AVL is the project name; wider ACT experiments belong outside this repository.

## Data and license

Examples are synthetic. Published checkpoints contain tensors and experiment metadata. Private conversations, credentials and local user configuration are excluded from this export.

Apache-2.0. See [LICENSE](LICENSE).
