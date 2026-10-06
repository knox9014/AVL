# AVL — AI Vector Language

[English](README.md) | [한국어](README.ko.md)

AVL explores learned vector communication that preserves useful meaning between AI units. Its long-term goal is to let shared neural units use communicated information across multiple tasks, with explicit checks for semantic retention and resource cost.

**Status: an early research prototype.** The current model has 98,928 shared parameters and 16-dimensional float32 messages. It can encode local English text, exchange vectors, and generate characters. The first YES/NO experiment did **not** learn to use remote facts: every input produced `YES`, with 50% accuracy. A useful general-purpose language, lossless compression, and intelligence gains from adding units have not been demonstrated.

## Included

- A shared local encoder, sender, message attention, state updater, and character decoder.
- Synchronous communication, blocked communication, and matched serial controls.
- Twelve focused tests, a reproducible fixed-budget smoke runner, and its checkpoint/source snapshots.
- A long-input payload probe with original synthetic texts, actual vectors, and raw payload bytes.
- [Semantic input design](docs/SEMANTIC_INPUT.md), [body design](docs/research/2026-10-05-english-connected-body.md), [smoke results](docs/research/2026-10-05-english-body-results.md), and [long-input measurements](docs/research/2026-10-06-long-vector-size.md).

`antlab` is the historical Python module name retained so published source hashes and checkpoint verification continue to work. AVL is the project name. The wider ACT experiments are outside this repository.

## Run

Use Python 3.10 or newer with PyTorch installed. The recorded experiment used PyTorch 2.13 on CPU with two threads. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s antlab/tests
python -m antlab.english_connected_smoke --verify antlab/runs/english-body-smoke-20261005
python -m antlab.english_connected_smoke --output antlab/runs/my-smoke
python -m antlab.english_vector_size_probe --output antlab/runs/my-size-probe.json
```

Output paths must be new; existing artifacts are not overwritten. Verification checks source/artifact hashes and independently regenerates ten evaluation conditions from the saved checkpoint. Different PyTorch versions or platforms may yield different floating-point results; the strict saved-artifact verifier targets the recorded implementation.

## What the measurements mean

| Raw English input | One vector payload | Representation size reduction |
|---:|---:|---:|
| 173 bytes | 64 bytes | 63.0% |
| 394 bytes | 64 bytes | 83.8% |
| 791 bytes | 64 bytes | 91.9% |
| 1,400 bytes | 64 bytes | 95.4% |

The vector size is fixed by architecture. These numbers are **not evidence that all original meaning survives**. On the long-input on/off pairs the model again always answered `YES`. Two units exchanging messages in both directions for two rounds use 256 logical payload bytes per example, excluding network overhead.

## Next research step

Train and evaluate meaning-preserving input representations before forcing stronger compression. Check facts, relations, requests, negation, conditions, and uncertainty with several questions about the same representation. Separate input-understanding failures from communication and output failures. Evaluate unseen combinations and task types; report payload and computation costs together with accuracy.

## Data and license

All included examples are synthetic. This export contains selected research files rather than workspace history, private conversations, credentials, or local user configuration. Model checkpoints contain tensors and experiment metadata.

Apache-2.0. See [LICENSE](LICENSE).
