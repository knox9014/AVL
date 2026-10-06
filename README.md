# AVL — AI Vector Language

[English](README.md) | [한국어](README.ko.md)

AVL researches learned vector communication that preserves meaning between AI units. The long-term goal is a useful AI communication language; this release provides a measured, bounded implementation.

**Status: v2 research prototype.** A 99,977-parameter model encodes controlled English into a 16-dimensional float32 vector. A receiver predicts nine meaning fields from that vector alone, and a fixed renderer can express the predicted frame in canonical English. All three final checkpoints passed the frozen heldout gates, including unseen wording and meaning combinations together. **This is finite-grammar success, not general English understanding.** See [v2 results](docs/AVL_V2_RESULTS.md).

## Run

Use Python 3.10 or newer with PyTorch installed. Run from the repository root:

~~~bash
python -m pip install -r requirements.txt
python -m antlab.semantic_v2_demo --text "When it rains, that the door is not closed is now possible." --text "Please, now ensure that the budget limit is at most 20 USD."
python -m unittest discover -s antlab/tests
python -m antlab.semantic_v2_run --output antlab/runs/semantic-v2-20261006 --verify
~~~

The receiver predicts kind, subject, predicate, polarity, certainty, condition, amount, comparator and unit. It receives actual serialized vectors, without source text or answer labels. A vector carries 64 payload bytes; a packet adds a 12-byte header. Two segments use 140 bytes, excluding network overhead. A short original can be smaller.

The demo uses seed 44, chosen before evaluation. Output is marked model_prediction. Unsupported text stays unchanged locally with unsupported status. Acceptance means membership in the finite grammar, not that a prediction has been independently verified.

## Scope and documentation

- [Optional local multilingual translation gateway](docs/MULTILINGUAL_GATEWAY.md), [measured limitations](docs/MULTILINGUAL_RESULTS.md), and [AI communication references](docs/AI_COMMUNICATION_REFERENCES.md). The first backend declares 100 languages; all 24 foreign probe translations remained outside AVL's grammar. Translation is separately resourced and is not general multilingual AVL understanding.
- [English usage and Python API](docs/USAGE_V2.md), [한국어 사용법](docs/USAGE_V2.ko.md).
- [Frozen v2 protocol](docs/AVL_V2_PROTOCOL.md), [v2 results](docs/AVL_V2_RESULTS.md).
- [Post-hoc grammar coverage](antlab/runs/semantic-v2-coverage-20261006.json), [actual vector and English reconstruction example](antlab/runs/semantic-v2-demo-20261006.json).
- [Semantic input research direction](docs/SEMANTIC_INPUT.md).

Only saved English templates are accepted, with lamp/heater/fan/door/budget subjects and amounts 10/20/50/100 USD. The encoder's 41-word vocabulary comes from training text only. The caller supplies segment boundaries. Arbitrary prose, new entities or numbers, nested scope and automatic summarization remain unsupported.

Canonical rendering uses predicted fields, retains negation/conditions/uncertainty and rejects contradictory combinations. It does not restore original wording or verify correctness. Sender and receiver must use the same checkpoint: the codec version is not model identity. Vectors do not provide anonymization, authentication or network delivery.

A [post-hoc cross-play test](docs/AI_COMMUNICATION_REFERENCES.md#avl-cross-play-diagnostic) measured 100% exact recovery for each matching checkpoint but only 0.1650% averaged across six different-checkpoint pairs on 202 joint heldout inputs. Current independently trained AVL instances do not share a common vector language.

To repeat training, use a new output folder:

~~~bash
python -m antlab.semantic_v2_run --output antlab/runs/my-new-v2-study
~~~

The study uses three seeds, 1,800 steps each, training-only paraphrase pairs, final checkpoints and predeclared heldout gates. Verification regenerates data, checks source/artifact hashes and training metadata, and replays checkpoint inference through actual bytes. It does not retrain. Exact numerical replay targets the [recorded environment](antlab/runs/semantic-v2-environment-20261006.json).

## Historical failures

V0's 98,928-parameter connected character model always answered YES and scored 50% on its YES/NO smoke task: [smoke results](docs/research/2026-10-05-english-body-results.md), [long-input measurements](docs/research/2026-10-06-long-vector-size.md).

V1's 98,511-parameter model recovered reserved semantic combinations but failed its wording gates: [v1 results](docs/AVL_V1_RESULTS.md), [v1 usage](docs/USAGE.md). Its sources and artifacts remain unchanged. V2 changes architecture, data and objective together; scores across versions are not a controlled causal comparison.

Smaller representations, efficient practical compression and stronger general intelligence from adding AI units remain unproven. The finite ontology could be encoded much more compactly if its frame were already known.

The historical Python module name antlab is retained for recorded hashes. AVL is the project name; wider ACT experiments belong outside this repository.

## Data and license

Examples are synthetic. Published checkpoints contain tensors and experiment metadata. Private conversations, credentials and local user configuration are excluded from this export.

Apache-2.0. See [LICENSE](LICENSE).
