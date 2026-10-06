# Optional local translation gateway

[English](MULTILINGUAL_GATEWAY.md) | [한국어](MULTILINGUAL_GATEWAY.ko.md)

This frontend translates an explicit source language to English before AVL,
and can translate a receiver's predicted canonical English to a target
language. The first backend is [M2M100](https://huggingface.co/facebook/m2m100_418M),
which declares 100 languages, including Korean. It does not cover all languages,
and declared support does not guarantee meaning preservation.

The translator is separate from the 99,977-parameter AVL model. Its downloaded
files occupy approximately 1.94 GB, and CPU inference needs additional memory.
The actual measured environment and parameter count are in
[illustrative results](MULTILINGUAL_RESULTS.md). Model weights retain their
upstream MIT terms and attribution and are not redistributed in this repository.

## Setup

Start with the normal AVL dependencies and a compatible PyTorch installation.
The optional Transformers backend requires PyTorch 2.6 or newer to load these
PyTorch binary weights. The recorded run used Python 3.14, Transformers 4.57.6,
SentencePiece 0.2.2 and PyTorch 2.13.0+xpu in CPU mode.

```bash
python -m pip install -r requirements.txt
python -m pip install --target .translation-deps --only-binary=:all: -r requirements-translation.txt
python -m antlab.download_translation --revision 55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636
```

Installation and explicit model download access public package/model servers.
Inference loads local files only, does not download automatically, and sends no
source text to a remote service. The local weight directory contains a revision,
byte-count and SHA-256 manifest. Both dependency and weight directories are
Git-ignored. Newer optional versions may change numerical translation results;
use the recorded versions to reproduce the measured environment.

## Run

```bash
python -m antlab.multilingual_demo --list-languages
python -m antlab.multilingual_demo --translation-only --source-lang ko --target-lang en --text "램프가 켜져 있다."
python -m antlab.multilingual_demo --source-lang en --target-lang ko --text "It is certain that the lamp is on."
python -m antlab.multilingual_probe
```

`--translation-only` accepts ordinary text independently of the AVL ontology.
The regular mode requires translated English to match the frozen finite AVL
grammar. It does not silently rewrite arbitrary English into a known frame.
Repeat `--text` for caller-delimited segments. Language identifiers are explicit
ISO codes from `--list-languages`; there is no automatic language detector.

| Status | Meaning |
|---|---|
| `encoded` | English input admitted and encoded; not independently verified truth |
| `out_of_scope` | Translation/input outside AVL grammar; original and translation retained |
| `unsupported_language` | This backend does not declare the requested language route |
| `translation_unavailable` | Missing/invalid local resources, empty result or exceeded limits |
| `model_prediction` | Receiver's predicted frame/text, not guaranteed source meaning |
| `target_translation_unavailable` | Receiver keeps canonical English when target translation fails |

Source originals and pivot translations stay in local sender records. The
receiver accepts only serialized vector bytes. Its frame is rendered and then
translated; the source and intended reference frame are not receiver inputs.
Sender and receiver must use matching AVL checkpoints. Current cross-play
results do not establish compatibility across independently trained models.

Default translation limits are 256 source tokens and 128 generated tokens,
with no silent source truncation. Over-limit inputs retain their source through
the failure path; automatic segmentation of long prose remains future work.
Only nonempty translations are reported as predictions, never as verified
translations. Unanticipated programming errors propagate for diagnosis.

## Python API

```python
from antlab.local_translation import M2M100Translator
from antlab.semantic_v2_demo import load_checkpoint
from antlab.translation_gateway import TranslationGateway

backend = M2M100Translator('.translation-models/m2m100-418M')
gateway = TranslationGateway(load_checkpoint('antlab/runs/semantic-v2-20261006'), backend)
sent = gateway.encode(['It is certain that the lamp is on.'], 'en')
if sent['packet'] is not None:
    received = gateway.decode(sent['packet'], 'ko')
```

Alternative local translators can implement `name`, `supports(source, target)`
and `translate(text, source, target)`. Expected backend failures must raise
`TranslationError`. This extension point does not itself add language coverage
or align another model's vectors with AVL.

See [research references and cross-play](AI_COMMUNICATION_REFERENCES.md) for the
distinction between emergent messages, semantic vectors and transport standards.
