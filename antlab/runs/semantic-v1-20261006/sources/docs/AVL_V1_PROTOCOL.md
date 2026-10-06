# AVL v1: controlled English semantic transport

2026-10-06. Frozen before training. This is a bounded semantic-transport
prototype, not completion of a general language or of ACT. No ACT scaling
experiments belong in this repository. Existing v0 failures remain intact.

## Meaning and scope

One independently supplied English segment describes one proposition or
request. Supported kinds are fact/request; subjects lamp/heater/fan/door/budget;
predicates on/off/open/closed/limit; polarity positive/negative; certainty
certain/possible; condition none/rain/cold/night; amount none/10/20/50/100;
comparator none/at-most/at-least/exact; unit none/USD. The field order and
class order below are the wire model's fixed interpretation vocabulary.

```
kind: fact, request
subject: lamp, heater, fan, door, budget
predicate: on, off, open, closed, limit
polarity: positive, negative
certainty: certain, possible
condition: none, rain, cold, night
amount: none, 10, 20, 50, 100
comparator: none, at-most, at-least, exact
unit: none, USD
```

For devices, amounts/comparators/units are none. Door predicates are open/closed;
other devices use on/off. Budget segments have predicate limit, positive
polarity, numeric amount/comparator, and USD. A condition governs the entire
single segment. Negative polarity negates its stated predicate; it does not
silently infer the opposite state. Possible means possible, not confirmed.
Nested scope, arbitrary numbers/entities, multi-clause inference and ambiguous
negated modality are unsupported. Segment boundaries are supplied by the caller;
automatic semantic segmentation is not implemented. Multiple segments retain
order and separate vectors; no claim of automatic general understanding.

## Access and transport

The encoder sees only ASCII text, consistently lowercased and whitespace
normalized in both training and public inference. Originals remain unchanged
locally. The receiver sees only a serialized
float32 vector and an integer field selector, never text, encoder state,
answer labels, source IDs or evidence. The same vector must answer all nine
field queries. A versioned binary codec has a 12-byte header and 64-byte
payload per segment. Raw sources are retained locally for audit/fallback;
they are not given to the receiver. Unsupported text is kept unchanged with
an explicit unsupported status. A grammar admission check must not return
semantic labels to inference; it is not the trained encoder.

## Architecture and training

Separate new modules from hashed v0 sources. ASCII embedding128x48,
GRU48→64, sender Linear64→16 plus LayerNorm16; receiver field embedding9x16,
MLP32→243→243→8 with GELU. Invalid output classes are masked by field
cardinality using negative infinity, while valid logits must remain finite.
Approximately0.1M shared parameters; exact count is tested.
Cross-entropy supervises all fields from the same transmitted representation;
there is no answer-text teacher-forcing shortcut. Training transmits through
a differentiable numerical vector; evaluation uses actual codec bytes.

Seeds11/22/33, 1,500steps per seed, batch128, AdamW learning rate0.003,
weight decay0.01, gradient clip1, CPU2threads, hard30minute total budget.
Use the final checkpoint, not selection on heldout results. No additional
tuning after observing heldout results; a changed hypothesis is a new study.

## Data and gates

Enumerate allowed semantic combinations before text rendering. Deterministically
reserve one fifth of frames for combination test. Use disjoint surface templates
for training, validation and phrasing test. Device/budget templates and scope
rules are saved and hashed before training. Datasets are synthetic, public,
deduplicated by normalized full input text, and regenerated in verification.

Report train/validation/combination/phrasing exact-frame and per-field accuracy,
macro class accuracy, confusion matrices, source byte sizes and actual wire sizes.
Controls: zero vectors, vectors shuffled among examples while keeping selectors,
query-only majority predictions, and gold-frame task oracle. All three seeds
must reach at least95% exact-frame accuracy and95% polarity/certainty/condition
accuracy on each heldout split to pass. Failure is recorded, not relabeled success.

An oracle confirms task/label correctness, not model ability. Zero/shuffle and
majority controls show whether useful information is in the vector; their low
accuracy alone does not establish generality. A raw-text classifier can be a
future separately budgeted baseline; do not claim superiority over one here.

## Completion for this release

Working encode/transmit/decode interfaces, multi-segment ordered packets,
support/fallback handling, tests, frozen study and replayable artifacts,
English/Korean usage docs and a public repository update. Semantic retention
is considered demonstrated only for heldout parts that actually pass their
gates. General English meaning preservation remains a research goal even if
the controlled grammar passes.
