# AVL v2: word-order semantic transport and paraphrase agreement

2026-10-06. New bounded study, frozen before v2 full training. The observed
v1 failures motivated this architecture and broader surface grammar; v2 is
not an independent replication of an unseen task. Existing v1 sources,
checkpoints, reports and failure results remain unchanged. No v2 heldout
result may trigger tuning within this study. A revised hypothesis requires
another separately declared study. This remains a finite synthetic semantic
transport experiment, not general English understanding or completion of ACT.

## Meaning and access

Each caller-supplied segment contains one fact or request. The exact nine
field vocabularies and device/budget restrictions remain those of v1:

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

Device amount/comparator/unit are none; door uses open/closed and other
devices on/off. Budget uses limit, positive polarity, numeric amount,
numeric comparator and USD. Negation applies to the stated proposition;
possible does not mean confirmed. A prefix or suffix condition governs the
entire single segment. Nested conditions, arbitrary entities/numbers and
multi-clause inference are unsupported. Segment boundaries are supplied;
multiple segments retain order. Unsupported input keeps its original text
with an explicit unsupported status. Grammar admission is boolean only,
never an encoder output or source of semantic labels at inference.

The encoder sees ASCII text lowercased with whitespace collapsed. Lexical
tokens use `[a-z]+|[0-9]+`, preserving word order and ignoring punctuation.
The 41-word vocabulary is constructed from training text only. Empty,
non-ASCII and unknown-word inputs are rejected by tokenization. Original
sources remain unchanged locally for audit/fallback. A receiver sees only
16 finite float32 numbers and field selectors; it never receives text,
encoder hidden state, gold labels, frame IDs or evidence. The existing
versioned codec carries a 12-byte packet header and 64 bytes per segment.
Sender and receiver must use the same learned checkpoint. The existing AVL1
packet format version identifies byte structure, not the identity of learned
weights. Mixing checkpoints and authenticated network transport are unsupported.

## Architecture and objective

Word embedding43x64 includes two reserved indices. A bidirectional GRU
with input64 and hidden64 per direction produces ordered 128-dimensional
states. A Linear128→1 softmax attention pool and a padding-masked mean pool
are averaged. Linear128→16 plus LayerNorm16 sends the shared vector.
The receiver uses field embedding9x16 and MLP32→192→192→8 with GELU.
Invalid classes are negative infinity; all valid scores must remain finite.
Exactly99,977 trainable parameters: sender54,897 and receiver45,080.
No separate raw-text classifier is trained within this budget.

For each batch, sample128 anchors uniformly from training rows. For each
anchor independently sample uniformly a different normalized surface
expressing its same training frame. Partners never come from validation,
combination, phrasing or joint data. Each view answers all nine field queries;
the cross-entropy is averaged across both views and all fields. Add
0.05 times MSE between each pair's L2-normalized 16-dimensional vectors.
The total objective is CE +0.05*MSE. No labels enter the receiver; labels
only define supervised targets and the training-only paraphrase pairing.

Seeds44/55/66, 1,800 optimizer steps per seed, batch128 paired examples
(256 encoded views), AdamW learning rate0.003, weight decay0.01, gradient
clip1, CPU2threads and a single hard1,800second total run budget cover all
seeds and evaluation. Use final checkpoints without heldout selection.
Loss/CE/consistency/gradient logs are saved at step1, every100 steps and the
final step, without duplicate log steps. Smoke tests use explicitly marked
smoke configurations and cannot constitute study success.

## Frozen data and controls

Enumerate448 allowed frames before rendering. Hash the canonical frame with
the `avl-v2:` prefix and reserve hash modulo5 equal0 for the combination
test:101 frames. The remaining347 frames use training templates0..7,
validation templates8/9 and phrasing templates10/11. Combination-reserved
frames use templates0/1 for the combination test and templates10/11 for
the joint test. Joint therefore withholds both meaning combinations and
surface forms simultaneously. It was added before any v2 performance was
observed and supersedes the preliminary four-split proposal.
Prefix/suffix condition placement alternates
by template parity. Template families cover fact, certain request and
possible request; complete templates are in the hashed data source.
Training has2,776 rows, validation694, combination202, phrasing694 and joint202.
Normalized full input strings are deduplicated across all splits. Every
heldout word appears in training. Templates, partitions and vocabulary are
saved and hashed before any optimization, and data regeneration is verified.

Report exact-frame accuracy, each field's accuracy, per-class support and
confusion, macro class accuracy over classes represented in the gold split,
raw ASCII source bytes, packet headers and actual wire payload sizes.
Primary evaluation uses actual encoder→codec bytes→receiver transport.
The zero-vector control preserves field selectors. The shuffled control
uses one nonzero circular permutation over the entire split, with offset
`seed % (split_size - 1) + 1`; no source row remains at its original index.
Permutation occurs before batched decoding, and its indices are saved.
Same-frame duplicates can still share meaning, so shuffled accuracy does
not by itself measure generality. Query-only predictions use each field's
training-majority class (lowest index wins ties). The gold-frame task oracle
returns saved gold labels by construction; its perfect score checks the metric
path, not independent semantic annotation correctness or model ability.

A direct numerical-vector roundtrip metric uses this same trained model's
encoder vectors without the codec. It only checks float32 serialization
equality and agreement with byte-transport predictions. It is not a separately
trained raw-text baseline and establishes no superiority over one.

Each seed must reach at least95% exact-frame accuracy and95% accuracy for
polarity, certainty and condition on every heldout split, including joint.
All three seeds
must pass for `study_passed=true`; a smoke or failed gate never passes.
Results are reported whether they pass or fail. Any demonstrated retention
is limited to the specific heldout portions and finite grammar tested.

## Artifacts and replay

Save immutable source snapshots and hashes, protocol, configuration, full
JSON-native data, final checkpoints, per-seed controls/metrics, timing and a
file-hash manifest. Verification requires exact frozen configuration,
regenerated data, matching sources and checkpoint metadata. It reconstructs
seeded initial weights without retraining to check reported parameter delta,
checks finite nonnegative loss components and their total relationship, and
replays all controls, byte totals and gate decisions for every saved seed.
Rehashed tampered metrics or training metadata are rejected by these semantic
checks; modified checkpoints/configuration are rejected by file/metadata
checks. The manifest provides local audit consistency, not cryptographic
authentication against an attacker able to replace every artifact and source.
