# AVL named values: experiment 1

2026-10-07. Experimental extension, declared before execution. Preserve frozen
v2 models/data and previous query studies. This extends named instances and
numeric literals, not object taxonomies or unconstrained natural language.

## Design and disclosure

Use a hybrid AI message: learned relation vector plus explicit typed literals.
Names and exact unsigned64-bit integers are copied through a separately counted
channel. They are not decoded from the learned16-dimensional vector, and their
successful transport is not learned understanding or compression.

Controlled input syntax inserts 'named "identifier"' after the existing
'the lamp/heater/fan/door/budget'. Identifiers are ASCII letters/digits/underscore/
hyphen,1..64characters, first character alphanumeric. Exactly one named subject
is required. For budgets, one decimal amount0..2^64-1 is required.

An explicit lexical adapter removes the name and replaces the budget numeral
with20 before existing grammar admission/tokenization. This is designed
canonicalization, not a learned parser. It must preserve other text, including
negation, certainty, condition and comparator, and reject unsupported grammar.
Original unsupported text remains local and is never silently dropped.

Sender encodes canonical text with the existing frozen v2 encoder. Each message
carries one ordinary AVL1 vector packet plus an AVN1 outer header12bytes,
the ASCII identifier, and an optional little-endian unsigned64-bit integer.
Header <4sBBHI: magic AVN1, version1, flags(amount-present bit0), identifier
length, inner packet length. Inner length76 and one vector are required.
Reject unknown flags/versions, malformed identifiers, inconsistent lengths,
extra bytes and invalid nested vector packets. Same learned checkpoint is
required; no model identity or authentication is supplied by this header.

Receiver decodes the vector, validates the existing predicted frame, then
attaches the literal identifier and restores the exact integer for budget
frames. Budget canonical prediction must contain amount20; amount presence must
agree with the predicted subject type. These checks do not prove predictions
correct. Literal metadata is visible to the receiver and carries information;
it is not a hidden gold-label channel or a vector-only experiment.

## Reserved test literals and scope

No sender retraining occurs. The following values are fixed before audit:
7,17,37,125,1000,16777217,9007199254740993,18446744073709551615.
These are not the v2 amount vocabulary10/20/50/100. Test identifiers:
<subject>-east-7 and <subject>-west-12 for each of the five known subject types.
They name new instances of known types, not newly understood concepts.

Use all256 device frames and48 budget frames with canonical amount20,
templates0and10, two names per type, and eight budget amounts:2,560 sources.
Audit all three frozen sender/receiver seeds44/55/66 without training/tuning.
Reuse previously observed grammar and models; this is an engineering audit,
not fresh independent semantic generalization. Targets are evaluation only.

Report exact literal retention, full meaning-plus-literal accuracy, per-field
accuracy and raw source versus total outer/inner/literal/header bytes. Do not
claim compression by omitting the literal channel. Unsupported cases and
packet-corruption tests are separate from admitted-source accuracy.
All three seeds require100% literal retention and >=95% full meaning accuracy.

Numeric semantic questions may reuse the revision-3 learned numeric operator.
Compute integer difference in Python BEFORE conversion to float32. Accept only
delta[-101,101], the declared operator curriculum range. Outside this range
return explicit unsupported; do not extrapolate silently. Candidates must also
be unsigned64-bit integers. Test candidate amount-1/amount/amount+1 when valid,
all three comparators, and report accuracy/macro recall with class supports.
This tests new literal values under a known arithmetic feature/curriculum,
not new mathematical operations. A deterministic comparator is the annotation
oracle, never the learned operator's inference decision.

## Execution and artifact integrity

Local process startup is unavailable; execute in bounded CPU CI. Train the
previous revision-3 operators with their unchanged protocol, then load their
saved weights safely with weights_only=True for this audit. Verify operator
hashes against its report and sender hashes against the frozen manifest.
Hard audit budget600seconds, CPU2threads, exclusive new output directory.
Save source/checkpoint/operator/test-definition hashes, counts, bytes and
per-seed results. Preserve raw report as an artifact and durable summary in Git.

A pass establishes an exact-literal interface around a bounded learned relation
language. Next studies must separately test learned input extraction, new
concepts, broader heldout tasks and codebook compatibility. Existing default
float32 inference remains unchanged.
