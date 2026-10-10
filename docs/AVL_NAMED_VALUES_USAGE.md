# Experimental named values

AVL remains a learned AI-language experiment. This optional interface combines
a learned relation vector with explicit exact identifiers and integers. It does
not teach new subject categories, invent meanings for names, or remove the need
for matching model checkpoints.

## Run

With the repository's CPU PyTorch runtime installed:

```sh
python -m antlab.semantic_named_values --seed 44 --text 'At night, it is possible that the lamp named "lamp-east-7" is not on.'
python -m antlab.semantic_named_values --seed 44 --text 'It is certain that the budget named "budget-west-12" limit is at most 9007199254740993 USD.'
```

The first output should retain subject role lamp, identifier lamp-east-7,
negative polarity, possible certainty and night condition. The second should
retain the exact decimal integer, without float rounding. Output remains a model
prediction. Original text is retained locally; unsupported grammar returns an
explicit unsupported record.

Only the existing lamp/heater/fan/door/budget roles and finite v2 sentence
templates are supported. Put exactly one `named "identifier"` immediately after
the subject role. Identifiers are ASCII letters, digits, underscore or hyphen,
1..64 characters, beginning with a letter or digit. Budget values are decimal
integers from 0 through 18446744073709551615.

## Programmatic interface

`send_named_segments(model, texts)` returns local sources, supported indices,
unsupported records and a list of byte packets. Send the packets to
`receive_named_packets(matching_model, packets)`. The receiver receives the
learned vector and explicit typed literals; it never receives the original
sentence. A decoded role/literal mismatch raises ValueError rather than returning
a silently repaired meaning.

`query_named_amount(numeric_operator, meaning, candidate)` accepts the revision3
numeric operator and uint64 candidates. It computes the exact integer difference
before creating float32 features. Distances beyond 101 return
unsupported_numeric_distance; this interface does not claim arithmetic
generalization beyond the declared operator curriculum. The answer concerns the
numeric constraint; it does not assert that a conditional/uncertain statement is
a current world fact.

The AVN1 outer header adds12 bytes, followed by the identifier, optionally8 integer
bytes, and one76-byte AVL1 packet. Count all these bytes when comparing formats.
No compression advantage is assumed. There is no model negotiation,
authentication, multi-subject binding or independent-model alignment here.

## Reproduce the fixed audit

```sh
python -m antlab.semantic_query_operators --output query-operators-output
python -m antlab.semantic_named_values_audit --operators query-operators-output --output named-values-output
```

Output directories must not exist. The audit checks frozen checkpoint and
operator hashes, evaluates the preregistered2560 sentences under seeds44/55/66,
and writes JSON reports and numeric query records.
See [the protocol](AVL_NAMED_VALUES_PROTOCOL.md) for fixed data and success gates.
