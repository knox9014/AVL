# AVL query revision 3: factorized learned operators

2026-10-07. Declared after two failed receiver pilots, before revision-3 execution.
This is a post-hoc engineering experiment on observed finite v2 data. It is not
a new independent confirmation, a universal AI language, or cross-model alignment.

## Motivation and claim boundary

The flat heads learned supported training claims but missed heldout supported
claims; existing field-decoder/rule answers were correct. Test whether explicit
factor access plus an abstract operator curriculum resolves this bounded failure.
Both architecture AND training curriculum change, so a gain cannot identify
their separate causal effects. Reuse the frozen v2 receiver's supervised factor
knowledge. This does not show a fresh receiver acquiring the vector codebook.

AVL remains an AI language project. These nine factors are an intermediate
auxiliary structure, not a permanent specification of all future messages.

## Inference access

Receive only actual deserialized 16-float AVL vectors and the previously declared
24 question identifiers/candidates. Frozen v2 decoding supplies nine hard class
decisions (ordinary classifier argmax), without gold fields, text or row IDs.
Metadata answers use that learned decoder.

For each device proposition, provide a learned operator six factors:
fact, certain, unconditional, subject matches query, predicate matches query,
positive polarity. The operator is MLP6->32->32->3 with GELU.

For each numeric question, provide another learned operator six features:
budget subject flag, three comparator indicators, amount-minus-candidate in
dollars, and its absolute value. Amount/comparator come from learned decisions;
candidate comes from the explicit query. This arithmetic feature design is
hand-specified. Neither inference operator executes the answer oracle,
conjunction branch, threshold comparison or frame lookup to choose an answer.

## Abstract supervision and budget

Proposition curriculum: all64 binary six-factor patterns. Labels are unknown
unless the first five factors are true; polarity selects entailed/contradicted.
Numeric curriculum: budget flag0/1, three comparators, integer deltas-101..101;
1,218 examples. Oracle labels follow inclusive at-most/at-least and exact
comparisons, or unknown for a nonbudget subject. Every abstract logical pattern
is supervised; do not claim generalization to unseen truth tables or operations.
This curriculum encodes prior semantic knowledge and is an additional resource.

Each operator has1,379parameters, total2,758 newly trained parameters. Use head
seeds2044/2055/2066, with numeric seeds offset100. Per operator:3,000 AdamW
steps, batch256, lr0.003, weight_decay0.00001, gradient clip1. Class-balanced
sampling uses inverse training-class support. Final parameters only, no
checkpoint selection. CPU2threads, total experiment hard budget600seconds.
Frozen sender54,897 and field receiver45,080 parameters remain unchanged.
Total inference parameters are102,735. No language-unit scaling is performed.

## Evaluation and unchanged success criterion

Use the same previously observed joint202 rows. For every family and every
sender seed44/55/66 require >=95% accuracy AND >=95% macro recall, unchanged
from the failed pilots. Report all class support/recall, zero-vector and
segment-shuffled-vector controls, and the train-majority query-only baseline.
Apply corruption before frozen field decoding. Compare previous pilot scores
as historical, different-supervision comparisons, not matched causal ablations.

Audit all5,376 admitted grammar surfaces separately, including sender training
examples. This is coverage, not new generalization evidence. Record abstract
curriculum training metrics separately from AVL evaluation. No v2 text/frame
examples train the new operators, but their abstract semantic patterns DO
overlap with evaluation meanings. Do not describe this as all test semantics
being unseen during all training.

Write raw reports, curriculum data, operator weights and source/checkpoint
hashes to an exclusive new output. Preserve a durable summary and link CI.
Report failures without gate changes or tuning against revision-3 outcomes.

A pass would establish a working learned-operator receiver for this bounded
language and supplied semantic curriculum. It would not establish arbitrary
prose understanding, learnability of a new codebook or universal AI language.
