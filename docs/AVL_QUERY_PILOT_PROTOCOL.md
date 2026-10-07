# AVL frozen-vector query pilot, revision 1

2026-10-07. Declared before the first pilot execution. This is a new bounded
receiver-learning pilot, not a revision of the frozen v2 study. AVL remains an
AI language research project; field-based labels are auxiliary evaluation tools.

## Hypothesis and access

A newly initialized receiver can learn to answer multiple semantic question
families from frozen AVL v2 messages. It gets only deserialized 16-float vectors,
a question identifier and (for numeric questions) one normalized candidate.
It does not receive text, gold fields, source IDs, checkpoint hidden states or
rule-derived answers at inference. Gold frames generate supervised labels only.
This is bounded semantic-query learning, not general reasoning, an emergent
universal language or independent-model compatibility.

## Frozen pilot choices

Use final v2 sender seeds 44/55/66 without updating them. Head initialization
seeds are 1044/1055/1066 respectively. Use existing v2 train rows (2,776) for
query learning and existing joint rows (202; 101 reserved frames) for evaluation.
No v2 joint frame or its queries enter head training. These data have been
observed in previous work and are reused diagnostics, not fresh independent
evidence. No tuning on the joint results within this revision.

Questions: kind, certainty, condition; eight positive device atoms
(lamp/heater/fan on/off, door open/closed); budget constraint membership for
candidates 0/9/10/11/19/20/21/49/50/51/99/100/101 USD.
All 24 questions apply to every segment. See semantic_queries.py for conservative
label semantics. A conditional is not discharged without world context.
Candidate membership interprets a described constraint and does not claim a
world state or permission to act.

Receiver: question embedding12x8; concatenate vector16, embedding8 and
candidate/100 feature1; MLP25->64->64->13 with GELU. Fixed per-question output
masks exclude unrelated answer categories. 13 answer labels are declared in
semantic_query_pilot.py. AdamW lr0.003, weight_decay0.01; 1,200 steps, batch256;
gradient norm clip1; two CPU threads; final head only. Sampling gives equal
total mass to each represented family/answer group in training. No test labels
in sampling. Three head trainings; hard 600-second pilot budget checked between
stages and training steps. Cache encoded byte-roundtripped vectors before head
training; report their wire cost separately from repeated query evaluation.

## Controls and reporting

Compare learned received-vector head, same head on zero vectors, same head on
a fixed nonzero row permutation, per-query training-majority answers, and
existing field decoder followed by rules. Gold-label rules are the annotation
oracle, not an independently learned method. Shuffling occurs at segment level
with seed-dependent nonzero circular offset. Whole-segment duplicates can
share meaning, so report the control without assuming it destroys all meaning.

Report support/confusion, accuracy, represented-class macro recall, and
per-class recall for each family: kind, certainty, condition, proposition,
constraint_allows. Missing classes have null recall, never perfect recall.
Record head parameters, final loss, training time, environment versions,
source/checkpoint/data SHA256, byte counts, and unchanged sender weights.
Save JSON plus head tensors as CI artifacts; no automatic main-branch writes.

A pilot gate requires >=95% accuracy AND >=95% represented-class macro recall
in every family for every seed. All three must pass. Even a pass is limited to
these known question families and reused finite-grammar data. A failure remains
a result; a modified hypothesis requires a new revision and separate output.
Do not present lower-performance controls as cost-matched independently trained
baselines. Full application, larger vocabulary, receiver sample efficiency and
multiple independently learned language spaces remain future work.
