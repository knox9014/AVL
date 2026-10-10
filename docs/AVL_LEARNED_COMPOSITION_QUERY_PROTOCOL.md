# AVL learned three-entity query receiver pilot1
Preregistered2026-10-08 before receiver implementation or final evaluation.

Question: does a new neural receiver answer directed relation questions from
two frozen learned vectors, explicit graph endpoints and query indices on
withheld semantic structures? Compare a frozen primitive decoder plus explicit
graph/transitivity rules, a separately trained no-vector receiver, zero vectors,
shuffled vectors and removed topology/query. Preserve failures; no retuning.

Reuse unchanged binding seeds44/55/66; reproduce primitive training in CI,
safe-load/freeze, verify source/data hashes and record newly reproduced weight
hashes. Receiver seed144/155/166 respectively. New receiver: raw50features =
32 vector coordinates +12 one-hot endpoint coordinates +6 one-hot query
coordinates. Ordered vectors/edges and query source/target are as transmitted.
MLP50->64 GELU ->64 GELU ->5 logits (7749parameters); no frozen relation
logits or oracle labels are supplied as neural features. Structural scaffolding
and four primitive classes are already supervised and remain explicit.

Training:3000steps, batch128, AdamW lr0.003 weight_decay0.0001, clip norm1.
Each batch independently samples output class uniformly, then uniformly from
that class's training examples. Class labels are used only for training and
audit scoring. Train a second identical receiver on the same batches with the
first32 features zero; separate optimizer, same settings. Fixed steps, no
validation selection, scheduler, early stop or supplementary curriculum.
Neural training is on a tiny finite grammar; no universal language claim.

Use the same synthetic triple generator, all4 orientations,6 table permutations
and six directed queries per base scene:AB,BA,BC,CB,AC,CA.
Train template0 on12 tuples; withhold(0,0),(1,1),(0,2),(3,1).
Holdout is closed under chain reversal (r1,r2)->(inverse(r2),inverse(r1)).
Thus heldout path questions include left, right AND undetermined, unlike the
prior audit. Above/below positive transitivity is observed during training;
left/right are observed on direct queries but not positive two-hop training.

Create query-rooted semantic signatures: map queried source->0,target->1,
remaining node->2; orient each edge toward ascending anonymous indices
(invert its relation when necessary), sort edges. This audit signature includes
the full labeled graph, ignores names/table/clause order and equivalent inverse
wording. Assert train/joint signature sets disjoint and retain support/counts.
This is a semantic split check, not a feature or target provided at inference.

Partitions: train1152scenes/6912queries; joint384/2304 template3;
phrasing1152/6912 template3; new_names1536/9216 template3(all16tuples);
new_joint384/2304 template3(withheld tuples only, subset of new_names).
Known/new triples unchanged. New_names includes seen structures and is reported
separately from new_joint. Train overlap is expected in phrasing and seen
new-name structures; joint/new_joint overlap must be zero.

Query source/target slots are part of a real ACQ1 wrapper:
<4sI magic,AVC1_length> (8bytes), unchanged AVC1 bytes, two uint8 indices.
Exact framing, distinct valid query slots and finite embedded vectors required.
181bytes/query with five-character fixture names. No implicit model identity,
default inference change or network deployment. Symbolic gold baseline uses
the same wrapper and33byte explicit symbolic graph =43bytes.
Text reference is source UTF8 + identical name table + the same10query-wrapper
bytes; model distribution/network framing excluded. Repeating graph bytes per
query is measured explicitly; no caching/network speed benefit claimed.

Evaluate full5-class support/recall/confusion and observed macro recall,
direct queries, two-hop queries, supported positive two-hop and undetermined.
Report independently trained no-vector and wrong matching-checkpoint controls.
Paired coherent table permutation, reversed clause order, equivalent inverse
wording, valid first-edge inverse replacement; eligible directed-query reversal.
All supervision and structural preprocessing must be disclosed.
The rule baseline consumes transmitted vectors via frozen primitive receiver,
then uses endpoints and the specified same-direction transitivity rule.
Its100% is a comparison; no neural superiority gate.

For EACH receiver seed on EACH joint/phrasing/new_names/new_joint partition:
overall accuracy and observed5class macro>=.95; direct, positive two-hop and
undetermined accuracies>=.95; invariant paired accuracy>=.95;
replacement accuracy and eligible query-reversal paired accuracy>=.95;
separately trained no-vector macro<=.50 and neural-minus-no-vector macro>=.40.
No gate-driven tuning. Report a failure as failure even if ordinary tests pass.
Training scope never expands after seeing final test results.

CPU2threads,600seconds excluding primitive reproduction; fresh output folder,
finite loss/gradients, frozen checkpoint and tensor-state hash before/after.
Save receiver/control state dicts, full reports, selected fixture data/hash,
source provenance in30-day artifact, durable complete compact JSON and results
in the existing AVL-only draft PR. Synthetic sources only, no ACT/private data.
