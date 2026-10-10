# AVL three-entity composition pilot1
Declared 2026-10-08 before implementation or final evaluation.

Question: can an explicitly factorized language envelope combine two existing,
frozen learned relation vectors while preserving three entity bindings,
clause order, equivalent inverse descriptions and name-table permutations?
This tests designed compositional transport, NOT learned discovery of composition.

Reuse unchanged binding pilot seeds44/55/66 and reproduce fixed training in CI.
Safe-load and freeze weights; verify source/data configuration against the prior
binding summary. Record newly reproduced checkpoint hashes, never assert old
tensor identity. No adapter fitting, new neural training or parameter selection.

Source grammar: two supported binding clauses joined by exact delimiter " And ".
Three distinct ASCII names, exactly two different edges spanning all three
names. Each clause's two names are lexically replaced by local slots using their
global table-index order; relation words are not parsed by the neural sender.
The parser supplies endpoints and segmentation, not relation labels. This
structural supervision is deliberate and must appear in every result claim.

AVC1 experimental envelope: <4sBB magic,name_count,edge_count> (6 bytes), three
uint16-length ASCII names, two pairs of uint8 endpoint indices (4 bytes), then
one unchanged AVL1 packet with two ordered width16 float32 vectors (140 bytes).
Each edge uses ascending endpoint indices; duplicate/self/disconnected topology
or anything other than3names/2edges is rejected. Exact bounded length, valid
identifiers, finite tensors and exact vector count are mandatory.
Default AVL1 inference is not replaced.

Logical scenes: A r1 B; B r2 C; r1/r2 in left/right/above/below (16 tuples).
Known triples: first4 triples of combinations(obj00..obj05,3), SHA256 ordered by
"avl-composition-v1:"+colon-joined names. New triples analogously new00..new05.
Four joint tuples (0,2),(2,0),(1,3),(3,1) are withheld from the multi-clause
development fixture. Each primitive relation is already fully supervised in
the old binding training. No neural model learns from these multi-clause
development rows: withheld means unseen constructed combinations in this
envelope audit, not evidence of end-to-end learned compositional generalization.

Generate four partitions with all4 clause orientations and6 name-table
permutations: development uses known triples,12 other tuples,template0;
joint uses known triples,4 withheld tuples,template3;
phrasing uses known triples,12 other tuples,template3;
new_names uses new triples,all16 tuples,template3.
Expected rows1152/384/1152/1536. Clause order both ways evaluated as paired
intervention, not doubled base fixtures. Record hashes and per-tuple support.

Decode both edge vectors with frozen receiver, then use transmitted topology
to answer queries A->B and B->C. A->C rule: identical directional relations
entail that relation; any other pair is undetermined because orthogonal
coordinates are unconstrained. Rule is hand-written, not learned reasoning.
Report exact two-edge reconstruction, per-edge accuracy, per-tuple support/
confusion with observed-class macro recall, and derived5class query support.
Zero-message, cyclic row-shuffled vectors, wrong checkpoint receiver and
constant modal tuple are descriptive controls. Swap just the two vector rows
with topology fixed; on differing canonical labels require changed meaning.
Replace first source edge relation by inverse and require correct new meaning.
Equivalent orientation, coherent table permutation and reversed clause order
must preserve physical meaning. Counts of eligible paired rows must be explicit.

For EACH seed and EACH final partition joint/phrasing/new_names:
exact meaning and observed-tuple macro recall>=.95; paired invariant accuracy
and valid replacement accuracy>=.95; zero exact<=.30 and gap>=.60; on eligible
vector swaps changed exact>=.95 and original agreement<=.05.
No checkpoint incompatibility or efficiency success gate. No post-test tuning.

Byte accounting constructs actual packets. Compare transmitted bytes with
source UTF8 + same name table, and an explicit structured symbolic oracle
using the same envelope header, name table, endpoints and two uint8 relations.
Include all per-message structure, exclude model identity negotiation,
network framing and model distribution and say so. Metadata carries binding
topology; do not claim vector-only language. Report no learned receiver
advantage over symbolic oracle. CPU2threads,600second audit budget excluding
unchanged binding reproduction; fail on time, invalid data or nonfinite values.
Fresh output; retain data/full reports/checkpoint hashes in30-day artifact and
durable complete JSON in GitHub. Only synthetic fixtures; AVL-only draft PR.
