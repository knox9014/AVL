# AVL shared relation operator and four-node transfer pilot1
Declared2026-10-08 before implementation/final inspection.

Reuse the observed failed flat receiver as a HISTORICAL reference, not a fresh
blind comparator or same-information/capacity baseline. Its joint results are
already public. Reuse the same three-node train/joint semantic split openly.
New structural transfer test: four nodes and three edges, never used in neural
training. Full primitive relation combinations are64; no new primitive concepts.

Architecture: frozen primitive sender AND its supervised4class receiver.
Parse explicit graph endpoints, find the unique directed path, reverse primitive
probability channels with known inverse permutation[1,0,3,2] when traversing
against a canonical edge. Routing/inversion are designed; not neural discovery.
One learned shared channel operator g:2->16 GELU->1 (65parameters) is applied
identically to each of four relation channels. Append one learned unknown logit
bias (1parameter). Total66 new parameters, plus676 frozen primitive receiver
parameters =742 active receiver parameters. This is a strong structural prior
and access to pretrained class semantics, unlike the prior raw50feature MLP.

For one-edge queries evaluate g(p,p); for two edges evaluate g(p,q).
For longer paths recursively softmax the five logits, take the four directional
probabilities (unknown mass remains discarded, not renormalized) and combine
with next edge. No hard equality/transitivity or absorbing-unknown branch in
learned inference; the rule baseline retains explicit equality logic.
One operator is shared across relation channels AND every path step. Direct
examples already supervise(1,1) probability patterns across all four channels;
two-edge examples supervise mixed patterns. Thus semantic patterns at this
binary operator level are already taught. Do not claim zero-shot rule discovery.

Primitives seeds44/55/66 reproduced with unchanged original configuration,
safe-load/freeze and verify source/data/file/tensor hashes. Operator seeds
244/255/266.3000steps,batch128,AdamW0.003,weight_decay0.0001,clip1.
Uniform output-class sampling then uniform within class. No validation selection,
extra curriculum, checkpoint choice or post-test tuning.
Train another66parameter operator on identical batches with primitive vectors
zeroed BEFORE frozen decoder softmax; same topology, routing and training.
No-vector controls must be reported, not hidden by primitive decoder biases.

Three-node training: old12tuples,template0,1152scenes and6912queries.
Old joint/new_joint/phrasing cohorts are retained and explicitly observed.
Query-rooted signatures still36train versus12joint, disjoint.
Four-node transfer: first2 quadruples of combinations(obj00..obj07,4) SHA256
ordered by "avl-chain-v2:"+colon-joined names; analogously new00..new07.
All64 relation triples,orientation masks0/7, lexicographic4node permutation
indices0/9/14/23,template3.1024scenes/12288directed queries per known/new cohort.
Queries all ordered distinct pairs; include distance1,2,3. Three-step positive
relations require all three directions identical; otherwise undetermined.
Report distance-specific and positive/unknown support, full5class confusion.
The reference rule uses only transmitted vectors via the frozen decoder.
Four-node cohort is fresh STRUCTURE at greater path length; its primitive and
binary-operation semantic cases are already supervised.

AVC2 experimental bounded chain packet: <4sBB magic,node_count,edge_count>,
3or4 unique length-prefixed ASCII names,2or3 pairs of uint8 canonical endpoints,
then unchanged AVL1 vector packet (one width16 vector per edge).
Require connected chain,degree<=2,unique edges,all nodes used,finite vectors.
ACQ2 wrapper:<4sI magic,graph_byte_length>,graph,two distinct uint8 query indices.
Actual lengths with5character identifiers:181bytes three-node query,254four-node.
Explicit symbolic gold representation uses same headers/names/endpoints/queries
plus one uint8 relation per edge:43/53bytes. Text reference includes same name
table and10query-wrapper bytes; graph repeated per question. Network/model
identity/distribution excluded. Existing AVC1/ACQ1/defaultAVL1 unchanged.

Interventions: zero vectors, cyclic-shuffled whole edge-vector matrices among
same-size scenes, mismatched primitive receiver checkpoints (three cyclic pairs),
independently trained no-vector operator. Base evaluation uses all scenes.
Paired inverse wording, coherent table permutation, clause-order reversal and
first-edge inverse replacement use preregistered deterministic subsamples:
group by physical relation tuple; select SHA256-order rows by compact sorted
JSON. Take4 per tuple for three nodes,1 per tuple for four nodes (48/16/64scenes).
Record sample hashes/counts. Directed-query reversal pairs use every base query.
Inference depends only on transmitted bytes and model parameters, not truth rows.

For EVERY model on EVERY old joint/phrasing/new_joint and fresh known4/new4:
accuracy and observed-class macro>=.95; direct,positive multi-hop,undetermined
accuracies>=.95; at four nodes positive distance3 accuracy>=.95;
paired invariant and replacement accuracy>=.95; eligible query-reversal both
correct>=.95; separately trained no-vector macro<=.50 and gap>=.40.
Report failures unchanged; do not tune protocol after inspecting final scores.
No superiority gate over the perfect rule baseline.

CPU2threads,600seconds excluding primitive reproduction. Fresh output folder;
finite loss/gradients, frozen tensor/file hashes before/after, safe-load saved
operator state; full data/reports/weights30-day artifact and complete durable JSON.
Only synthetic AVL research content; update existing draft PR, preserve main.
