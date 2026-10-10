# Grounded vector language experiment — 2026-10-09

## Question
Does explicit shared semantic grounding improve learning speed and independently trained model cross-play, and can a two-dimensional latent preserve the same finite meaning with fewer bytes? This is a controlled first step toward an AI-learnable vector language, not a claim of a globally optimal language.

Prior AVL binding cross-play failed despite perfect within-model accuracy. Related primary work motivates measuring community transmission / ease of learning separately from task success:
- https://github.com/google-deepmind/emergent_communication_at_scale (population games and ease-of-learning experiments)
- https://research.google/pubs/on-the-role-of-population-heterogeneity-in-emergent-communication/
This study uses supervised shared anchors, not those papers' emergent reinforcement-learning setup.

## Locked design before implementation or model evaluation
Three arms: free16 (original architecture and CE objective, fresh reproduction); grounded16 (same architecture, CE plus shared-anchor loss); grounded2 (same embedding16/GRU48 and receiver hidden32, latent2). Seeds44/55/66 in every arm. Independent agents never exchange weights, examples or pairwise calibration. They do share the designed four-class anchor semantics in grounded arms.

Anchor vectors: left=(-0.8,0), right=(0.8,0), above=(0,0.8), below=(0,-0.8). Grounded16 pads14 zeros. Supervised label is relative to explicit name-table order. Loss CE + mean over examples of SUM over dimensions squared anchor error (weight1). Free16 has no anchor penalty. All arms use AdamW lr0.003/wd0.0001, batch128, gradient clip1,2000 updates and identical seeded row indices. No model selection or tuning from final tests. Grounding injects extra semantic supervision; any effect is not a same-information architecture result.

Reuse exact binding dataset/splits/canonical name adapter and vocabulary. These are previously observed finite fixtures,16 unique canonical training forms and four existing relation meanings. Copied novel identifiers are not new concepts. Check validation at steps250/500/1000/2000; report first sampled step with validation macro recall>=.95 (not exact convergence time). No early stopping. Wall seconds and parameter counts are descriptive CPU measurements, not universal speed claims.

Final evaluations on train/validation/pair/joint/new_names/phrasing: matching accuracy/macro/confusion, zero and cyclic-shuffle controls; six ordered off-diagonal pairs per arm per final partition, through actual float32 serialization. Cross-play gate: every off-diagonal final partition macro recall>=.95. Matching gate: every final partition macro recall>=.95. Failure preserved, no post-test tuning.
Auxiliary bounded +/-0.05 deterministic noise before wire; not a probabilistic robustness guarantee or gate. Report coordinate MSE and example vectors.

Experimental AVS1 packet: little-endian <4sBBHI>, magic AVS1, version1, contract0(free) or1(grounded four-relation anchors), width2/16,count1..1024; float32 payload. Free contract requires width16. Metadata claims are checked at receive; no automatic guessed model compatibility. Names remain a separately accounted explicit table. Default AVL1 untouched.
Record actual wire bytes including header and table; compare text+same table and one-byte gold symbolic relation+same table. Prototype grounding reduces continuous representations to a designed four-meaning space; do not claim universal semantics or superiority over symbolic labels.

Deadline900secondsCPU2threads. Save report, checkpoints, training curves and hashes. Unit tests cover malformed packets, float32 roundtrip, anchor/inversion contract, same-initialization free16/grounded16, gradient flow and wire inference. Full regression before results. Next stage requires larger grounded semantics and withheld semantic compositions, not only renamed four-class surfaces.
