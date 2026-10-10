# Grounded AVL vector learning comparison — 2026-10-09

## Result
Explicit shared semantic grounding repairs independently initialized same-architecture model cross-play on the finite four-relation binding task. Compact grounded2 vectors preserve the same declared meaning with lower per-message wire cost. Learning-speed superiority is **not demonstrated**.

| Arm | Matching final macro recall | Off-diagonal mean / minimum across4 final splits | Parameters | Joint bytes per message incl. names |
| --- | --- | --- | --- | --- |
| free16 |100% all3 seeds|18.4896% /0%|11124|90|
| grounded16 |100% all3 seeds|100% /100%|11124|90|
| grounded2 |100% all3 seeds|100% /100%|9990|34|

The six ordered joint cross-play values for free16 are25%,25%,0%,25%,37.5%,0% (mean18.75%). Both grounded arms achieve100% for every ordered pair on pair/joint/new_names/phrasing. Each seed is independently initialized/trained, **using the same training dataset, vocabulary and architecture family**; grounded arms also share explicit supervised anchors. No sender/receiver pair calibration or parameter sharing. This is not heterogeneous pretrained-LLM interoperability.

All nine models already reach100% validation macro recall at the first registered250-update observation, so sampled learning curves cannot distinguish acquisition speed. Training seconds range5.43–5.60 here; CPU-specific and not a demonstrated speed gain. Each model receives256000 training row exposures, but the adapter leaves only16 unique canonical training forms.

## Why it changes compatibility
Free16 is CE-only end-to-end sender/receiver training. Grounded16 has exactly the same initial parameters/architecture for the same seed but adds supervised squared distance to common anchors: left=(-.8,0), right=(.8,0), above=(0,.8), below=(0,-.8), padded with zeros. Grounded2 uses the first two dimensions only. The receiver is still a learned MLP; it never receives the label instead of the vector during inference.

Observed grounded2 sender44 example: source obj12 now is left of obj14., table [obj12,obj14], transmitted vector approximately[-.801316,-.010003]. Reversing the explicit table changes its target meaning to right and produces approximately[.769868,.014299]. These are actual learned outputs, not exact hard-coded anchor replacements. Some heldout surfaces deviate from anchors; coordinate errors are preserved in the full report. Categories are externally supervised, so this is a designed grounded communication protocol, not autonomous language emergence.

Zero-vector control25% and cyclic-shuffle control12.5% on joint for every arm/seed. +/-0.05 seeded coordinate noise retains100% matching final macro recall. This is one declared noise realization, not worst-case or distribution-free robustness; shuffle is an ordered intervention, not a chance estimate. No trained no-message learner is added in this study.

## Wire accounting
New experimental AVS1 header12bytes + float32 payload8bytes for grounded2 or64bytes for width16, plus14bytes for the two five-character literal names on joint.
Joint480 messages: grounded2=16320bytes, width16=43200, text+same names=19320, one-byte symbolic gold+names=7200.
Grounded2 is62.22% smaller than the old full vector message and15.53% smaller than this specific text reference, but2.2667x the gold symbolic representation. Model distribution, network framing, semantics specification and negotiation are excluded. This does not establish general compression superiority. Default AVL1 unchanged.

## Reproducibility and evidence
Protocol commit a301ab1987fcffec5f84c6df47de9db6abe68c93 precedes implementation and evaluation.
Test-first failure: [run37813745810](https://github.com/knox9014/AVL/actions/runs/37813745810), missing semantic_grounded_pilot, training skipped.
Tested head d6c82c9b12ad35f6a75a81ce6c4250607264ad2f.
[Successful run37814605918](https://github.com/knox9014/AVL/actions/runs/37814605918), job113439886701:6 focused and167 full tests pass. Full nine-model study completes in53.96seconds, Python3.12/torch2.6.0CPU/2threads. Declared grounded gates pass; free cross-play gate fails and remains recorded.
Artifact11567155489, SHA256 9b5ca496884722bd9582699002cf8f68acf83472028cbfb4c77b1e3027cb5441,30-day retention.
[Durable complete report](../../antlab/runs/grounded-vector-report-20261009.json) retains all training curves, confusion/support, coordinate errors, example vectors, source/data/checkpoint hashes and provenance. Subsequent commits only record results.

New source/test/report text scans have no findings/binary skips; synthetic content manually reviewed. Scans do not prove absence of all private information. Prior alignment matrix-expression false positive in the full staged tree remains separate and unchanged.

## Limits and next experiment
Four previously observed relation meanings and copied identifiers are a tiny supervised codebook. No novel semantic class, negation, compositional learned operator or unseen-model-family transfer is evaluated. Additional anchor supervision explains compatibility; comparison is not information-matched to CE-only. Width2 has fewer parameters, so it is not an equal-capacity architecture comparison.

The relevant next learnability test must use fresh receivers, smaller training budgets and a larger grounded meaning space with truly heldout semantic combinations. Register its protocol before evaluation; do not infer faster acquisition from these saturated250-step measurements. Avoid declaring an optimum from any single success.

Primary references informing the transmission/ease-of-learning questions:
- [DeepMind emergent communication at scale code](https://github.com/google-deepmind/emergent_communication_at_scale)
- [Population heterogeneity study](https://research.google/pubs/on-the-role-of-population-heterogeneity-in-emergent-communication/)
Their emergent population/RL settings differ from this supervised-anchor study.
