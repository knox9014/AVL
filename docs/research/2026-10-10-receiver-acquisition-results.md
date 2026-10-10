# Fresh receiver acquisition results — 2026-10-10

## Outcome
Shared grounded vectors can be acquired from one labeled training form per known relation and transferred to other same-family senders. This establishes a finite low-example interoperability result, **not faster own-sender learning or global language optimality**.

243 newly initialized receivers were trained:81 per arm,27 for each4/8/16-example budget. Senders were frozen; no old receiver parameters or pairwise calibration were used. All senders share the same training data/architecture family; grounded ones also share supervised semantic anchors.

| Arm | Own-sender final success | Other-sender final success | Median first sampled own >=95% | Median first sampled BOTH others >=95% |
| --- | --- | --- | --- | --- |
| free16 |27/27 at each budget|0/27 at each budget|5updates|not reached by100|
| grounded16 |27/27 at each budget|27/27 at each budget|5updates|5updates|
| grounded2 |27/27 at each budget|27/27 at each budget|10updates|10updates|

Success requires >=95% macro recall on **every** declared final partition, for every applicable sender, at the fixed100-update endpoint. In fact all grounded final confusion matrices are perfect100%; free own-sender metrics are also100%. No threshold-reaching trial was excluded: all own receivers and all grounded cross-sender receivers reached their validation thresholds, while no free cross-sender receiver reached its threshold.

Curves at0/1/5/10/20/50/100 updates reveal that grounded16 reaches100% validation for all trials by the10-update observation and grounded2 by20. Four examples with20 full-batch updates are80 repeated example exposures, not80 new examples. The final test was evaluated only at100updates; earlier curves are validation results. A sampled threshold crossing is not an exact convergence time or a guarantee of remaining above threshold.

Smaller vectors are not automatically easier to learn: grounded2 takes a later sampled median crossing than both width16 arms. Fixed learning rate, vector magnitude and input dimension/parameter counts are not isolated, so this is a measured configuration tradeoff, not a causal explanation. No natural-language/text acquisition baseline is tested.

## Controls and meaning limits
Zero-vector final macro recall25% everywhere. Grounded shuffled final macro recall6.25–12.5%, depending on partition. Free shuffled outcomes vary up to65.625%; cyclic shifts retain ordered answer information and are not a chance baseline.

The support selector uses training rows ONLY, unique canonical tokens, balanced class coverage and deterministic nested4/8/16 supports under selection seeds101/202/303. There are16 canonical training forms, four per known class. Receiver initialization seeds401/402/403 and sender seeds44/55/66 form structured repetitions, not81 independent unseen-world trials. Copied new names and old four meanings do not establish new concepts, new semantic combinations or heterogeneous pretrained-model transfer.

The frozen vector senders had already received the full previous supervised training. Four examples means incremental receiver acquisition cost, **not total cost of creating this protocol**. Grounded senders got extra anchor supervision; all use the same corpus. Protocol metadata and names are explicit; no labels substitute for vectors on the actual AVS1 wire.

## Reproducibility
Protocol commit6ec848350b869382e98fe85ff8a4bd5199d141fe precedes implementation/model evaluation.
Test-first missing-module failure: [run38011885223](https://github.com/knox9014/AVL/actions/runs/38011885223); acquisition training skipped.
Tested implementation head1cb9e99f7b819b8a1d7d7248484d0dd9c577b1ab.
[Successful run38012084326](https://github.com/knox9014/AVL/actions/runs/38012084326), job114094119655:6 focused and173 full tests pass; complete experiment137.45secondsCPU2threads/Python3.12/torch2.6.0CPU, including9-sender reproduction.
New sender source and dataset hashes match the prior locked configuration. All nine checkpoint file hashes differ from the prior artifact, so original artifact tensor identity is **unverified**. Current senders' before/after tensor fingerprints match exactly: no sender changed during receiver learning. This is a fresh reproduction, not a newly blinded semantic benchmark.

Artifact11654261713, SHA2567598741bb907b00cf4303db05344a5eef4b3c7052cc2c78a285968b3f0eabf20,30-day retention: full report, sender reproduction and receiver checkpoints.
[Durable lossless report](../../antlab/runs/receiver-acquisition-report-20261010.json) retains all243 trials, selected forms, curves, final metrics, checkpoint hashes and source/frozen provenance.297 unique confusion/support metrics are dictionary-interned; metric_id references metric_dictionary. Expand with expand_report. Compaction round-trip is tested and checked after the study; this compresses research evidence, not message semantics.

Local stdlib related checks18 pass. Staged/tracked publication scans still report only the pre-existing alignment matrix product at semantic_alignment_pilot.py:39 as an email pattern. Manual reinspection confirms the matched text is Python matrix multiplication. New experiment source/test/evidence scans have no findings or binary skips; scanner warnings are not suppressed and scans are not proof of no semantic private information.

## Next work
Keep acquisition cost and compatibility separate. Test input-norm controls and a larger multi-factor meaning space with heldout semantic combinations before choosing between compact and higher-dimensional grounded vectors. Preserve this negative speed finding rather than changing hyperparameters after evaluation.

[Reproduction guide](../AVL_RECEIVER_ACQUISITION_GUIDE.md) · [Vector language criteria](../AVL_VECTOR_LANGUAGE_CRITERIA.md).
