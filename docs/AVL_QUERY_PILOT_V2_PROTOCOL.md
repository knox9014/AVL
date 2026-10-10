# AVL query pilot revision 2: learned feature reuse

2026-10-07. Declared after revision 1 and its post-hoc diagnostic, before this
revision's execution. Motivated by revision 1: training proposition class recalls
were 100% for entailed/contradicted, but joint-test recalls were 0%; decoded-field
rules recovered all query answers. This supports a receiver-generalization
problem in this setting, not loss of all relevant vector information.

## Hypothesis

Reusing the frozen v2 field receiver's soft probabilities as learned features
helps a newly trained query head generalize across heldout semantic frames.
The new receiver still receives AVL vectors, query IDs and numeric candidates.
Gold frames and rule answers are training labels only. No rule executes inside
the learned query head. The frozen feature extractor has prior nine-field
supervision; this is an auxiliary-structure experiment, not an unconstrained
emergent-language demonstration or interoperability between independent models.

## Matched comparison

Keep revision-1 data, questions, masks, seed pairing, balanced sampling, learning
rate, 1,200 steps, final-head selection, metrics and >=95% per-family accuracy
and macro-recall gates unchanged. Reuse the already observed joint data and
label this as a post-hoc architecture study, not fresh confirmatory evidence.

Both arms use question embedding12x8 and MLP81->64->64->13 (10,349 trainable
parameters) and the same initialization seed per sender. Arm raw_padded uses
the received vector16 followed by 56 zeros; arm frozen_probabilities uses all
nine frozen decoder softmax distributions (9x8=72). Both receive identical query
features, training labels and sampling schedules. Sender and field decoder stay
frozen. Report the 45,080 pretrained decoder parameters used by the second arm
separately from its new head. Raw inference can omit that decoder.

Run both arms for sender seeds44/55/66 and head seeds1044/1055/1066, CPU2threads,
within600seconds. Compare transmitted, zero-vector, shuffled-vector and
query-only controls. For probability features, apply corruption to the actual
received vectors BEFORE the frozen feature decoder, including zero vectors.
Feature extraction never sees labels or frame IDs. Cache each feature view.

Record hashes, per-family class support/recall, training diagnostics, timings and
head weights. Each arm passes only if every family passes for every seed. A
successful workflow may record a failed research gate; never conflate the two.
No tuning based on this revision's joint scores is permitted within revision 2.
No claim of sample efficiency: both receivers already depend on the v2 encoder,
and one additionally reuses supervised receiver knowledge.

Success would motivate a task-diverse auxiliary training path for AVL. It would
not make nine human-readable fields the definition of the final AI language.
Long-term messages must be tested on meanings and tasks beyond this finite
ontology. Network expansion remains outside scope.
