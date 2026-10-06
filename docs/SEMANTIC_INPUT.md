# Meaning-preserving input representations

Status: proposed design; an automatic semantic organizer has not been implemented.

The purpose of the input layer is to help the model distinguish and use meaning. A cleaner summary or a JSON object alone does not establish understanding.

Proposed flow:

`user text → learned meaning encoder → local representations → vector communication → computation → user answer`

## Preserve distinctions

Represent requests, stated facts, entities and relations, numeric constraints and units, negation, conditional scope, uncertainty, and contradictory claims. Preserve unhandled text rather than silently dropping it. Keep the original source and evidence spans in the input layer so extraction and interpretation can be audited.

For example, `Turn off the lamp.` and `Switch the lamp off.` share a request. `Do not turn off the lamp.` is a prohibition, while `The lamp may be off.` is an uncertain state claim. These differences must remain usable by downstream units.

Do not force arbitrarily long input into one fixed 16-dimensional vector before testing retention. Evaluate multiple local representations for complex input and record the resulting memory and transmission costs. Segmentation into sentences is not itself semantic understanding.

## Learning and evaluation

Start by checking whether the existing encoder preserves distinctions in local tasks. Use paraphrases, minimal changes in negation and conditions, and multiple questions about the same representation. Human-readable structured annotations can be auxiliary training targets and diagnostic labels; they are not assumed to be the deployed AI language.

Separate input-to-representation retention from representation-to-message retention. Compare gold annotations with automatically inferred representations. Reserve new phrasings, entity/numeric combinations, and task types for evaluation before training. Freeze budgets and criteria before running a new study.

Measure omitted content, invented content, incorrect scope, evidence errors, task accuracy, and total payload/computation. A vector distance or a fluent answer alone is not a sufficient test of meaning preservation.

Do not give every receiving unit the full original text while claiming success through a compressed channel. Keep the original accessible through explicit, measured evidence requests when needed. External LLM organizers, if evaluated later, must have their capabilities and costs reported separately.
