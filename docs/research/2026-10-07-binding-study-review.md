# Follow-up review: evidence and a two-subject AVL binding study

Reviewed2026-10-07. This document records source checks and a candidate benchmark,
not a completed neural experiment or a frozen training preregistration.

## What closer inspection changes

[LatentMAS v1, sections3.1/3.2 and appendixB.2](https://arxiv.org/html/2511.20639v1)
uses internal state alignment and KV reuse. The proof's induction explicitly
relies on keys/values made by the same model on the same inputs. It therefore
does not establish semantic alignment between arbitrary independent models.
Its input/output projection is not a universal translator between AVL codebooks.
The experimental setup reports8 A100-80G GPUs; duplicating that study is a
different resource class from a small AVL CPU pilot. Its tables include accuracy
regressions in some conditions, so advantages are not uniform.
No paper result here has been reproduced.

[On the Pitfalls of Measuring Emergent Communication, AAMAS2019,
sections3 and6](https://ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf)
distinguishes informative messages from messages that change listener behavior.
Test-time channel removal can introduce distribution shift. For AVL, supplement
zero vectors with valid alternative-message interventions and a separately
trained no-channel baseline. This applies the paper's evaluation reasoning;
it does not reproduce its matrix-game experiment.

[Learning Multi-Object Positional Relationships, AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/29685)
is a useful grounding reference. Its abstract and publicly indexed PDF passages
describe varied observations with the same abstract relation; the direct PDF
retrieval failed during this follow-up. This review has not audited its complete
training implementation or hyperparameters.

[Generalization and Acquisition Speed,2020](https://arxiv.org/abs/2004.03420)
motivates measuring these properties rather than assuming compositionality
guarantees them. [EGG](https://github.com/facebookresearch/EGG) supplies reference
implementations; it has not been installed into AVL.

The Moltbook homepage was previously verified as a platform description.
Its skill document could not be retrieved in this follow-up. No live dialogue
corpus, agent autonomy, activity volume or training-data permission was verified.
Platform observations cannot substitute for controlled semantic-use evaluation.

## The question

Can a fresh sender communicate an ordered relation between two named instances
so that a listener selects the correct scene, including unseen pairs and names?

Example: A left-of B and B right-of A describe the same physical scene.
B left-of A describes a different scene. The model must preserve that distinction.

The current nine-field AVL model has no two-subject spatial-relation vocabulary.
This requires a separate research model/benchmark, not a claim that the existing
checkpoint can perform it. Preserve existing inference and artifacts.

## Candidate dataset and leak prevention

Use16 training-pool identifiers and8 disjoint test identifiers. Four physical
relations: left/right/above/below. For each unordered pair, enumerate all four
relations of the lexicographically first entity relative to the second.

Each physical scene has both equivalent grammatical orientations, four
sentence-template families, and both entity-table orders. All channels receive
the same two-name table. The sender may use table-conditioned slot tokens;
if so, declare this as a designed lexical binding adapter, not learned arbitrary
name comprehension. The table contains no relation, target class or source text.

Partition the120 training-pool unordered pairs into90 train pairs and30 reserved
pairs. Before execution, implement a fixed hash ordering and freeze its salt and
manifest; do not select pairs based on performance. Keep all physical relations
for a pair together. Opposite-word inverse paraphrases of a scene MUST remain
in the same semantic partition. Otherwise a nominal heldout scene can leak
through its inverse description.

Template families0/1 are training surfaces,2 is development,3 is heldout.
Development deliberately shares training scenes but changes surfaces; it is not
an independent semantic test. All final test partitions remain unobserved until
the declared training choices and checkpoints are frozen.

| Partition | Pairs | Template families | Sources |
|---|---:|---:|---:|
| Train |90|2|2880|
| Development surface |90|1|1440|
| Reserved pair, familiar surface |30|2|960|
| Reserved pair plus new surface |30|1|480|
| New names plus new surface |28|1|448|
| Familiar pair, new surface |90|1|1440|

Each count is pairs x4 physical relations x2 equivalent orientations x number
of templates x2 entity-table orders. These counts were checked by loop
enumeration in the research session; an actual generator and semantic-overlap
audit are still required.

Output is one of four scenes relative to the provided entity-table order.
Each fixed pair/table/surface/orientation context includes all four targets
equally. A listener seeing only the name table therefore has Bayes accuracy25%
under this enumerated design. Do NOT reserve only one relation per pair:
that would let heldout pair identity determine its target and invalidate this
metadata-only balance.

This is pair and phrasing generalization over a tiny relation vocabulary.
Do not call it general logical composition, multi-clause reasoning or grounding
in an independently observed physical world.

## Comparisons and interventions

Start with two separate questions rather than a single leaderboard:

1. Does a learned channel preserve and use binding? Compare continuous and
   discrete neural channels, an explicit compact structured oracle, and a
   separately trained no-message receiver. The oracle is a fidelity/cost bound,
   not an equally learned model.
2. Does a later real text/LLM channel improve utility at comparable total cost?
   This needs its own runtime budget; sending source bytes plus an oracle parser
   must not be presented as a competitive learned language baseline.

The learned listener sees only actual deserialized packets plus the same
relation-free entity table. Keep labels in training loss/evaluation only.
Record exact parameter counts, steps, training examples, runtime and complete
wire bytes, including table names, headers and discrete coding overhead.

Diagnostic interventions:
- Zero/random vectors: out-of-distribution robustness diagnostics only.
- Replace with a valid message from another relation of the SAME pair and table.
  Report agreement with the replacement scene as well as loss of agreement with
  the original scene. Arbitrary action changes alone are not useful semantics.
- Use the inverse paraphrase describing the SAME scene: output should stay fixed.
- Reverse the scene while keeping vocabulary and entity pair fixed: output
  should change to the correct opposite relation.
- Swap entity-table order and rebind sender slots coherently: scene-relative
  labels must invert consistently.
- Corrupt only the table/binding: separately label this an inconsistent-channel
  diagnostic; it is not expected to remain correct.

Train a no-channel receiver from scratch with the same non-message inputs and
optimization budget. Avoid using only an ablated trained receiver as proof
of communication benefit. Scene classes and query identity must not leak via
packet size, record order, filenames, batch position or metadata.

## Decisions before training

Freeze actual generator, split/dataset hashes, templates, vocabulary construction,
model sizes, continuous/discrete codec, three seeds, fixed optimizer/steps,
CPU budget, stopping rule and pass gates in a separate preregistration.
Candidate gates are>=95% accuracy AND macro recall on every test partition in
each seed,>=95% correct counterfactual/equivalence responses, and a substantial
gap to the newly trained metadata-only baseline. These are proposals, not passed
gates or final thresholds. Do not pool seeds to hide a failed one.

After local binding works, study fresh-listener sample efficiency and
independent-checkpoint cross-play. Calibration examples, translator supervision
and adapter costs must be stated explicitly.

No new neural models were trained in this review. Its completed output is the
source qualification, leakage analysis, counterbalanced count check and proposed
experimental design.
