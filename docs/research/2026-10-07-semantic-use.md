# AVL semantic-use research record

2026-10-07. Status: implementation draft; no new neural training or accuracy claim.

## Purpose

AVL is a learned language for AI use. Its intended messages are representations
that AI receivers can learn and use across tasks. Human-readable fields, JSON,
English renderers and deterministic answer rules are diagnostic scaffolding;
they are not the final AI language and must not define its eventual ceiling.

A successful finite-frame decoder is an initial communication demonstration,
not completion of this objective. Prioritize semantic use, learnability by new
receivers, transferable meaning and measured cost before expanding deployment.

## Literature checked

These notes summarize the linked abstracts, not full-paper replications.

- [Compositionality and Generalization in Emergent Languages (2020)](https://arxiv.org/abs/2004.09124):
  in the authors' settings, compositional structure aided transmission to new
  learners, while generalization and compositionality did not simply correlate.
  AVL implication: test new learners and novel combinations separately.
- [Emergent Communication of Generalizations (2021)](https://arxiv.org/abs/2106.02668):
  the task being communicated influenced systematicity and interpretability;
  their games required generalizations rather than only object reference.
  AVL implication: evaluate how a message supports decisions, not just identity
  recovery. This does not establish that their method will improve AVL.
- [Learning Translations (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/5):
  studies a joiner learning a community's communication from interaction data,
  including translation learning between emergent protocols.
  AVL implication: distinguish learned adaptation from zero-shot compatibility.
- [SONAR (2023)](https://arxiv.org/abs/2308.11466):
  uses a multilingual sentence representation and a text decoder.
  AVL implication: useful reference for a shared representation; multilingual
  sentence embeddings alone do not establish preservation of logical scope.
  No SONAR integration or additional model download was performed.

## Implemented draft

Added semantic_queries.answer_query and independent scenario fixtures.
The function consumes a validated predicted or gold frame, never source text:

- Metadata questions describe message kind, certainty and condition.
- Positive-atom questions return entailed, contradicted or undetermined.
- A request does not establish a current state.
- A possibility or ungrounded conditional does not establish a certain state.
- Different predicates are not treated as antonyms without a declared ontology.
- Numeric questions test whether a candidate meets a described USD constraint.
  They do not establish actual budget value or enforce a request.
- Malformed queries and inconsistent frames fail explicitly.

Use with predicted fields is a decoder-plus-rule baseline. Use with gold fields
is an annotation oracle. Neither is a learned semantic-query receiver, and
neither provides evidence of general intelligence or general AI interoperability.
No frozen v2 sources, weights, primary gates or default inference were changed.

## Next neural study: proposal, not frozen protocol

Compare a learned query receiver using only deserialized AVL vectors and query
identifiers against the existing field-decoder/rule route. Keep the sender
frozen initially to distinguish representation retention from receiver learning.
Report added receiver parameters and resources.

Keep all paraphrases and queries for a semantic frame in one partition.
Existing observed v2 tests are reused diagnostics, not fresh independent
evidence. Before training, freeze genuinely new scenarios, labels, budgets,
seeds and per-family gates. Include query-only, zero-vector and shuffled-vector
controls; report answer-class support, macro accuracy and minimal-pair errors.
Then consider task-diverse training of the message itself if retention is
insufficient. Diagnostic fields must remain auxiliary rather than a permanent
restriction that every future AVL message has exactly nine slots.

Later test a newly initialized receiver learning the frozen language with a
fixed sample budget. Separately evaluate alignment of independent languages.
Passing a learned adapter is not zero-shot mutual understanding.

## Validation record

The new fixture suite covers 448 permitted frames plus explicit uncertainty,
request, conditional, invalid-input and numeric-boundary cases. At publication
of this draft the tests have not run: local process startup is unavailable.
A separate minimal CI workflow is proposed to execute the stdlib-only query,
renderer and data tests on Python 3.10 and 3.12. No inference performance,
checkpoint reproduction or full-suite pass is claimed until actually executed.

Only synthetic semantics, code and public research references are included.
No user conversations, personal paths, credentials or raw environment logs
are part of this research record.
