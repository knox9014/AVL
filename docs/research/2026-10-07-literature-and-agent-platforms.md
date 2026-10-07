# AVL literature and agent-platform research
Reviewed 2026-10-07. Reference research only; no external accounts registered,
agents connected, messages posted, third-party datasets copied or dependencies
installed. Abstracts, official documentation/repositories and selected HTML
sections were inspected. These results have not been independently reproduced.

## Separate three concerns

- Language: a receiver uses communicated representations to solve a task.
- Transport: participants discover capabilities and exchange tasks/messages.
- Platform: participants meet and publish or collaborate.

AVL's immediate objective is the first. Transport and platform references inform
future interfaces without replacing tests of semantic understanding.

## Primary research references and proposed use

1. [Learning Multi-Object Positional Relationships via Emergent Communication,
   AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/29685).
   Studies communicating relations between two objects; different surface
   observations for speaker/listener help test abstraction.
   Proposed AVL use: swap named subjects and presentation order, reserve relation
   combinations, and test whether the listener selects the correct action.
   Transfer from image games to AVL text is a hypothesis, not an established result.

2. [Emergent Communication of Generalizations, NeurIPS2021](https://arxiv.org/abs/2106.02668).
   Uses sets and abstract concepts rather than only individual-object identity.
   Proposed AVL use: later test group descriptions and quantification on new
   combinations; copied individual identifiers do not demonstrate this ability.

3. [Emergent Language Generalization and Acquisition Speed are not tied to
   Compositionality,2020](https://arxiv.org/abs/2004.03420).
   Shows tasks where compositional structure does not automatically improve
   generalization or acquisition speed.
   Proposed AVL use: measure unseen-combination success and new-listener learning
   curves directly instead of equating an elegant vector structure with quality.

4. [Exploring Zero-Shot Emergent Communication in Embodied Multi-Agent
   Populations,2020](https://arxiv.org/abs/2010.15896).
   Investigates communication with previously unseen partners in an embodied
   setting, including shared coordination cues.
   Proposed AVL use: independently trained sender/receiver cross-play and explicit
   shared calibration conditions. The embodiment result is not a universal
   recipe for aligning arbitrary vector spaces.

5. [Latent Collaboration in Multi-Agent Systems / LatentMAS,
   preprint2025; official implementation](https://arxiv.org/abs/2511.20639),
   [code](https://github.com/Gen-Verse/LatentMAS).
   Exchanges internal continuous representations/working memory rather than
   relying entirely on text. This is closer to AVL's continuous channel goal
   than a social website. Its LLM hidden-state/KV machinery differs from AVL's
   compact16-dimensional packet and does not establish compatibility among
   arbitrary independently trained models. Reported benchmark gains are the
   authors' results, not an AVL result.

6. [When Less Latent Leads to Better Relay, preprint2026](https://arxiv.org/abs/2604.13349).
   Studies compression of KV relay and residual backfill.
   Proposed AVL use: evaluate task accuracy against actual transmitted bytes,
   encoding/decoding latency and memory. Fewer emitted text tokens alone do not
   establish lower wire cost.

7. [Beyond Tokens, survey preprint2026](https://arxiv.org/abs/2606.05711).
   Organizes latent communication by transmitted representation, alignment and
   receiver fusion. Use as a navigation aid, verifying individual methods in
   their primary papers. Do not inherit blanket guarantees from survey prose
   such as training-free methods being immune to distribution shift.

## Research implementations

[Meta EGG](https://github.com/facebookresearch/EGG) supplies sender/listener games,
discrete and continuous channels, population sampling and language diagnostics.
Reference its experiment structure; adoption would require version, license and
compatibility review.

[DeepMind Emergent Communication at Scale](https://github.com/google-deepmind/emergent_communication_at_scale)
provides population communication experiments for the ICLR2022 work.
Use as a reference for comparing multiple partners under a fixed resource budget;
more agents by itself is not evidence of a better language.

## Platforms and interoperability

[Moltbook official site](https://www.moltbook.com/) describes an AI-agent social
network with posts, comments, communities and voting. Its owner-verification
flow is described on the homepage. Only the public description was inspected:
the text rendering returned zero counters, so this review does not infer live
activity volume, inspect a representative conversation corpus, or authenticate
who authored any post. Similar-looking domains were excluded.

Potential AVL lessons: thread context, reply targets and identity. Social posts
are not an evaluation oracle or proof of autonomous language emergence. Public
visibility alone does not establish permission to bulk collect or redistribute
content; any future dataset needs provenance and applicable use terms.

[A2A official specification](https://a2a-protocol.org/latest/specification/)
defines agent discovery/capabilities, messages, tasks, artifacts, status,
extensions and transport bindings. It is an interoperability layer, not a
learned vector language. A later AVL adapter could carry a declared custom
payload and model/codec capability metadata; this is a design proposal and has
not been implemented or interoperability-tested.

## Recommended next experiment, not yet a preregistration

Prioritize a controlled two-subject relation/binding game before platform work.

Example: A is left of B. The listener must distinguish this from B is left of A,
regardless of name spelling or sentence order. Reserve entity-pair and
relation/surface combinations before training. Evaluate exact roles, relation,
action success and per-class support, not only whole-dataset accuracy.

Compare text, compact structured data, continuous AVL vectors, and a discrete
learned channel under matched access to literal metadata and bounded resources.
Run vector removal/shuffling, name-binding shuffling and position-only controls.
A copied identifier must not count as a learned binding result.

After binding works, measure new-listener acquisition at fixed sample counts,
then independent-checkpoint cross-play. Record calibration labels and adapter
costs separately. Only later consider A2A integration or controlled platform
trials. No experiment has been executed as part of this literature review.
