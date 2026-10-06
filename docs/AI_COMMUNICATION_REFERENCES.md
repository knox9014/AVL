# AI communication references and multilingual direction

2026-10-06. Research notes for AVL, based on primary papers and repositories.
These systems implement different kinds of communication; none supplies a
drop-in universal language understood by all AI models.

| Reference | Mechanism | What AVL can learn |
|---|---|---|
| [SONAR](https://github.com/facebookresearch/SONAR) / [paper](https://arxiv.org/abs/2308.11466) | Multilingual text/speech encoders share a sentence embedding space; a text decoder reconstructs/translates sentences. | Separate natural-language frontends from a shared representation; test reconstruction and negation/numbers, not similarity alone. Its demonstrated 1024-dimensional vectors are not AVL's 16-dimensional wire format. |
| [OmniSONAR paper](https://arxiv.org/abs/2603.16606) | Cross-language and cross-modal embeddings at thousands-of-languages scale. | A longer-term multilingual representation reference; broad embedding evaluation does not prove reliable all-language translation or compatibility with AVL. |
| [Large Concept Models](https://github.com/facebookresearch/large_concept_model) | Generates sentence-level concepts in SONAR space. | Process sequences of local meaning representations instead of squeezing arbitrary long input into one vector. Published experiments are much larger than AVL. |
| [EGG](https://github.com/facebookresearch/EGG) / [paper](https://aclanthology.org/D19-3010/) | Sender/receiver agents learn task-specific messages, including discrete symbols. | Train a sender and receiver jointly; evaluate new compositions and cross-play between independently trained agents. A game-specific code is not universal semantics. |
| [LatentMAS](https://github.com/Gen-Verse/LatentMAS) / [paper](https://arxiv.org/abs/2511.20639) | Agents exchange latent working memory and KV caches instead of textual intermediate reasoning. | Study latent-state sharing separately, including cache bytes and model compatibility. Cache exchange differs from a compact AVL packet. |
| [Coconut](https://github.com/facebookresearch/coconut) / [paper](https://arxiv.org/abs/2412.06769) | Reuses hidden states as embeddings for internal reasoning. | Latent computation within a model; not a model-independent inter-agent language. |
| [GibberLink](https://github.com/PennyroyalTea/gibberlink) / [ggwave](https://github.com/ggerganov/ggwave) | Prompted agents switch to a data-over-sound protocol. | Demonstrates a transport change. It does not demonstrate discovering a new semantic language. Binary AVL packets need no acoustic wrapper on ordinary networks. |
| [Translating Neuralese](https://arxiv.org/abs/1704.06960) | Translates learned agent messages by matching the listener's beliefs about the world, without paired messages and natural-language text. | Evaluate whether translated messages preserve a receiver's beliefs and downstream decisions, as well as whether reconstructed text looks plausible. Results concern the studied tasks, not arbitrary language. |
| [Compositionality and Generalization in Emergent Languages](https://arxiv.org/abs/2004.09124) | Studies generalization and acquisition of learned languages in controlled games. | A successful heldout-combination test alone does not establish compositional structure or interoperability. Test newly trained receivers and independently trained sender/receiver pairs. |
| [Agent2Agent specification](https://a2a-protocol.org/latest/specification/) | Standardizes agent discovery, tasks, messages and text/file/structured-data parts. | A possible future transport envelope for AVL packets and compatibility metadata. It does not define a shared neural meaning space or make arbitrary models understand AVL vectors. |
| [FIPA ACL in JADE](https://jade.tilab.com/technical-description/) | An established agent-message language, with registered content languages and ontologies. | Separate message intent and delivery from the ontology that gives content its meaning. This is a designed symbolic standard, not an emergent neural language. |

## Selected first implementation

Preserve the frozen AVL v2 model and add an explicit local translation layer:

`source-language text → translation to English → AVL encoder → packet-only receiver → predicted frame → canonical English → target-language translation`

The first backend is [M2M100 418M](https://huggingface.co/facebook/m2m100_418M),
whose model card declares 100 languages, including Korean, and 9,900 directions.
The actual tokenizer's language set is checked at load time. Unsupported
languages remain unsupported; this is not all existing languages. Translation
is a separately pretrained model with separate resource costs, not part of the
99,977-parameter AVL model. We download only its required PyTorch/tokenizer
files, not its duplicate TensorFlow weights. The model card identifies MIT;
external weights retain their own terms and attribution. We do not redistribute
these model weights in the AVL Git repository.

Translation outputs ordinary English and may remain outside AVL's finite
grammar. The gateway retains originals and translations locally and explicitly
rejects out-of-scope content rather than assigning invented meaning. No keyword
lookup or gold-frame mapping converts arbitrary translations into successful
vectors. Translation quality and semantic-channel accuracy need separate tests.

SONAR is the closer long-term reference for direct multilingual latent
representations. Its [repository](https://github.com/facebookresearch/SONAR)
documents version-specific fairseq2 dependencies and separate code/model
licenses. These requirements and model resources must be evaluated separately;
we have not installed SONAR or claimed its embeddings align with AVL.

## Required future evidence

AVL currently establishes compatibility only for sender and receiver sharing
the same trained checkpoint. A genuinely reusable AI language requires a
receiver from another training run or architecture to learn the same semantics.
For cross-play experiments, first hold the ontology and packet budget fixed,
reserve independent seeds, and report both unadapted transfer and adapter
training cost. Do not infer interoperability from matching vector dimensions.

The Neuralese listener-belief criterion suggests an additional, separate test:
ask the receiver to make the same evidence-based decisions before and after
translation. This is an AVL design inference from that paper, not an implemented
feature or proof of unrestricted semantic preservation.

## AVL cross-play diagnostic

We tested the frozen v2 joint heldout split (202 inputs, 101 meaning frames)
with all nine pairs of independently trained seed-44/55/66 senders and receivers.
Only serialized float32 vector packets crossed the receiver boundary. No
adapter or additional training was used; this is a post-hoc diagnostic.

| Sender / receiver | 44 | 55 | 66 |
|---|---|---|---|
| 44 | 100% | 0% | 0.9901% |
| 55 | 0% | 100% | 0% |
| 66 | 0% | 0% | 100% |

These are exact nine-field frame recovery scores, with the existing valid-field
vocabulary mask. Same-checkpoint scores match the primary study; the six
different-checkpoint pairs average 0.1650%. Identical architecture and vector
dimensions therefore did not yield a common interpretable code across these
training runs. This is not a test of every AI model or a proof that alignment
cannot work. It identifies a missing interoperability mechanism in current AVL.

[Full results](../antlab/runs/semantic-v2-crossplay-20261006.json).
Reproduce without training:

```bash
python -m antlab.semantic_v2_crossplay --study antlab/runs/semantic-v2-20261006 --output antlab/runs/my-crossplay.json
```

- Cross-language parallel inputs must preserve facts/requests, negation, scope,
  uncertainty, quantities and units after translation and vector reception.
- Reserve new languages, lexical forms, entities and numbers before training
  a genuinely multilingual AVL encoder; count rejected inputs as coverage gaps.
- Compare direct multilingual encoders with the English-pivot pipeline under
  recorded memory, latency and wire budgets.
- Track model/checkpoint, ontology and codec identities separately. Equal vector
  dimensions do not establish compatible meanings across models.
- Preserve unsupported original content and evidence instead of silently
  truncating, translating away distinctions or claiming lossless recovery.

한국어: AI가 서로 통신하는 연구는 실제로 있지만 종류가 다릅니다. SONAR는
다국어 의미 벡터와 복원, EGG는 과제에서 학습한 메시지, LatentMAS는 내부
캐시 교환입니다. GibberLink는 소리로 데이터를 보내는 방식입니다. AVL은
먼저 로컬 번역 계층을 붙이며, 번역기 지원 언어와 AVL의 의미 지원 범위를
구분합니다. 모든 언어·모든 의미를 지원한다고 주장하지 않습니다.
