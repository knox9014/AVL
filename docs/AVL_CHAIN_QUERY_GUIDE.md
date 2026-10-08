# Experimental AVL chain-query interface v2

This bounded AI-to-AI interface transmits learned relation vectors, explicit
entity bindings and a directed query. It supports3or4 named entities forming
one chain. It is a finite experimental language interface, not universal AVL.

## Meaning contract

Primitive relation channels are0:left of,1:right of,2:above,3:below.
Each means a strict coordinate inequality on its named axis; the other axis is
unconstrained. Channel4:undetermined means the given relations do not entail any
single supported directional answer. It is not a confidence threshold or
out-of-distribution detector.

A query asks for the relation from source entity to target entity. A chain of
identical directions entails that direction; a mixed-direction chain does not
determine a supported relation under this coordinate semantics. The rule
comparison uses that definition explicitly. The shared neural receiver learns
a channel operator under this target semantics; it can still make mistakes.

Entity identifiers are copied ASCII literals. The explicit lexical adapter
replaces them with local slots before the primitive sender. Renaming does not
teach a new concept. Source input contains exactly2or3 supported English clauses
joined by exact delimiter " And ". Supported clauses include "node0 is left of
node1." and "node0 now is left of node1.". Negation, quantifiers, arbitrary
verbs and additional relation classes are unsupported.

## Wire contract

| Packet | Fields |
| --- | --- |
| AVC2 | little-endian <4sBB>: magic,node_count,edge_count; length-prefixed names; endpoint pairs; AVL1 vectors |
| Name | little-endian uint16 length, followed by1..64ASCII identifier bytes |
| Topology | one pair of uint8 global table indices per edge; ascending endpoints |
| AVL1 payload | original ordered float32 width16 packet; exactly one vector per edge |
| ACQ2 | little-endian <4sI>: magic,graph_byte_length; AVC2 graph; uint8source,uint8target |

Names must be distinct and non-reserved. Edges must form a connected chain with
no duplicates, self edges or branching; node_count3/4 and edge_count2/3.
Queries use distinct valid table slots. Decoders validate exact byte lengths,
counts, topology and finite vector values. Table and topology carry binding
information; relation semantics are not vector-only.

With five-character names, a graph is171bytes at3nodes or244bytes at4nodes;
a complete query is181/254bytes. This interface repeats graph bytes per query.
It does not provide model identity negotiation, model distribution or network
framing. Use the matching reproduced primitive and operator artifacts for the
documented experiment. Existing AVC1, ACQ1 and default AVL1 remain unchanged.

## Reproduce and ask one question

Run with the repository's pinned CPU runtime and fresh output directories:

```sh
python -m antlab.semantic_binding_pilot --output binding-reproduction
python -m antlab.semantic_shared_operator_pilot --study binding-reproduction --output shared-operator-output
```

Then construct a query without fixture labels:

```python
import json
from pathlib import Path
import torch
from antlab.semantic_alignment_pilot import load_frozen
from antlab.semantic_shared_operator_pilot import (
    SharedOperator, graph_packets, pack_query, prepare, predict,
)

study = Path("binding-reproduction")
report = json.loads((study / "report.json").read_text(encoding="utf-8"))
seed = 44
primitive = load_frozen(
    study, seed, report["vocabulary"],
    report["seeds"][str(seed)]["checkpoint_sha256"],
)
saved = torch.load(
    "shared-operator-output/operator-seed-44.pt",
    map_location="cpu", weights_only=True,
)
assert saved["sender_seed"] == seed
operator = SharedOperator()
operator.load_state_dict(saved["operator_state"], strict=True)
operator.eval()

table = ["node0", "node1", "node2", "node3"]
source = (
    "node0 is left of node1. And node1 is left of node2. "
    "And node2 is left of node3."
)
graph = graph_packets(
    primitive, [{"text": source, "table": table}], report["vocabulary"],
)[0]
packet = pack_query(graph, 0, 3)
probabilities, lengths = prepare(primitive, [packet])
answer_id = predict(operator, probabilities, lengths)[0]
print(("left of", "right of", "above", "below", "undetermined")[answer_id])
```

Routing and known inverse-channel handling are explicit. The old supervised
primitive decoder reads each relation vector; a66parameter shared learned
operator composes the channel probabilities and is reused for each path step.
Direct examples already supervise positive channel agreement. This is a strong
architectural prior and supervised finite semantics, not autonomous invention
of a common language or novel concept learning.

Research protocol: [AVL_SHARED_OPERATOR_PROTOCOL.md](AVL_SHARED_OPERATOR_PROTOCOL.md).
Review the complete experiment outcome and limitations before using this
experimental receiver. Research gate success, when reported, is separate from
ordinary test execution and cannot establish universal model compatibility.
