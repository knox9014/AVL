# Experimental grounded vector language

This experiment optimizes the **learned vector representation**, keeping the literal name binding explicit. Four relation meanings have shared supervised coordinates; independent models learn nearby continuous outputs. This is an initial finite building block of an AI-learnable language.

Run the complete registered comparison:
```sh
python -m unittest antlab.tests.test_semantic_grounded_codec antlab.tests.test_semantic_grounded_pilot
python -m antlab.semantic_grounded_pilot --output grounded-vector-output
```
Requires PyTorch; the output directory must not already exist. Report and nine checkpoints are written there.

Use two independently trained grounded2 checkpoints:
```python
import torch
from antlab.semantic_grounded_pilot import VectorNet, encode, wire
from antlab.semantic_binding_data import RELATIONS

def load_agent(path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record["format"], record["arm"], record["width"], record["contract"]) != (
            "avl-grounded-vector-v1", "grounded2", 2, 1):
        raise ValueError("unsupported checkpoint contract")
    model = VectorNet(len(record["vocabulary"]), 2, 1)
    model.load_state_dict(record["model_state"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite parameters")
    model.eval()
    return model, record["vocabulary"]

sender, vocabulary = load_agent("grounded-vector-output/grounded2-44.pt")
receiver, _ = load_agent("grounded-vector-output/grounded2-55.pt")
rows = [{"text": "obj12 now is left of obj14.", "table": ["obj12", "obj14"]}]
with torch.no_grad():
    sent = encode(sender, rows, vocabulary)
    received = wire(sent, contract=1)
    predicted = int(receiver.receive(received).argmax(-1)[0])
print(sent.tolist(), RELATIONS[predicted])
```
The same APIs run through actual serialization during the audit. This standalone tutorial snippet has not been separately executed; reproduce with the saved study checkpoints.

AVS1 checks explicit version/contract/width/count and finite float32 payload. Contract1 defines the known supervised relation anchors; contract0 means the free16 comparison. Header validity alone does not certify a model learned the contract correctly; evaluation is required. Literal tables are separate metadata and must remain consistent. Unknown source grammar is not a general-purpose vector encoder. Existing AVL1/default checkpoints and graph formats are unchanged.

[Measured results and limits](research/2026-10-09-grounded-vector-results.md): grounded cross-play100% on the finite task, lower wire cost, **no demonstrated learning-speed improvement or optimality**.
