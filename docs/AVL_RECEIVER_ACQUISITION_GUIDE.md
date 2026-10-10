# Fresh receiver acquisition audit

This measures how a newly initialized neural receiver learns a frozen vector language from a small support set. It separates:
1. Recognizing new surfaces from the same sender.
2. Recognizing vectors from two other senders without calibration.

Run on Python3.12 with PyTorch2.6.0CPU:
```sh
python -m unittest antlab.tests.test_semantic_acquisition_data antlab.tests.test_semantic_acquisition_pilot
python -m antlab.semantic_acquisition_pilot --output receiver-acquisition-output
```
The output directory must not exist. The command first reproduces9 senders under the old locked protocol, then trains243 fresh receivers (81 per vector arm), each on4/8/16 distinct training forms for100 updates. Sender weights remain frozen.

Use only validated, detached float32 vectors with the correct explicit AVS1 contract. The internal `train_receiver` helper creates a fresh width-to32-to4 MLP; it does not reuse a sender's trained receiver. Full support batch at each update means example exposures=budget*updates, not distinct new examples.

Files:
- sender-reproduction/: prior sender report and checkpoints.
- *.pt: fresh receiver states and configuration metadata.
- report.json: all selected support, per-trial validation curves, final confusion/support metrics, gates, hashes and frozen checks.
- compact-report.json: lossless dictionary encoding of repeated metric records. Each object containing only metric_id references metric_dictionary. Expand with `expand_report` from semantic_acquisition_pilot. This is research-report compression, not a new language/wire codec.

Curves include update0. Null threshold means not reached within100updates. Conditional median counts only successful trials; always read it with the success fraction. The first sampled crossing is not the exact convergence time and is not required to remain above the threshold thereafter. Final gates use the100-update model on every declared final partition; no checkpoint selection.

Only four previously observed relation meanings are represented. Training surfaces collapse to16 canonical forms,4 per known class. All senders use the same data/architecture family; grounded arms received extra shared anchor supervision. Copies of names do not demonstrate new concept acquisition. Vector magnitude/input width can affect gradient-based optimization with fixed learning rate; this experiment does not isolate those factors or prove an optimal representation.

[Locked protocol](AVL_RECEIVER_ACQUISITION_PROTOCOL.md).
