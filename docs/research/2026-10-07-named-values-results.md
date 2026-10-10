# Exact named literals around learned AVL relations

The optional AVN1 interface can transmit new instance identifiers and previously
unseen integer amounts while retaining the existing learned relation meaning.
This is an engineering extension with explicit copied literal channels. It is
not evidence that the neural vector learned new entity concepts or number
representations.

## Fixed evaluation and result

[Protocol](../AVL_NAMED_VALUES_PROTOCOL.md) was committed before implementation
and execution at c1ffc0446cea5487d52f832435c9c338994a4b80. No sender, field decoder,
or numeric operator was tuned against this evaluation.

Each frozen seed44/55/66 was evaluated on2560 sources: all256 device frames and
48 canonical-20 budget frames, templates0/10, two new identifiers per known
subject role, and eight new budget values. Total evaluations7680.

| Metric | Seed44 | Seed55 | Seed66 |
|---|---:|---:|---:|
| Exact identifier and amount retention |100%|100%|100%|
| Full meaning including identifier |100%|100%|100%|
| Each of the nine existing fields |100%|100%|100%|
| Numeric boundary accuracy |100%|100%|100%|
| Numeric macro recall over present classes |100%|100%|100%|

Both preregistered gates passed: literal retention100% and full meaning>=95%.
The numeric evaluation separately contains69 cases per seed:39 allowed and30
disallowed. Undetermined has zero support and is excluded from macro recall;
this budget-only evaluation does not establish that class's recall.

Values include16777217,9007199254740993, and18446744073709551615. Exact Python
integer subtraction occurs before converting the bounded difference to float32.
For each value and comparator, candidates are amount-1, amount, amount+1 where
uint64-valid. Distances outside[-101,101] are explicitly unsupported. This reuses
the numeric operator's taught feature domain; it does not demonstrate new math.

## Complete wire cost

For2560 sources under each seed:

| Component | Bytes |
|---|---:|
| AVN1 outer headers |30720|
| AVL1 inner headers |30720|
| Float32 vector payload |163840|
| Explicit identifiers |32768|
| Explicit uint64 values |12288|
| **Total transmitted** |**270336**|
| Source UTF-8 text |223296|

The wire representation is21.07% larger than these source sentences. No
compression benefit is demonstrated. Transport/session/model-distribution costs
are excluded. Repeated canonical texts share encoding work inside one call;
the reported audit time is not a latency benchmark.

## Evidence and reproducibility

[Successful CI run](https://github.com/knox9014/AVL/actions/runs/37626489013)
tested head277639c2aa52a8564812d9709f5f4987207d5290. All120 repository tests passed
on Python3.12/PyTorch2.6.0CPU; query/renderer/data fixtures also passed on Python3.10
and3.12. The test suite covers malformed/truncated packets, invalid identifiers,
uint64 bounds, precision above2^53, unsupported raw input isolation, actual vector
consumption and decoded role/literal disagreement.

[Durable summary](../../antlab/runs/named-values-summary-20261007.json) retains
all source, checkpoint, operator, dataset and artifact hashes, field metrics,
class support, confusion and byte accounting.
[Artifact11484546491](https://github.com/knox9014/AVL/actions/runs/37626489013/artifacts/11484546491)
contains full reports, the synthetic dataset and numeric records; retention30days.
Its zip SHA256 is adb20130b2fb5acf592e8f56da7aa84dae546e203063423f60e78b9a9ef36fcf.
The named audit took0.4675 seconds in this run, excluding all installation,
regression tests and upstream operator training.

[Usage](../AVL_NAMED_VALUES_USAGE.md) includes CLI and programmatic examples.

## Scope and next step

The literal adapter removes the identifier and replaces budget integers with20
before the existing finite-grammar encoder. Receiver restores the explicit
values only after validating the predicted known-role frame. This adds a
designed side channel visible to the receiver; vector-only semantic bandwidth
claims would be invalid. Admission checks the previously observed grammar.

Still unsupported: new subject categories, general prose, multiple bound
subjects, cross-message references, arbitrary arithmetic distance, and
independently trained checkpoint compatibility. Matching checkpoints are
required; AVN1 does not carry model identity or authentication.

The next useful semantic step is a preregistered multi-clause binding study:
two differently named subjects, shuffled name bindings as a control, explicit
condition scope and conjunction. Require successful relation binding rather
than counting copied identifiers as learned semantics.
