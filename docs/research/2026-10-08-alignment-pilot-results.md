# AVL binding-space calibration results — 2026-10-08

The preregistered 16-example gates passed for all six ordered mismatched model
pairs and all three calibration-selection seeds. Eight examples also reached
100% on all four partitions, descriptively. A fresh 68-parameter label classifier
matched the bridge at eight and sixteen examples; these results do not establish
an advantage for representation alignment over teaching another classifier.

## Design and evidence

Protocol: [AVL_ALIGNMENT_PILOT_PROTOCOL.md](../AVL_ALIGNMENT_PILOT_PROTOCOL.md),
declared at bf77d0f241f2cfeaa333aa8464333c56eb347ca4 before model execution.
Six ordered pairs of training seeds44/55/66, calibration sizes4/8/16 and
selection seeds101/202/303 produce54 trials. Each selection contains unique
canonical training strings balanced across all four known classes. Frozen
sender/receiver checkpoints are never updated during calibration.
The bridge fits paired representations of the same calibration messages and
requires access to the target model. Wrong-pair targets are cyclically shifted
across different classes. The fresh classifier instead receives explicit labels.

Tested source head: 441ccba1508116a5e46f4bed99824933dff8ced2.
[Successful CI run](https://github.com/knox9014/AVL/actions/runs/37724409724):
2 focused tests, 130 full regression tests, unchanged binding reproduction,
and calibration completed. Calibration took16.947 seconds on CPU2threads,
excluding reproduction. Initial test-first run37724262900 failed on the
intentionally absent implementation; no models ran and no protocol was retuned.
Author self-review covered training-only selection, frozen weights, wrong-class
controls, actual AVL1 serialization, cost accounting and unsupported claims.

The unchanged source/data configuration was reproduced rather than downloading
the earlier artifact. All three checkpoint file SHA256 values differ from the
prior run. Within-checkpoint gates passed again, but tensor identity with the
earlier checkpoints was not established; these are newly reproduced checkpoints,
not proven byte-identical originals. Current checkpoint hashes are verified
against the current binding report and checked unchanged after calibration.
Original binding source/data hashes match the prior durable record.

## Accuracy

Each row below summarizes18 trials (six model pairs x three selection seeds) on
the balanced480-row joint partition. Macro recall equals accuracy here.
All other partitions have the same adapted/control ranges and means.

| Calibration examples | Bridge mean [min,max] | Wrong-pair mean [min,max] | Fresh classifier mean [min,max] |
| --- | --- | --- | --- |
| 4 | 99.31% [87.5%,100%] | 1.39% [0%,12.5%] | 98.61% [87.5%,100%] |
| 8 | 100% [100%,100%] | 0% [0%,0%] | 100% [100%,100%] |
| 16 | 100% [100%,100%] | 0% [0%,0%] | 100% [100%,100%] |

Unadapted joint cross-play averages18.75%, ranging0–37.5%.
Unadapted pair-partition mean is17.71%; new-name and phrasing means are18.75%.
Wrong-label classifier matches wrong-pair bridge aggregate results.
At16 examples every trial and partition passes accuracy/macro recall>=95%,
wrong-pair accuracy<=30% and the>=60 percentage-point gap. Smaller sizes were
descriptive, not alternate success gates.

## Limits and costs

Only16 unique canonical training strings and four already supervised classes are
involved. Renaming objects relies on the designed lexical slot adapter.
Test surfaces do not establish unseen semantic composition, autonomous language
agreement, heterogeneous architecture compatibility or new concept learning.
A class-balanced four-example selection already supplies one example of every
class; this is substantial supervision for this tiny task.

The bridge has272 parameters (1088 raw float32 bytes); the fresh classifier has68
(272 bytes). The full tensor archive with all variants is205086 bytes.
Messages remain90bytes including the name table: no wire reduction occurred.
Assumed calibration payload is166n bytes for paired representations and91n for
labeled classifier examples; at4/8/16 these are664/1328/2656 versus364/728/1456.
These provisioning assumptions exclude source text, model access/distribution,
computation, adapter distribution and network overhead. They are not measured
network costs. No adapter/model identity negotiation is implemented.

The appropriate next experiment is a larger, preregistered semantic task with
multiple bound entities and heldout structures, comparing the same bridge with
a direct classifier and an explicit symbolic representation. Production default
inference and AVL1 remain unchanged.

## Durable record

[Summary JSON](../../antlab/runs/alignment-pilot-summary-20261008.json) retains all
54 trials, class supports/recalls, selected examples and hashes.
Full confusion matrices, reproduced weights and adapter tensors are in the
30-day CI artifact11526853231, ZIP SHA256
24765f5669b7cc9c12c3a395a0451aeb0675dd2d3fb5775a3cbb5ce74597d57a.
Only synthetic research fixtures are used.
