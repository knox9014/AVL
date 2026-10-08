# AVL calibration implementation plan
Goal: fixed sample-count frozen-codebook calibration, not new language training.
Spec: docs/AVL_ALIGNMENT_PILOT_PROTOCOL.md.
Architecture: one calibration research module reusing unchanged binding data,
codec and models; focused unittest and isolated CPU CI workflow.
Direct execution follows the user's standing autonomous-development instruction.

1. Write tests of real affine fitting, finite validation and balanced nested
unique canonical calibration selection. Observe missing implementation failure.
2. Implement fit_ridge(x,y), apply_ridge(x,weights), select_calibration(rows,n,seed);
safe-load frozen checkpoints, pin data/source hashes, fit54 declared trials.
3. Run focused and full regression, unchanged binding reproduction and calibration.
4. Review per-pair gates, controls, supervision and cost. Save full artifact and
durable summary. Scan explicit new files for publication and record results.
5. Update existing AVL draft PR only; preserve default inference and main.

Review focus: no test-template calibration; no canonical duplicate inflation;
wrong pairs genuinely change target class; affine bias unpenalized; actual
float32 byte receive after alignment; paired-vector and label supervision
reported separately; checkpoint hashes stable before/after; no tuning on results.


Completed2026-10-08: missing-module test-first failure observed; implementation
passed2 focused and130 full tests at441ccba1508116a5e46f4bed99824933dff8ced2.
All54 calibration trials completed and16-example gates passed. Results, retained
support/recall evidence and artifact provenance recorded. All8/16-example bridges
and fresh classifiers score100%; no bridge advantage established. Reproduction
file hashes differ from earlier checkpoints and tensor identity is unverified.
Publication scan false positive on Python @ operator manually reviewed and
recorded in DEVELOPMENT_LOG; scanner did not return clean. Existing draft PR
updated; main and default inference preserved. No protocol retuning.
