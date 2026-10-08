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
