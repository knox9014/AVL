"""Post-hoc finite-grammar coverage audit, separate from preregistered gates.

The all-grammar set includes training examples. The unmeasured set covers the
remaining admitted surfaces, but was not preregistered and cannot revise the
primary study's pass/fail outcome. No training or source modification occurs.
"""

import argparse
import json
from pathlib import Path

import torch

from .semantic_v2_data import _frames, _render, _row, make_data
from .semantic_v2_demo import load_checkpoint
from .semantic_v2_run import (FORMAT, ROOT, SOURCES, digest, evaluate,
                              majority_labels, validate_configuration, verify)


def make_audit_data():
    """Enumerate all 448 frames x 12 surfaces and the 808 unmeasured rows."""
    primary_ids = {row['id'] for rows in make_data().values() for row in rows}
    rows = [_row(frame, index) for frame in _frames() for index in range(12)]
    if len(rows) != 5376 or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('unexpected or duplicate frozen grammar enumeration')
    if len(primary_ids) != 4568 or not primary_ids <= {row['id'] for row in rows}:
        raise ValueError('primary dataset does not match the frozen grammar')
    unmeasured = [row for row in rows if row['id'] not in primary_ids]
    if len(unmeasured) != 808:
        raise ValueError('unexpected unmeasured grammar count')
    return {'all': rows, 'unmeasured': unmeasured}


def _without_predictions(evaluations):
    # Keep exact counts, per-field/macro accuracy and confusion matrices, plus
    # wire/source sizes and every existing control. Repeated row predictions
    # dominate JSON size and are unnecessary for this separate coverage report.
    return {key: {k: v for k, v in value.items() if k != 'predictions'}
            if isinstance(value, dict) else value
            for key, value in evaluations.items()}


def run(study, output):
    """Verify a completed full study, audit its final checkpoints, write anew."""
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg = validate_configuration(json.loads((study / 'config.json').read_text(encoding='utf-8')))
    primary = json.loads((study / 'report.json').read_text(encoding='utf-8'))
    if (cfg['smoke_only'] or primary.get('format') != FORMAT
            or primary.get('status') != 'complete' or primary.get('smoke_only') is not False
            or set(primary.get('seeds', {})) != {'44', '55', '66'}):
        raise ValueError('completed full three-seed v2 study required')
    # Main verification checks the complete manifest, source/checkpoint hashes,
    # metadata, regenerated data and recorded primary inference. Never mutate
    # its report or infer new gates from this post-hoc audit.
    verified = verify(study)
    audit = make_audit_data()
    majority = majority_labels(make_data()['train'])
    report = dict(
        format='avl-v2-posthoc-coverage-v1', status='posthoc_coverage',
        preregistered=False, changes_primary_gates=False,
        primary_study_passed=primary['study_passed'], training_performed=False,
        scope='finite admitted grammar only; all includes training examples; unmeasured is post-hoc',
        primary_verification=verified,
        counts={'all': len(audit['all']), 'primary': 4568, 'unmeasured': len(audit['unmeasured'])},
        source_hashes={source: digest(ROOT / source) for source in
                       SOURCES + ['antlab/semantic_v2_demo.py', 'antlab/semantic_v2_coverage.py']},
        primary_source_hashes=primary['source_hashes'],
        checkpoint_hashes={f'seed-{seed}.pt': digest(study / f'seed-{seed}.pt') for seed in cfg['seeds']},
        seeds={},
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cfg['threads'])
        for seed in cfg['seeds']:
            # The existing loader uses torch.load(..., weights_only=True).
            model = load_checkpoint(study, seed=seed)
            report['seeds'][str(seed)] = {'evaluations': {
                split: _without_predictions(evaluate(
                    model, rows, majority, seed, cfg['evaluation_batch_size']))
                for split, rows in audit.items()}}
    finally:
        torch.set_num_threads(previous_threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an existing artifact being overwritten even
    # if another process creates the requested output while evaluation runs.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.study, args.output)
    print(json.dumps({key: report[key] for key in
                      ('status', 'preregistered', 'changes_primary_gates', 'counts')}, indent=2))


if __name__ == '__main__':
    main()
