"""Post-hoc, declared revision-2 query experiment with matched head capacity."""
import argparse
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from .semantic_query_pilot import (
    QUESTIONS, SOURCES, QueryReceiver, canonical_hash, digest, encode_rows,
    labels_for, measure, predict, train_head)
from .semantic_v2_data import make_data
from .semantic_v2_demo import load_checkpoint


@torch.no_grad()
def features(model, vectors, arm):
    if arm == 'raw_padded':
        return F.pad(vectors, (0, 56))
    if arm == 'frozen_probabilities':
        return model.decode(vectors).softmax(-1).reshape(len(vectors), 72)
    raise ValueError('unsupported feature arm')


def run(study, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    study = Path(study)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= 600:
            raise TimeoutError('revision-2 budget exceeded')
    torch.set_num_threads(2)
    data = make_data()
    train_rows, test_rows = data['train'], data['joint']
    if {tuple(r['labels']) for r in train_rows} & {tuple(r['labels']) for r in test_rows}:
        raise ValueError('frame leakage')
    manifest = json.loads((study / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('data.json', 'config.json', 'seed-44.pt', 'seed-55.pt', 'seed-66.pt'):
        if digest(study / name) != manifest['files'][name]:
            raise ValueError('primary artifact mismatch')
    if json.loads((study / 'data.json').read_text(encoding='utf-8')) != data:
        raise ValueError('primary data mismatch')
    train_targets, targets = labels_for(train_rows), labels_for(test_rows)
    majority = torch.stack([torch.bincount(train_targets[:, i], minlength=13).argmax()
                            for i in range(len(QUESTIONS))])
    root = Path(__file__).resolve().parents[1]
    sources = SOURCES + ('antlab/semantic_query_features.py', 'docs/AVL_QUERY_PILOT_V2_PROTOCOL.md')
    report = dict(format='avl-query-feature-pilot-v2', completed=False, pilot_passed=False,
                  posthoc_architecture_study=True, reused_observed_data=True,
                  sender_training=False, source_hashes={p: digest(root / p) for p in sources},
                  data_sha256=canonical_hash(data),
                  environment=dict(python=platform.python_version(), torch=torch.__version__),
                  questions=QUESTIONS, seeds={})
    for seed in (44, 55, 66):
        model = load_checkpoint(study, seed)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint = study / ('seed-' + str(seed) + '.pt')
        before = digest(checkpoint)
        train_vectors, train_bytes = encode_rows(model, train_rows, deadline)
        vectors, test_bytes = encode_rows(model, test_rows, deadline)
        offset = seed % (len(vectors) - 1) + 1
        arms = {}
        for arm in ('raw_padded', 'frozen_probabilities'):
            deadline()
            train_features = features(model, train_vectors, arm)
            head, training = train_head(train_features, train_targets, seed+1000,
                                       deadline, head_factory=lambda: QueryReceiver(vector_width=72))
            if training['parameters'] != 10349:
                raise ValueError('matched head budget changed')
            evaluations = {
                'transmitted': measure(targets, predict(head, features(model, vectors, arm))),
                'zero': measure(targets, predict(head, features(model, torch.zeros_like(vectors), arm))),
                'shuffled': measure(targets, predict(head, features(model, vectors.roll(offset, 0), arm))),
                'query_only': measure(targets, majority.expand(len(test_rows), -1)),
            }
            train_eval = measure(train_targets, predict(head, train_features))
            gates = {name: value['accuracy'] >= .95 and value['macro_recall'] >= .95
                     for name, value in evaluations['transmitted'].items()}
            filename = arm + '-' + str(seed) + '.pt'
            torch.save(dict(format=report['format'], arm=arm, sender_seed=seed,
                            head_seed=seed+1000, state_dict=head.state_dict()), output / filename)
            arms[arm] = dict(training=training, evaluations=evaluations,
                             train_diagnostic=train_eval, gates=gates,
                             reused_frozen_decoder_parameters=45080 if arm == 'frozen_probabilities' else 0,
                             head_sha256=digest(output / filename))
        if digest(checkpoint) != before:
            raise ValueError('checkpoint changed')
        report['seeds'][str(seed)] = dict(arms=arms, checkpoint_sha256=before,
                                         train_wire_bytes=train_bytes, test_wire_bytes=test_bytes)
    deadline()
    arm_passed = {arm: all(all(entry['arms'][arm]['gates'].values()) for entry in report['seeds'].values())
                  for arm in ('raw_padded', 'frozen_probabilities')}
    report.update(completed=True, arm_passed=arm_passed,
                  pilot_passed=all(arm_passed.values()), total_seconds=time.monotonic()-started,
                  wire_scope='cached vector packets only; query and network costs excluded',
                  scope='bounded receiver feature reuse; no general-language or interoperability claim')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.study, args.output)
    summary = dict(arm_passed=report['arm_passed'], total_seconds=report['total_seconds'],
                   seeds={seed: {arm: {mode: {family: dict(accuracy=v['accuracy'],
                                                        macro_recall=v['macro_recall'],
                                                        support=v['support'], recall=v['recall'])
                                               for family, v in values.items()}
                                       for mode, values in entry['evaluations'].items()}
                                  for arm, entry in item['arms'].items()}
                          for seed, item in report['seeds'].items()})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
