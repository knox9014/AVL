"""Declared revision-3 learned operators on frozen AVL factor decisions.

The answer oracle runs only to build abstract training targets. Inference uses
two learned neural operators, not the oracle. Nine typed factors and arithmetic
features are designed inductive biases, not a claim of an unconstrained language.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_query_pilot import (
    ANSWERS, QUESTIONS, SOURCES, canonical_hash, digest, encode_rows, labels_for, measure)
from .semantic_v2_data import VOCABS, _frames, _row, make_data
from .semantic_v2_demo import load_checkpoint


def curricula():
    prop_x, prop_y = [], []
    for bits in itertools.product((0., 1.), repeat=6):
        prop_x.append(bits)
        prop_y.append((1 if bits[5] else 2) if all(bits[:5]) else 0)
    numeric_x, numeric_y = [], []
    for budget, comparator, delta in itertools.product((0., 1.), range(3), range(-101, 102)):
        numeric_x.append((budget, float(comparator == 0), float(comparator == 1),
                          float(comparator == 2), float(delta), float(abs(delta))))
        allowed = (delta >= 0 if comparator == 0 else delta <= 0
                   if comparator == 1 else delta == 0)
        numeric_y.append(0 if not budget else 1 if allowed else 2)
    return dict(
        proposition=dict(inputs=prop_x, targets=prop_y),
        numeric=dict(inputs=numeric_x, targets=numeric_y))


class Operator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 32), nn.GELU(),
                                     nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, inputs):
        if (inputs.ndim != 2 or inputs.shape[1] != 6 or not len(inputs)
                or inputs.dtype != torch.float32 or not torch.isfinite(inputs).all()):
            raise ValueError('expected finite nonempty float32 six-factor inputs')
        output = self.network(inputs)
        if not torch.isfinite(output).all():
            raise ValueError('nonfinite operator scores')
        return output


def train_operator(data, seed, deadline, steps=3000):
    torch.manual_seed(seed)
    operator = Operator()
    x = torch.tensor(data['inputs'], dtype=torch.float32)
    y = torch.tensor(data['targets'], dtype=torch.long)
    counts = Counter(y.tolist())
    weights = torch.tensor([1 / counts[int(label)] for label in y], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(operator.parameters(), lr=.003, weight_decay=.00001)
    started = time.monotonic()
    for step in range(steps):
        deadline()
        selected = torch.multinomial(weights, 256, replacement=True, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(operator(x[selected]), y[selected])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite gradient')
        optimizer.step()
    operator.eval()
    with torch.no_grad():
        pred = operator(x).argmax(-1)
    recalls = [float((pred[y == c] == c).double().mean()) for c in range(3)]
    return operator, dict(seed=seed, steps=steps, seconds=time.monotonic()-started,
                          final_loss=float(loss.detach()), class_support=[counts[c] for c in range(3)],
                          class_recall=recalls, accuracy=float((pred == y).double().mean()),
                          parameters=sum(p.numel() for p in operator.parameters()))


def proposition_features(decisions, query):
    subject = VOCABS[1].index(query['subject'])
    predicate = VOCABS[2].index(query['predicate'])
    return torch.stack((decisions[:, 0] == 0, decisions[:, 4] == 0,
                        decisions[:, 5] == 0, decisions[:, 1] == subject,
                        decisions[:, 2] == predicate, decisions[:, 3] == 0), -1).float()


def numeric_features(decisions, query):
    amounts = torch.tensor([0., 10., 20., 50., 100.])
    delta = amounts[decisions[:, 6]] - query['candidate']
    return torch.stack(((decisions[:, 1] == 4).float(),
                        (decisions[:, 7] == 1).float(), (decisions[:, 7] == 2).float(),
                        (decisions[:, 7] == 3).float(), delta, delta.abs()), -1)


@torch.no_grad()
def receive(model, proposition, numeric, vectors):
    if (vectors.ndim != 2 or vectors.shape[1] != 16 or not len(vectors)
            or vectors.dtype != torch.float32 or not torch.isfinite(vectors).all()):
        raise ValueError('expected finite nonempty float32 AVL vectors')
    decisions = model.decode(vectors).argmax(-1)
    columns = [decisions[:, 0], decisions[:, 4] + 2, decisions[:, 5] + 4]
    for query in QUESTIONS[3:11]:
        classes = proposition(proposition_features(decisions, query)).argmax(-1)
        columns.append(torch.tensor([10, 8, 9])[classes])
    for query in QUESTIONS[11:]:
        classes = numeric(numeric_features(decisions, query)).argmax(-1)
        columns.append(torch.tensor([10, 11, 12])[classes])
    return torch.stack(columns, -1)


@torch.no_grad()
def evaluate(model, proposition, numeric, vectors, targets, majority, seed):
    predictions = dict(
        transmitted=receive(model, proposition, numeric, vectors),
        zero=receive(model, proposition, numeric, torch.zeros_like(vectors)),
        shuffled=receive(model, proposition, numeric,
                         vectors.roll(seed % (len(vectors)-1) + 1, 0)),
        query_only=majority.expand(len(vectors), -1))
    return {name: measure(targets, pred) for name, pred in predictions.items()}


def run(study, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    study = Path(study)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= 600:
            raise TimeoutError('revision-3 hard budget exceeded')
    torch.set_num_threads(2)
    data = make_data()
    manifest = json.loads((study / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('data.json', 'config.json', 'seed-44.pt', 'seed-55.pt', 'seed-66.pt'):
        if digest(study / name) != manifest['files'][name]:
            raise ValueError('primary artifact hash mismatch')
    if json.loads((study / 'data.json').read_text(encoding='utf-8')) != data:
        raise ValueError('primary data mismatch')
    abstract = curricula()
    (output / 'curricula.json').write_text(json.dumps(abstract, indent=2)+'\n', encoding='utf-8')
    majority_targets = labels_for(data['train'])
    majority = torch.stack([torch.bincount(majority_targets[:, i], minlength=len(ANSWERS)).argmax()
                            for i in range(len(QUESTIONS))])
    audit_rows = [_row(frame, index) for frame in _frames() for index in range(12)]
    if len(audit_rows) != 5376:
        raise ValueError('finite coverage changed')
    joint_targets, audit_targets = labels_for(data['joint']), labels_for(audit_rows)
    root = Path(__file__).resolve().parents[1]
    sources = SOURCES + ('antlab/semantic_query_operators.py', 'docs/AVL_QUERY_PILOT_V3_PROTOCOL.md')
    report = dict(format='avl-query-learned-operators-v3', completed=False, pilot_passed=False,
                  posthoc_engineering_study=True, abstract_semantics_fully_supervised=True,
                  new_codebook_learning=False, sender_training=False,
                  environment=dict(python=platform.python_version(), torch=torch.__version__),
                  source_hashes={p: digest(root / p) for p in sources},
                  data_sha256=canonical_hash(data), curricula_sha256=digest(output / 'curricula.json'),
                  scope='bounded AI-language receiver improvement with supervised factors and abstract operators',
                  seeds={})
    for seed in (44, 55, 66):
        deadline()
        before = digest(study / ('seed-' + str(seed) + '.pt'))
        model = load_checkpoint(study, seed)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        proposition, prop_training = train_operator(abstract['proposition'], seed+2000, deadline)
        numeric, numeric_training = train_operator(abstract['numeric'], seed+2100, deadline)
        if prop_training['parameters'] + numeric_training['parameters'] != 2758:
            raise ValueError('new operator parameter budget changed')
        joint_vectors, joint_bytes = encode_rows(model, data['joint'], deadline)
        audit_vectors, audit_bytes = encode_rows(model, audit_rows, deadline)
        joint = evaluate(model, proposition, numeric, joint_vectors, joint_targets, majority, seed)
        coverage = measure(audit_targets, receive(model, proposition, numeric, audit_vectors))
        gates = {name: values['accuracy'] >= .95 and values['macro_recall'] >= .95
                 for name, values in joint['transmitted'].items()}
        filename = 'operators-' + str(seed) + '.pt'
        torch.save(dict(format=report['format'], sender_seed=seed,
                        proposition_state=proposition.state_dict(),
                        numeric_state=numeric.state_dict()), output / filename)
        if digest(study / ('seed-' + str(seed) + '.pt')) != before:
            raise ValueError('frozen checkpoint changed')
        report['seeds'][str(seed)] = dict(
            proposition_training=prop_training, numeric_training=numeric_training,
            joint_evaluations=joint, coverage=coverage, gates=gates,
            joint_wire_bytes=joint_bytes, coverage_wire_bytes=audit_bytes,
            wire_scope='batched float32 vectors only; query and network costs excluded',
            frozen_parameters=99977, new_parameters=2758, inference_parameters=102735,
            checkpoint_sha256=before, operator_sha256=digest(output / filename))
    deadline()
    report.update(completed=True, pilot_passed=all(all(s['gates'].values()) for s in report['seeds'].values()),
                  total_seconds=time.monotonic()-started)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.study, args.output)
    summary = dict(pilot_passed=report['pilot_passed'], total_seconds=report['total_seconds'],
                   seeds={seed: dict(
                       proposition_training=entry['proposition_training'],
                       numeric_training=entry['numeric_training'],
                       new_parameters=entry['new_parameters'],
                       inference_parameters=entry['inference_parameters'],
                       checkpoint_sha256=entry['checkpoint_sha256'],
                       operator_sha256=entry['operator_sha256'],
                       joint={mode: {family: dict(accuracy=v['accuracy'],
                                                  macro_recall=v['macro_recall'],
                                                  support=v['support'], recall=v['recall'])
                                     for family, v in values.items()}
                              for mode, values in entry['joint_evaluations'].items()},
                       coverage={family: dict(accuracy=v['accuracy'], macro_recall=v['macro_recall'],
                                              support=v['support'], recall=v['recall'])
                                 for family, v in entry['coverage'].items()})
                          for seed, entry in report['seeds'].items()},
                   source_hashes=report['source_hashes'],
                   data_sha256=report['data_sha256'], curricula_sha256=report['curricula_sha256'])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
