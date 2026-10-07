"""Revision-1 frozen-vector query pilot; no sender training or generality claim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_queries import answer_query, SUBJECT_PREDICATES
from .semantic_v2_data import FIELDS, VOCABS, make_data
from .semantic_v2_demo import load_checkpoint
from .semantic_v2_model import tokenize

ANSWERS = ('fact', 'request', 'certain', 'possible', 'none', 'rain', 'cold',
           'night', 'entailed', 'contradicted', 'undetermined', 'allowed', 'disallowed')
QUESTIONS = ([{'type': name} for name in ('kind', 'certainty', 'condition')] +
             [dict(type='proposition', subject=subject, predicate=predicate)
              for subject, predicates in SUBJECT_PREDICATES.items() for predicate in predicates] +
             [dict(type='constraint_allows', candidate=value, unit='USD')
              for value in (0, 9, 10, 11, 19, 20, 21, 49, 50, 51, 99, 100, 101)])
FAMILIES = ('kind', 'certainty', 'condition', 'proposition', 'constraint_allows')
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('antlab/semantic_queries.py', 'antlab/semantic_query_pilot.py',
           'antlab/semantic_v2_data.py', 'antlab/semantic_v2_model.py',
           'antlab/semantic_model.py', 'antlab/semantic_v2_demo.py',
           'antlab/semantic_codec.py', 'antlab/semantic_render.py',
           'docs/AVL_QUERY_PILOT_PROTOCOL.md')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def labels_for(rows):
    return torch.tensor([[ANSWERS.index(answer_query(row['frame'], query))
                          for query in QUESTIONS] for row in rows], dtype=torch.long)


class QueryReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        # Numeric candidates share an identifier; their values enter separately.
        self.question = nn.Embedding(12, 8)
        self.network = nn.Sequential(nn.Linear(25, 64), nn.GELU(),
                                     nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 13))
        allowed = torch.zeros(12, 13, dtype=torch.bool)
        for index, names in enumerate((('fact', 'request'), ('certain', 'possible'),
                                       ('none', 'rain', 'cold', 'night'))):
            allowed[index, [ANSWERS.index(name) for name in names]] = True
        allowed[3:11, 8:11] = True
        allowed[11, 10:13] = True
        self.register_buffer('allowed', allowed)

    def forward(self, vectors, question_ids, candidates):
        n = len(vectors)
        if (vectors.shape != (n, 16) or n < 1 or vectors.dtype != torch.float32
                or not torch.isfinite(vectors).all()):
            raise ValueError('expected nonempty finite float32 vectors')
        if (question_ids.shape != (n,) or question_ids.dtype != torch.long
                or (question_ids < 0).any() or (question_ids >= 12).any()):
            raise ValueError('invalid question identifiers')
        if (candidates.shape != (n, 1) or candidates.dtype != torch.float32
                or not torch.isfinite(candidates).all()):
            raise ValueError('invalid candidate features')
        logits = self.network(torch.cat((vectors, self.question(question_ids), candidates), -1))
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite receiver output')
        return logits.masked_fill(~self.allowed[question_ids], -torch.inf)


def query_features(segments):
    ids = torch.tensor(list(range(11)) + [11] * 13, dtype=torch.long).repeat(segments)
    candidates = torch.tensor([query.get('candidate', 0) / 100 for query in QUESTIONS],
                              dtype=torch.float32).repeat(segments).reshape(-1, 1)
    return ids, candidates


def measure(targets, predictions):
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 24:
        raise ValueError('invalid query result shape')
    result = {}
    for family in FAMILIES:
        columns = [i for i, query in enumerate(QUESTIONS) if query['type'] == family]
        gold, pred = targets[:, columns].flatten(), predictions[:, columns].flatten()
        confusion = torch.bincount(gold * len(ANSWERS) + pred,
                                   minlength=len(ANSWERS)**2).reshape(len(ANSWERS), -1)
        support = confusion.sum(1)
        recall = [int(confusion[i, i]) / int(support[i]) if support[i] else None
                  for i in range(len(ANSWERS))]
        present = [r for r in recall if r is not None]
        result[family] = dict(examples=len(gold), correct=int((gold == pred).sum()),
                             accuracy=float((gold == pred).double().mean()),
                             macro_recall=sum(present) / len(present),
                             classes=list(ANSWERS), support=support.tolist(),
                             recall=recall, confusion=confusion.tolist())
    return result


@torch.no_grad()
def encode_rows(model, rows, deadline):
    vectors, wire_bytes = [], 0
    for start in range(0, len(rows), 128):
        deadline()
        tokens, lengths = tokenize([row['text'] for row in rows[start:start + 128]])
        packet = pack_vectors(model.encode(tokens, lengths))
        vectors.append(unpack_vectors(packet))
        wire_bytes += len(packet)
    return torch.cat(vectors), wire_bytes


@torch.no_grad()
def predict(head, vectors):
    ids, candidates = query_features(len(vectors))
    repeated = vectors.repeat_interleave(len(QUESTIONS), dim=0)
    predictions = []
    for start in range(0, len(ids), 512):
        predictions.append(head(repeated[start:start+512], ids[start:start+512],
                                candidates[start:start+512]).argmax(-1))
    return torch.cat(predictions).reshape(len(vectors), len(QUESTIONS))


def train_head(vectors, targets, seed, deadline):
    torch.manual_seed(seed)
    head = QueryReceiver()
    ids, candidates = query_features(len(vectors))
    repeated = vectors.repeat_interleave(len(QUESTIONS), dim=0)
    flattened = targets.flatten()
    families = [query['type'] for _ in range(len(vectors)) for query in QUESTIONS]
    counts = Counter(zip(families, flattened.tolist()))
    weights = torch.tensor([1 / counts[(family, label)]
                            for family, label in zip(families, flattened.tolist())],
                           dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
    started = time.monotonic()
    for step in range(1200):
        deadline()
        chosen = torch.multinomial(weights, 256, replacement=True, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(repeated[chosen], ids[chosen], candidates[chosen]),
                               flattened[chosen])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite gradient')
        optimizer.step()
    head.eval()
    return head, dict(steps=1200, seconds=time.monotonic() - started,
                      final_sampled_loss=float(loss.detach()),
                      parameters=sum(p.numel() for p in head.parameters()))


@torch.no_grad()
def rule_predictions(model, vectors):
    predicted = model.decode(vectors).argmax(-1).tolist()
    output, invalid = [], 0
    for labels in predicted:
        frame = {field: vocab[index] for field, vocab, index in zip(FIELDS, VOCABS, labels)}
        try:
            output.append([ANSWERS.index(answer_query(frame, query)) for query in QUESTIONS])
        except ValueError:
            # Invalid complete frames are errors, not repaired or scored as unknown.
            output.append([-1] * len(QUESTIONS))
            invalid += 1
    return torch.tensor(output, dtype=torch.long), invalid


def run(study, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= 600:
            raise TimeoutError('query pilot time budget exceeded')
    torch.set_num_threads(2)
    data = make_data()
    train_rows, test_rows = data['train'], data['joint']
    train_frames = {tuple(row['labels']) for row in train_rows}
    test_frames = {tuple(row['labels']) for row in test_rows}
    if train_frames & test_frames or (len(train_rows), len(test_rows)) != (2776, 202):
        raise ValueError('frozen frame partitions changed')
    manifest = json.loads((Path(study) / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('data.json', 'config.json', 'seed-44.pt', 'seed-55.pt', 'seed-66.pt'):
        if digest(Path(study) / name) != manifest['files'][name]:
            raise ValueError('primary artifact hash mismatch')
    saved_data = json.loads((Path(study) / 'data.json').read_text(encoding='utf-8'))
    if saved_data != data:
        raise ValueError('saved primary data mismatch')
    train_targets, targets = labels_for(train_rows), labels_for(test_rows)
    majority = torch.stack([torch.bincount(train_targets[:, i], minlength=len(ANSWERS)).argmax()
                            for i in range(len(QUESTIONS))])
    report = dict(format='avl-frozen-vector-query-pilot-v1', completed=False,
                  pilot_passed=False, sender_training=False, reused_observed_v2_data=True,
                  scope='bounded known question families; not general reasoning or universal AI language',
                  environment=dict(python=platform.python_version(), torch=torch.__version__),
                  data_sha256=canonical_hash(data), questions=QUESTIONS,
                  source_hashes={name: digest(ROOT / name) for name in SOURCES}, seeds={})
    for sender_seed in (44, 55, 66):
        deadline()
        checkpoint = Path(study) / ('seed-' + str(sender_seed) + '.pt')
        before = digest(checkpoint)
        model = load_checkpoint(study, sender_seed)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        train_vectors, train_wire = encode_rows(model, train_rows, deadline)
        test_vectors, test_wire = encode_rows(model, test_rows, deadline)
        head, training = train_head(train_vectors, train_targets, sender_seed + 1000, deadline)
        with torch.no_grad():
            variants = dict(
                transmitted=predict(head, test_vectors),
                zero=predict(head, torch.zeros_like(test_vectors)),
                shuffled=predict(head, test_vectors.roll(sender_seed % (len(test_vectors)-1) + 1, 0)),
                query_only=majority.expand(len(test_rows), -1))
            rules, invalid = rule_predictions(model, test_vectors)
        measured = {name: measure(targets, pred) for name, pred in variants.items()}
        # Count invalid predictions as wrong without feeding -1 to bincount.
        if invalid:
            raise ValueError('field-rule baseline produced invalid frames; explicit handling needed')
        measured['decoded_field_rules'] = measure(targets, rules)
        gates = {family: values['accuracy'] >= .95 and values['macro_recall'] >= .95
                 for family, values in measured['transmitted'].items()}
        weight_name = 'query-head-' + str(sender_seed) + '.pt'
        torch.save(dict(format=report['format'], sender_seed=sender_seed,
                        head_seed=sender_seed+1000, state_dict=head.state_dict()), output / weight_name)
        if digest(checkpoint) != before:
            raise ValueError('sender checkpoint changed')
        report['seeds'][str(sender_seed)] = dict(
            head_seed=sender_seed+1000, checkpoint_sha256=before, training=training,
            train_wire_bytes=train_wire, test_wire_bytes=test_wire,
            wire_scope='cached input vectors in batches of 128; queries and network overhead excluded',
            evaluations=measured, gates=gates, invalid_decoded_frames=invalid,
            head_sha256=digest(output / weight_name))
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
    print(json.dumps(dict(pilot_passed=report['pilot_passed'],
                          total_seconds=report['total_seconds'],
                          families={seed: {family: dict(accuracy=value['accuracy'], macro_recall=value['macro_recall'])
                                           for family, value in entry['evaluations']['transmitted'].items()}
                                    for seed, entry in report['seeds'].items()}), indent=2))


if __name__ == '__main__':
    main()
