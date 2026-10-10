"""Preregistered named-literal audit; no training or threshold tuning."""
import argparse
import json
from pathlib import Path
import time
import torch
from .semantic_named_values import (MAX_UINT64, send_named_segments, receive_named_packets, query_named_amount)
from .semantic_query_operators import Operator
from .semantic_query_pilot import digest, canonical_hash
from .semantic_v2_data import FIELDS, _frames, _row, make_data
from .semantic_v2_demo import load_checkpoint

AMOUNTS = (7, 17, 37, 125, 1000, 16777217, 9007199254740993, 18446744073709551615)

def audit_rows():
    result = []
    for frame in _frames():
        if frame['subject'] == 'budget' and frame['amount'] != '20':
            continue
        subject = frame['subject']
        for index in (0, 10):
            for suffix in ('east-7', 'west-12'):
                entity = subject + '-' + suffix
                for amount in AMOUNTS if subject == 'budget' else (None,):
                    text = _row(frame, index)['text'].replace('the ' + subject, 'the ' + subject + ' named "' + entity + '"', 1)
                    expected = dict(frame, entity=entity)
                    if amount is not None:
                        text = text.replace('20 USD', str(amount) + ' USD')
                        expected['amount'] = str(amount)
                    result.append(dict(text=text, meaning=expected))
    if len(result) != 2560:
        raise ValueError('preregistered dataset size changed')
    return result

def run(study, operators, output):
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= 600:
            raise TimeoutError('named literal audit budget exceeded')
    torch.set_num_threads(2)
    study, operators, output = Path(study), Path(operators), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((study / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('data.json', 'config.json', 'seed-44.pt', 'seed-55.pt', 'seed-66.pt'):
        if digest(study / name) != manifest['files'][name]:
            raise ValueError('primary artifact hash mismatch')
    if json.loads((study / 'data.json').read_text(encoding='utf-8')) != make_data():
        raise ValueError('primary data mismatch')
    operator_report = json.loads((operators / 'report.json').read_text(encoding='utf-8'))
    if operator_report['format'] != 'avl-query-learned-operators-v3' or not operator_report['completed']:
        raise ValueError('expected completed operator study')
    rows = audit_rows()
    (output / 'dataset.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    report = dict(format='avl-named-values-v1', completed=False, named_values_passed=False,
                  learned_new_entity_concepts=False, literal_channel_explicit=True,
                  sender_training=False, records=len(rows), dataset_sha256=canonical_hash(rows),
                  source_hashes={name: digest(root / name) for name in (
                      'antlab/semantic_named_values.py', 'antlab/semantic_named_values_audit.py',
                      'docs/AVL_NAMED_VALUES_PROTOCOL.md')},
                  operator_report_sha256=digest(operators / 'report.json'), seeds={})
    for seed in (44, 55, 66):
        deadline()
        checkpoint_hash = digest(study / ('seed-' + str(seed) + '.pt'))
        operator_path = operators / ('operators-' + str(seed) + '.pt')
        previous = operator_report['seeds'][str(seed)]
        if checkpoint_hash != previous['checkpoint_sha256'] or digest(operator_path) != previous['operator_sha256']:
            raise ValueError('operator provenance mismatch')
        weights = torch.load(operator_path, map_location='cpu', weights_only=True)
        if weights['format'] != operator_report['format'] or weights['sender_seed'] != seed:
            raise ValueError('operator checkpoint identity mismatch')
        numeric = Operator().eval()
        numeric.load_state_dict(weights['numeric_state'], strict=True)
        if any(not torch.isfinite(value).all() for value in numeric.state_dict().values()):
            raise ValueError('nonfinite numeric weights')
        model = load_checkpoint(study, seed)
        sent = send_named_segments(model, [row['text'] for row in rows])
        if sent['unsupported'] or sent['supported_indices'] != list(range(len(rows))):
            raise ValueError('preregistered source rejected')
        meanings = receive_named_packets(model, sent['packets'])
        correct = {field: 0 for field in (*FIELDS, 'entity')}
        exact, literals = 0, 0
        numeric_meanings = {}
        for row, actual in zip(rows, meanings):
            expected = row['meaning']
            for field in correct:
                correct[field] += actual[field] == expected[field]
            exact += actual == expected
            literals += actual['entity'] == expected['entity'] and actual['amount'] == expected['amount']
            if expected['subject'] == 'budget':
                numeric_meanings.setdefault((expected['amount'], expected['comparator']), actual)
        confusion = [[0]*3 for _ in range(3)]
        numeric_records = []
        names = ('undetermined', 'allowed', 'disallowed')
        for (literal, comparator), meaning in numeric_meanings.items():
            amount = int(literal)
            for candidate in (amount-1, amount, amount+1):
                if not 0 <= candidate <= MAX_UINT64:
                    continue
                answer = query_named_amount(numeric, meaning, candidate)
                if answer['status'] != 'model_prediction':
                    raise ValueError('within-curriculum query rejected')
                allowed = (candidate <= amount if comparator == 'at-most' else
                           candidate >= amount if comparator == 'at-least' else candidate == amount)
                target, prediction = 1 if allowed else 2, names.index(answer['answer'])
                confusion[target][prediction] += 1
                numeric_records.append(dict(amount=literal, comparator=comparator,
                                            candidate=str(candidate), target=names[target], **answer))
        support = [sum(row) for row in confusion]
        recalls = {names[c]: confusion[c][c]/support[c] for c in range(3) if support[c]}
        count = sum(support)
        numeric_metrics = dict(records=count, support=dict(zip(names, support)), recall=recalls,
                               macro_recall=sum(recalls.values())/len(recalls),
                               accuracy=sum(confusion[c][c] for c in range(3))/count,
                               confusion=confusion, absent_classes=['undetermined'],
                               scope='budget queries only; exact integer difference precedes float32 features')
        total = len(rows)
        wire = dict(total=sum(map(len, sent['packets'])), outer_headers=12*total,
                    inner_headers=12*total, vector_payload=64*total,
                    identifiers=sum(len(row['meaning']['entity']) for row in rows),
                    integer_literals=8*sum(row['meaning']['subject']=='budget' for row in rows),
                    source_utf8=sum(len(row['text'].encode('utf-8')) for row in rows))
        if wire['total'] != sum(wire[key] for key in (
                'outer_headers', 'inner_headers', 'vector_payload', 'identifiers', 'integer_literals')):
            raise ValueError('wire accounting mismatch')
        metrics = dict(full_meaning_accuracy=exact/total, literal_retention=literals/total,
                       per_field_accuracy={key: value/total for key, value in correct.items()})
        gates = dict(literal_retention=metrics['literal_retention']==1,
                     full_meaning=metrics['full_meaning_accuracy']>=.95)
        report['seeds'][str(seed)] = dict(**metrics, gates=gates, wire_bytes=wire, numeric=numeric_metrics,
                                        checkpoint_sha256=checkpoint_hash,
                                        operator_sha256=digest(operator_path))
        (output / ('numeric-' + str(seed) + '.json')).write_text(json.dumps(numeric_records, indent=2)+'\n', encoding='utf-8')
        deadline()
        if digest(study / ('seed-' + str(seed) + '.pt')) != checkpoint_hash:
            raise ValueError('frozen checkpoint changed')
    report.update(completed=True, named_values_passed=all(all(s['gates'].values()) for s in report['seeds'].values()),
                  total_seconds=time.monotonic()-started)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--operators', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.operators, args.output), indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
