"""Frozen AVL v1 semantic transport study and artifact replay.

Evaluation crosses the actual byte codec before invoking the receiver. Full
configuration is immutable; smoke artifacts are explicitly excluded from gates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .english_connected_model import encode_local_texts
from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_data import FIELDS, VOCABS, make_data, normalize_text
from .semantic_model import SemanticNet

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'avl-semantic-study-v1'
SOURCES = ['antlab/semantic_model.py', 'antlab/semantic_codec.py',
           'antlab/semantic_data.py', 'antlab/semantic_run.py',
           'antlab/english_connected_model.py', 'docs/AVL_V1_PROTOCOL.md']
SPLITS = ('train', 'validation', 'combination', 'phrasing')
MODES = ('transmitted', 'zero', 'shuffled', 'query_only', 'gold_oracle')


def configuration(smoke=False):
    return dict(format=FORMAT, smoke_only=bool(smoke), seeds=[11] if smoke else [11, 22, 33],
                steps=2 if smoke else 1500, batch_size=128, learning_rate=.003,
                weight_decay=.01, gradient_clip=1., threads=2, max_seconds=1800,
                optimizer='AdamW', objective='all_nine_fields_cross_entropy',
                checkpoint='final', evaluation_batch_size=128,
                dtype='float32', parameters=98511)


def validate_configuration(cfg):
    # Never use configuration() here: tests may patch only the run preset.
    full = dict(format=FORMAT, smoke_only=False, seeds=[11, 22, 33], steps=1500,
                batch_size=128, learning_rate=.003, weight_decay=.01,
                gradient_clip=1., threads=2, max_seconds=1800, optimizer='AdamW',
                objective='all_nine_fields_cross_entropy', checkpoint='final',
                evaluation_batch_size=128, dtype='float32', parameters=98511)
    if not isinstance(cfg, dict) or cfg.keys() != full.keys() or type(cfg['smoke_only']) is not bool:
        raise ValueError('invalid configuration schema')
    if not cfg['smoke_only']:
        if cfg != full:
            raise ValueError('full study preset is frozen')
    else:
        for key in full.keys() - {'smoke_only', 'seeds', 'steps', 'batch_size'}:
            if cfg[key] != full[key]:
                raise ValueError('smoke configuration changed frozen field: ' + key)
        if not isinstance(cfg['seeds'], list) or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']) or any(type(seed) is not int or seed not in full['seeds'] for seed in cfg['seeds']):
            raise ValueError('invalid smoke seeds')
        for key in ('steps', 'batch_size'):
            if type(cfg[key]) is not int or not 1 <= cfg[key] <= full[key]:
                raise ValueError('invalid smoke budget')
    return cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('AVL v1 hard total wall-clock budget exceeded')


def _batch(rows):
    tokens, lengths = encode_local_texts([[normalize_text(row['text'])] for row in rows])
    return tokens[:, 0], lengths[:, 0]


def _labels(rows):
    labels = torch.tensor([row['labels'] for row in rows], dtype=torch.long)
    if labels.shape != (len(rows), 9):
        raise ValueError('expected nine semantic labels')
    for index, vocab in enumerate(VOCABS):
        if ((labels[:, index] < 0) | (labels[:, index] >= len(vocab))).any():
            raise ValueError('semantic label outside vocabulary')
    return labels


def _mask(logits):
    if logits.ndim != 3 or logits.shape[1:] != (9, 8):
        raise ValueError('receiver logits must be [batch,9,8]')
    logits = logits.clone()
    for index, vocab in enumerate(VOCABS):
        if not torch.isfinite(logits[:, index, :len(vocab)]).all():
            raise ValueError('nonfinite valid receiver logits')
        logits[:, index, len(vocab):] = -torch.inf
    return logits


def metrics(gold, predictions):
    gold = torch.as_tensor(gold, dtype=torch.long)
    predictions = torch.as_tensor(predictions, dtype=torch.long)
    if gold.shape != predictions.shape or gold.ndim != 2 or gold.shape[1] != 9 or not len(gold):
        raise ValueError('expected equally sized nonempty semantic matrices')
    fields = {}
    for index, (field, vocab) in enumerate(zip(FIELDS, VOCABS)):
        count = len(vocab)
        target, pred = gold[:, index], predictions[:, index]
        if ((target < 0) | (target >= count) | (pred < 0) | (pred >= count)).any():
            raise ValueError('metric label outside field vocabulary')
        confusion = torch.bincount(target * count + pred, minlength=count * count).reshape(count, count)
        support = confusion.sum(1)
        recalls = [int(confusion[i, i]) / int(support[i]) if support[i] else None for i in range(count)]
        represented = [value for value in recalls if value is not None]
        fields[field] = dict(accuracy=float((target == pred).double().mean()),
                             macro_class_accuracy=sum(represented) / len(represented),
                             macro_policy='mean_over_classes_present_in_gold',
                             classes=list(vocab), class_support=support.tolist(),
                             class_accuracy=recalls, confusion=confusion.tolist())
    return dict(examples=len(gold), exact_frame_correct=int((gold == predictions).all(1).sum()),
                exact_frame_accuracy=float((gold == predictions).all(1).double().mean()),
                macro_class_accuracy=sum(item['macro_class_accuracy'] for item in fields.values()) / 9,
                fields=fields, predictions=predictions.tolist())


@torch.no_grad()
def evaluate(model, rows, majority, seed, batch_size=128, deadline=None):
    model.eval()
    gold = _labels(rows)
    predictions = {mode: [] for mode in MODES}
    generator = torch.Generator().manual_seed(seed + 10000)
    wire_bytes = 0
    for start in range(0, len(rows), batch_size):
        _deadline(deadline)
        batch = rows[start:start + batch_size]
        tokens, lengths = _batch(batch)
        vectors = model.encode(tokens, lengths)
        packet = pack_vectors(vectors)
        wire_bytes += len(packet)
        received = unpack_vectors(packet)
        # Only deserialized vectors enter the receiver. Every vector answers all
        # nine selectors; permutation changes rows only, never field selectors.
        variants = dict(transmitted=received, zero=torch.zeros_like(received),
                        shuffled=received[torch.randperm(len(received), generator=generator)])
        for mode, value in variants.items():
            predictions[mode].extend(_mask(model.decode(value)).argmax(-1).tolist())
        predictions['query_only'].extend([list(majority) for _ in batch])
        predictions['gold_oracle'].extend(gold[start:start + len(batch)].tolist())
    _deadline(deadline)
    result = {mode: metrics(gold, pred) for mode, pred in predictions.items()}
    result.update(raw_source_bytes=sum(len(row['text'].encode('ascii')) for row in rows),
                  wire_bytes=wire_bytes, packets=(len(rows) + batch_size - 1) // batch_size,
                  payload_bytes=len(rows) * 64, header_bytes=wire_bytes - len(rows) * 64)
    return result


def majority_labels(rows):
    labels = _labels(rows)
    return [int(torch.bincount(labels[:, index], minlength=len(vocab)).argmax())
            for index, vocab in enumerate(VOCABS)]


def train(model, rows, cfg, seed, deadline):
    tokens, lengths = _batch(rows)
    labels = _labels(rows)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    started = time.monotonic()
    logs = []
    model.train()
    for step in range(1, cfg['steps'] + 1):
        _deadline(deadline)
        indices = torch.randint(len(rows), (cfg['batch_size'],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = _mask(model(tokens[indices], lengths[indices])['logits'])
        loss = F.cross_entropy(logits.reshape(-1, 8), labels[indices].reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if not torch.isfinite(norm):
            raise ValueError('nonfinite training gradient')
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == cfg['steps']:
            logs.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm)))
    _deadline(deadline)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite final checkpoint')
    delta = sum(float((value - initial[key]).double().square().sum()) for key, value in model.state_dict().items()) ** .5
    return dict(steps=cfg['steps'], seconds=time.monotonic() - started, parameter_delta_l2=delta, logs=logs)


def _gates(evaluations):
    return {split: bool(evaluations[split]['transmitted']['exact_frame_accuracy'] >= .95 and
                       all(evaluations[split]['transmitted']['fields'][field]['accuracy'] >= .95
                           for field in ('polarity', 'certainty', 'condition')))
            for split in ('validation', 'combination', 'phrasing')}


def run(output, smoke=False):
    cfg = validate_configuration(configuration(smoke=smoke))
    if cfg['smoke_only'] != bool(smoke):
        raise ValueError('run mode must match saved configuration')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + cfg['max_seconds']
    torch.set_num_threads(cfg['threads'])
    data = make_data()
    if set(data) != set(SPLITS) or any(not data[split] for split in SPLITS):
        raise ValueError('unexpected dataset splits')
    write_json(output / 'config.json', cfg)
    write_json(output / 'data.json', data)
    source_hashes = {}
    for source in SOURCES:
        _deadline(deadline)
        target = output / 'sources' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        source_hashes[source] = digest(target)
    write_json(output / 'source_hashes.json', source_hashes)
    majority = majority_labels(data['train'])
    report = dict(format=FORMAT, status='running', smoke_only=cfg['smoke_only'],
                  study_passed=False, scope='bounded synthetic semantic transport; not general English',
                  source_hashes=source_hashes, majority_labels=majority, seeds={})
    try:
        for seed in cfg['seeds']:
            _deadline(deadline)
            torch.manual_seed(seed)
            model = SemanticNet()
            parameters = sum(p.numel() for p in model.parameters())
            if parameters != cfg['parameters']:
                raise ValueError('architecture parameter count changed')
            training = train(model, data['train'], cfg, seed, deadline)
            checkpoint_name = 'seed-' + str(seed) + '.pt'
            torch.save(dict(format=FORMAT, seed=seed, config=cfg, parameters=parameters,
                            steps_completed=training['steps'], state_dict=model.state_dict()), output / checkpoint_name)
            evaluations = {split: evaluate(model, data[split], majority, seed,
                                           cfg['evaluation_batch_size'], deadline) for split in SPLITS}
            gates = _gates(evaluations)
            report['seeds'][str(seed)] = dict(training=training, evaluations=evaluations,
                                             gates=gates, heldout_passed=all(gates.values()))
            write_json(output / 'report.json', report)
        _deadline(deadline)
        report.update(status='complete', total_seconds=time.monotonic() - started,
                      study_passed=not cfg['smoke_only'] and all(value['heldout_passed'] for value in report['seeds'].values()))
        report['total_wire_bytes'] = sum(split['wire_bytes'] for value in report['seeds'].values() for split in value['evaluations'].values())
        report['total_raw_source_bytes'] = sum(split['raw_source_bytes'] for value in report['seeds'].values() for split in value['evaluations'].values())
        write_json(output / 'report.json', report)
        files = ['config.json', 'data.json', 'source_hashes.json', 'report.json'] + ['seed-' + str(seed) + '.pt' for seed in cfg['seeds']] + ['sources/' + source for source in SOURCES]
        write_json(output / 'manifest.json', dict(format=FORMAT, files={name: digest(output / name) for name in files}))
        return report
    except Exception as exc:
        report.update(status='failed', study_passed=False, total_seconds=time.monotonic() - started,
                      error=type(exc).__name__ + ': ' + str(exc))
        write_json(output / 'report.json', report)
        raise


def verify(output):
    output = Path(output)
    def read(name):
        return json.loads((output / name).read_text(encoding='utf-8'))
    manifest = read('manifest.json')
    if manifest.get('format') != FORMAT or not isinstance(manifest.get('files'), dict):
        raise ValueError('invalid manifest')
    for name, expected in manifest['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not (output / name).is_file() or digest(output / name) != expected:
            raise ValueError('artifact hash mismatch: ' + name)
    cfg = validate_configuration(read('config.json'))
    required = {'config.json', 'data.json', 'source_hashes.json', 'report.json'} | {'seed-' + str(seed) + '.pt' for seed in cfg['seeds']} | {'sources/' + source for source in SOURCES}
    if set(manifest['files']) != required:
        raise ValueError('manifest does not cover exact study artifacts')
    hashes = read('source_hashes.json')
    if set(hashes) != set(SOURCES):
        raise ValueError('unexpected source snapshot set')
    for source, expected in hashes.items():
        if digest(ROOT / source) != expected or digest(output / 'sources' / source) != expected:
            raise ValueError('source hash mismatch: ' + source)
    data = read('data.json')
    if data != make_data():
        raise ValueError('regenerated data mismatch')
    report = read('report.json')
    if report.get('format') != FORMAT or report.get('status') != 'complete' or report.get('smoke_only') != cfg['smoke_only'] or report.get('source_hashes') != hashes:
        raise ValueError('invalid completed report')
    if set(report['seeds']) != {str(seed) for seed in cfg['seeds']} or not 0 <= report['total_seconds'] <= cfg['max_seconds']:
        raise ValueError('seed or total time budget mismatch')
    training_seconds = sum(value['training']['seconds'] for value in report['seeds'].values())
    if not 0 <= training_seconds <= report['total_seconds']:
        raise ValueError('training time budget mismatch')
    torch.set_num_threads(cfg['threads'])
    majority = majority_labels(data['train'])
    if majority != report['majority_labels']:
        raise ValueError('query-only control mismatch')
    for seed in cfg['seeds']:
        saved = report['seeds'][str(seed)]
        if saved['training']['steps'] != cfg['steps'] or saved['training']['parameter_delta_l2'] <= 0:
            raise ValueError('training completion mismatch')
        expected_log_steps = sorted({1, cfg['steps']} | set(range(100, cfg['steps'] + 1, 100)))
        logs = saved['training']['logs']
        if (not isinstance(logs, list) or
                [entry.get('step') for entry in logs] != expected_log_steps):
            raise ValueError('training log step schedule mismatch')
        for entry in logs:
            for field in ('loss', 'gradient_norm'):
                value = entry.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('invalid training log ' + field)
        if not 0 <= saved['training']['seconds'] <= report['total_seconds']:
            raise ValueError('invalid individual training time')
        checkpoint = torch.load(output / ('seed-' + str(seed) + '.pt'), map_location='cpu', weights_only=True)
        if checkpoint['format'] != FORMAT or checkpoint['seed'] != seed or checkpoint['config'] != cfg or checkpoint['parameters'] != cfg['parameters'] or checkpoint['steps_completed'] != cfg['steps']:
            raise ValueError('checkpoint configuration mismatch')
        # Recreate only the seeded initial weights; this is not a training replay.
        # Preserve caller RNG state while checking the reported weight movement.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = SemanticNet()
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if sum(p.numel() for p in model.parameters()) != cfg['parameters'] or any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('invalid checkpoint weights')
        delta = sum(float((value - initial[key]).double().square().sum())
                    for key, value in model.state_dict().items()) ** .5
        if delta != saved['training']['parameter_delta_l2']:
            raise ValueError('reconstructed parameter delta mismatch')
        replay = {split: evaluate(model, data[split], majority, seed, cfg['evaluation_batch_size']) for split in SPLITS}
        if replay != saved['evaluations']:
            raise ValueError('replayed evaluation mismatch for seed ' + str(seed))
        gates = _gates(replay)
        if gates != saved['gates'] or saved['heldout_passed'] != all(gates.values()):
            raise ValueError('heldout gate mismatch')
    expected_pass = not cfg['smoke_only'] and all(value['heldout_passed'] for value in report['seeds'].values())
    if report['study_passed'] != expected_pass:
        raise ValueError('study pass flag mismatch')
    for total, field in [('total_wire_bytes', 'wire_bytes'), ('total_raw_source_bytes', 'raw_source_bytes')]:
        if report[total] != sum(split[field] for value in report['seeds'].values() for split in value['evaluations'].values()):
            raise ValueError('transport total mismatch')
    return dict(status='verified', format=FORMAT, smoke_only=cfg['smoke_only'],
                study_passed=report['study_passed'], seeds_verified=cfg['seeds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.verify and args.smoke:
        parser.error('--verify uses the saved configuration; do not combine with --smoke')
    result = verify(args.output) if args.verify else run(args.output, smoke=args.smoke)
    print(json.dumps({key: result[key] for key in ('status', 'smoke_only', 'study_passed')}, indent=2))


if __name__ == '__main__':
    main()
