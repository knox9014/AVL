"""Fixed tiny English-format smoke; this does not measure general language ability."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .english_connected_model import EnglishConnectedNet, encode_local_texts


ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'act-english-connected-smoke-1'
SOURCES = ['antlab/english_connected_model.py', 'antlab/english_connected_smoke.py',
           'docs/research/2026-10-05-english-connected-body.md']
CONDITIONS = ('connected', 'blocked', 'remote_swapped', 'blocked_swapped', 'serial')


def configuration():
    return dict(format=FORMAT, smoke_only=True, seed=105, steps=400, batch_size=32,
                learning_rate=.003, weight_decay=.01, gradient_clip=1., rounds=2,
                units=2, output_unit=0, max_new_tokens=8, threads=2, max_seconds=900,
                optimizer='AdamW', objective='next_character_cross_entropy_with_EOS')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                          encoding='utf-8')


def make_data():
    """Pairs differ only in the remote fact; phrase templates are fixed before training."""
    templates = {
        'train': (['Question: Is the lamp on?', 'Is the lamp on?',
                   'Please tell me: Is the lamp on?'],
                  ['The lamp is {state}.', 'Fact: The lamp is {state}.']),
        'heldout': (['Question: Is the lamp on now?',
                     'Please tell me: Is the lamp on now?'],
                    ['The lamp is {state} now.', 'Fact: The lamp is {state} now.']),
    }
    data = {}
    for split, (questions, facts) in templates.items():
        rows = []
        for question in questions:
            for fact in facts:
                for state in ('off', 'on'):
                    local = [question, fact.format(state=state)]
                    identity = hashlib.sha256(json.dumps(local, separators=(',', ':')).encode()).hexdigest()
                    rows.append(dict(id=identity, local_texts=local, fact_template=fact,
                                     fact_state=state, answer='YES' if state == 'on' else 'NO'))
        data[split] = rows
    return data


def teacher_batch(rows):
    targets = [[ord(c) for c in row['answer']] + [2] for row in rows]
    length = max(len(row) for row in targets)
    # Padding an already finished short answer has ignored supervision. Its input
    # at that ignored position is EOS, not an additional answer character.
    inputs = [[1] + row[:-1] + ([2] + [0] * (length-len(row)-1) if len(row) < length else [])
              for row in targets]
    targets = [row + [0] * (length-len(row)) for row in targets]
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _finite_model(model):
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('nonfinite model parameters')


def train(model, rows, deadline):
    cfg = configuration()
    tokens, lengths = encode_local_texts([row['local_texts'] for row in rows])
    decoder, target = teacher_batch(rows)
    generator = torch.Generator().manual_seed(cfg['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    started = time.monotonic()
    logs = []
    model.train()
    for step in range(1, cfg['steps'] + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError('English smoke wall-clock limit exceeded')
        indices = torch.randint(len(rows), (cfg['batch_size'],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens[indices], lengths[indices], decoder[indices], rounds=cfg['rounds'])['logits']
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite training logits')
        loss = F.cross_entropy(logits.reshape(-1, 128), target[indices].reshape(-1), ignore_index=0)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite training loss')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('nonfinite or missing training gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        _finite_model(model)
        if time.monotonic() >= deadline:
            raise TimeoutError('English smoke wall-clock limit exceeded')
        if step % 100 == 0:
            logs.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm.detach())))
    model.eval()
    return dict(steps=cfg['steps'], seed=cfg['seed'], examples=len(rows), batch_size=cfg['batch_size'],
                logs=logs, seconds=time.monotonic()-started,
                changed_parameter_tensors=[key for key, value in model.state_dict().items()
                                           if not torch.equal(value, initial[key])])


def evaluate(model, rows, condition):
    if condition not in CONDITIONS:
        raise ValueError('unknown English smoke condition')
    swapped = condition in ('remote_swapped', 'blocked_swapped')
    mode = 'blocked' if condition in ('blocked', 'blocked_swapped') else 'serial' if condition == 'serial' else 'connected'
    local, expected = [], []
    for row in rows:
        state = ('off' if row['fact_state'] == 'on' else 'on') if swapped else row['fact_state']
        local.append([row['local_texts'][0], row['fact_template'].format(state=state)])
        expected.append(('NO' if row['answer'] == 'YES' else 'YES') if swapped else row['answer'])
    tokens, lengths = encode_local_texts(local)
    cfg = configuration()
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        # Actual generation starts with BOS internally. No gold decoder input.
        result = model.generate(tokens, lengths, max_new_tokens=cfg['max_new_tokens'],
                                rounds=cfg['rounds'], mode=mode)
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
        raise ValueError('generation changed model weights')
    if not bool(torch.isfinite(result['state']).all()) or not bool(torch.isfinite(result['messages']).all()):
        raise ValueError('nonfinite communication state')
    correct = sum(ids == [ord(c) for c in answer] + [2]
                  for ids, answer in zip(result['token_ids'], expected))
    return dict(condition=condition, examples=len(rows), correct=correct, accuracy=correct/len(rows),
                ids=[row['id'] for row in rows], local_texts=local, expected=expected,
                token_ids=result['token_ids'], texts=result['texts'],
                eos_terminated=sum(bool(ids) and ids[-1] == 2 for ids in result['token_ids']),
                logical_payload_bytes=result['logical_payload_bytes'])


def evaluate_all(model, data):
    results = {split: {condition: evaluate(model, rows, condition) for condition in CONDITIONS}
               for split, rows in data.items()}
    for entries in results.values():
        if entries['blocked']['token_ids'] != entries['blocked_swapped']['token_ids']:
            raise ValueError('blocked generation depends on remote fact')
        if entries['connected']['token_ids'] != entries['serial']['token_ids']:
            raise ValueError('serial generation differs from synchronous generation')
    return results


def load_model(folder):
    checkpoint = torch.load(Path(folder) / 'checkpoint.pt', weights_only=True)
    expected = dict(format=FORMAT, seed=105, steps=400, parameters=98928)
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError('checkpoint metadata mismatch')
    model = EnglishConnectedNet()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    _finite_model(model)
    model.eval()
    return model


def run(output):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    cfg = configuration()
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    deadline = time.monotonic() + cfg['max_seconds']
    report = dict(status='running', smoke_only=True, config=cfg,
                  interpretation='Tiny fixed English YES/NO lamp format; no generic language or scaling claim.')
    manifest = dict(status='running', sources={}, files={})

    def save():
        write_json(folder / 'report.json', report)
        manifest['files']['report.json'] = digest(folder / 'report.json')
        write_json(folder / 'manifest.json', manifest)

    save()
    try:
        for relative in SOURCES:
            target = folder / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            manifest['sources'][relative] = digest(target)
        data = make_data()
        for name, value in (('config.json', cfg), ('data.json', data)):
            write_json(folder / name, value)
            manifest['files'][name] = digest(folder / name)
        save()
        model = EnglishConnectedNet()
        report['training'] = train(model, data['train'], deadline)
        torch.save(dict(format=FORMAT, seed=105, steps=400, parameters=98928,
                        state_dict=model.state_dict()), folder / 'checkpoint.pt')
        manifest['files']['checkpoint.pt'] = digest(folder / 'checkpoint.pt')
        report['evaluations'] = evaluate_all(model, data)
        if time.monotonic() >= deadline:
            raise TimeoutError('English smoke wall-clock limit exceeded')
        if any(digest(ROOT / path) != value for path, value in manifest['sources'].items()):
            raise ValueError('source changed during English smoke')
        report['status'] = manifest['status'] = 'complete'
        save()
        return report
    except Exception as exc:
        report['status'] = manifest['status'] = 'failed'
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise


def verify(output):
    folder = Path(output)
    torch.set_num_threads(2)
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if report['status'] != 'complete' or manifest['status'] != 'complete' or report['smoke_only'] is not True:
        raise ValueError('failed or unfinished smoke cannot verify')
    if report['config'] != configuration():
        raise ValueError('configuration mismatch')
    if set(manifest['sources']) != set(SOURCES) or set(manifest['files']) != {'report.json', 'config.json', 'data.json', 'checkpoint.pt'}:
        raise ValueError('manifest coverage mismatch')
    for path, expected in manifest['sources'].items():
        if digest(ROOT / path) != expected or digest(folder / 'source' / path) != expected:
            raise ValueError('source hash mismatch')
    for path, expected in manifest['files'].items():
        if digest(folder / path) != expected:
            raise ValueError('artifact hash mismatch')
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    data = json.loads((folder / 'data.json').read_text(encoding='utf-8'))
    if cfg != configuration() or data != make_data():
        raise ValueError('saved configuration or data mismatch')
    training = report['training']
    if (training['steps'], training['seed'], training['examples'], training['batch_size']) != (400, 105, 12, 32):
        raise ValueError('training budget mismatch')
    if [row['step'] for row in training['logs']] != [100, 200, 300, 400]:
        raise ValueError('training log coverage mismatch')
    if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0
           for row in training['logs'] for key in ('loss', 'gradient_norm')):
        raise ValueError('invalid training log')
    if not isinstance(training['seconds'], (int, float)) or not math.isfinite(training['seconds']) or not 0 <= training['seconds'] <= cfg['max_seconds']:
        raise ValueError('invalid training runtime')
    model = load_model(folder)
    changed = training['changed_parameter_tensors']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'])
        initial = EnglishConnectedNet().state_dict()
    actual_changed = [key for key, value in model.state_dict().items()
                      if not torch.equal(value, initial[key])]
    if not changed or changed != actual_changed:
        raise ValueError('invalid parameter update record')
    if evaluate_all(model, data) != report['evaluations']:
        raise ValueError('generated evaluation mismatch')
    return dict(status='complete', smoke_only=True, verified_conditions=10,
                train_accuracy=report['evaluations']['train']['connected']['accuracy'],
                heldout_accuracy=report['evaluations']['heldout']['connected']['accuracy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.verify) if args.verify else run(args.output)
    except Exception as exc:
        print(json.dumps(dict(status='failed', error=f'{type(exc).__name__}: {exc}')))
        return 1
    if args.output:
        result = dict(status=result['status'], smoke_only=True,
                      train_accuracy=result['evaluations']['train']['connected']['accuracy'],
                      heldout_accuracy=result['evaluations']['heldout']['connected']['accuracy'])
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
