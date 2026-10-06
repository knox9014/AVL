"""Post-hoc sender/receiver cross-seed probe on the frozen joint heldout split.

This measures compatibility among three independently trained AVL v2 instances.
It does not establish interoperability among general AI models. No adapters,
training, gate changes or source changes are performed.
"""

import argparse
import json
from pathlib import Path

import torch

from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_v2_data import make_data
from .semantic_v2_demo import load_checkpoint
from .semantic_v2_run import ROOT, SOURCES, _batch, _labels, _mask, digest, metrics, verify


@torch.inference_mode()
def evaluate_pair(sender, receiver, rows, batch_size=128):
    """Receiver sees only the sender's actual deserialized float32 vectors."""
    predictions, wire_bytes, packets = [], 0, 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        tokens, lengths = _batch(batch)
        packet = pack_vectors(sender.encode(tokens, lengths))
        received = unpack_vectors(packet)
        predictions.extend(_mask(receiver.decode(received)).argmax(-1).tolist())
        wire_bytes += len(packet)
        packets += 1
    result = metrics(_labels(rows), predictions)
    result.pop('predictions')
    result.update(wire_bytes=wire_bytes, payload_bytes=64 * len(rows),
                  header_bytes=wire_bytes - 64 * len(rows), packets=packets)
    return result


def run(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    verified = verify(study)
    if verified['smoke_only'] or verified['seeds_verified'] != [44, 55, 66]:
        raise ValueError('verified full three-seed study required')
    cfg = json.loads((study / 'config.json').read_text(encoding='utf-8'))
    primary = json.loads((study / 'report.json').read_text(encoding='utf-8'))
    rows = make_data()['joint']
    if len(rows) != 202:
        raise ValueError('frozen joint split must have 202 rows')
    models = {seed: load_checkpoint(study, seed=seed) for seed in cfg['seeds']}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cfg['threads'])
    try:
        pairs = {f'{sender}->{receiver}': evaluate_pair(models[sender], models[receiver], rows,
                                                       cfg['evaluation_batch_size'])
                 for sender in cfg['seeds'] for receiver in cfg['seeds']}
    finally:
        torch.set_num_threads(old_threads)
    same = [pairs[f'{seed}->{seed}']['exact_frame_accuracy'] for seed in cfg['seeds']]
    cross = [value['exact_frame_accuracy'] for key, value in pairs.items()
             if key.split('->')[0] != key.split('->')[1]]
    primary_joint = {str(seed): primary['seeds'][str(seed)]['evaluations']['joint']['transmitted']['exact_frame_accuracy']
                     for seed in cfg['seeds']}
    report = dict(
        format='avl-v2-posthoc-crossplay-v1', status='posthoc_crossplay',
        preregistered=False, training_performed=False, adapters_used=False,
        changes_primary_gates=False, primary_study_passed=primary['study_passed'],
        dataset='frozen_joint_heldout', examples=202, semantic_frames=101,
        surface_templates=[10, 11], same_seed_mean_exact=sum(same)/len(same),
        different_seed_mean_exact=sum(cross)/len(cross),
        primary_joint_exact=primary_joint,
        same_seed_matches_primary={str(seed): pairs[f'{seed}->{seed}']['exact_frame_accuracy'] == primary_joint[str(seed)]
                                   for seed in cfg['seeds']},
        receiver_input='deserialized_packet_vectors_and_field_selectors_only',
        scope='Cross-seed AVL v2 compatibility only; not general AI interoperability.',
        primary_verification=verified,
        source_hashes={source: digest(ROOT / source) for source in
                       SOURCES + ['antlab/semantic_v2_demo.py', 'antlab/semantic_v2_crossplay.py']},
        checkpoint_hashes={f'seed-{seed}.pt': digest(study / f'seed-{seed}.pt') for seed in cfg['seeds']},
        dataset_file='data.json', dataset_sha256=digest(study / 'data.json'), pairs=pairs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
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
    print(json.dumps({
        'status': report['status'], 'dataset': report['dataset'], 'examples': report['examples'],
        'same_seed_mean_exact': report['same_seed_mean_exact'],
        'different_seed_mean_exact': report['different_seed_mean_exact'],
        'pairs': {key: value['exact_frame_accuracy'] for key, value in report['pairs'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
