"""Post-hoc codec comparison on frozen AVL grammar; no training or gate changes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from . import semantic_codec as float_codec
from . import semantic_codec_int8 as int8_codec
from .semantic_v2_coverage import make_audit_data
from .semantic_v2_data import make_data
from .semantic_v2_demo import load_checkpoint
from .semantic_v2_run import ROOT, SOURCES, _batch, _labels, _mask, metrics, verify


@torch.inference_mode()
def compare(model, rows):
    predictions = {'float32': [], 'int8': []}
    wire = dict.fromkeys(predictions, 0)
    codec_seconds = dict.fromkeys(predictions, 0.)
    maximum_error = 0.
    for start in range(0, len(rows), 128):
        batch = rows[start:start+128]
        tokens, lengths = _batch(batch)
        vectors = model.encode(tokens, lengths)
        for name, codec in (('float32', float_codec), ('int8', int8_codec)):
            began = time.perf_counter()
            packet = codec.pack_vectors(vectors)
            received = codec.unpack_vectors(packet)
            codec_seconds[name] += time.perf_counter()-began
            wire[name] += len(packet)
            predictions[name].extend(_mask(model.decode(received)).argmax(-1).tolist())
            if name == 'int8':
                maximum_error = max(maximum_error, (vectors-received).abs().max().item())
    results = {}
    for name in predictions:
        result = metrics(_labels(rows), predictions[name])
        result.pop('predictions')
        result.update(wire_bytes=wire[name], codec_seconds=codec_seconds[name])
        results[name] = result
    return dict(examples=len(rows), results=results,
                changed_frame_predictions=sum(a != b for a, b in zip(predictions['float32'], predictions['int8'])),
                maximum_absolute_vector_error=maximum_error,
                wire_reduction_fraction=1-wire['int8']/wire['float32'])


def run(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError('use a new output path')
    verified = verify(study)
    if verified['smoke_only']:
        raise ValueError('full study required')
    audit = make_audit_data()
    splits = dict(joint=make_data()['joint'], all_grammar=audit['all'], unmeasured=audit['unmeasured'])
    torch.set_num_threads(2)
    started = time.perf_counter()
    seeds = {str(seed): {split: compare(load_checkpoint(study, seed), rows)
                       for split, rows in splits.items()} for seed in (44, 55, 66)}
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(format='avl-int8-codec-audit-v1', status='posthoc_codec_comparison',
                  preregistered=False, training_performed=False, changes_primary_gates=False,
                  scope='Frozen finite grammar only; all_grammar includes training examples. Lossy codec, not general semantic preservation.',
                  device='cpu', threads=2, torch=torch.__version__, elapsed_seconds=time.perf_counter()-started,
                  primary_verification=verified, receiver_input='packet_bytes_only_plus_fixed_field_vocabulary_mask',
                  payload_bytes_per_vector=dict(float32=64, int8=20), header_bytes=12,
                  source_hashes={source: digest(ROOT/source) for source in SOURCES +
                      ['antlab/semantic_codec_int8.py', 'antlab/semantic_int8_audit.py',
                       'antlab/semantic_v2_coverage.py', 'antlab/semantic_v2_demo.py']},
                  checkpoint_hashes={f'seed-{seed}.pt': digest(study/f'seed-{seed}.pt') for seed in (44, 55, 66)},
                  seeds=seeds)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.study, args.output)
    print(json.dumps({seed: {split: dict(float32=r['results']['float32']['exact_frame_accuracy'],
                   int8=r['results']['int8']['exact_frame_accuracy'], changed=r['changed_frame_predictions'],
                   wire_reduction=r['wire_reduction_fraction']) for split, r in splits.items()}
                   for seed, splits in report['seeds'].items()}, indent=2))


if __name__ == '__main__':
    main()
