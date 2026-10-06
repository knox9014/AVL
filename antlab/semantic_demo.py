"""Controlled grammar demo with explicit unsupported-input fallback."""
import argparse
import json
from pathlib import Path

import torch
from .english_connected_model import encode_local_texts
from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_data import FIELDS, VOCABS, is_supported, normalize_text
from .semantic_model import SemanticNet


@torch.inference_mode()
def transmit_segments(model, texts):
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
        raise ValueError('supply a nonempty list of source segments')
    supported = [i for i, text in enumerate(texts) if is_supported(text)]
    unsupported = [dict(index=i, status='unsupported', text=text)
                   for i, text in enumerate(texts) if i not in supported]
    packet = None
    if supported:
        tokens, lengths = encode_local_texts([[normalize_text(texts[i])] for i in supported])
        packet = pack_vectors(model.encode(tokens[:, 0], lengths[:, 0]))
    return dict(sources=list(texts), supported_indices=supported, unsupported=unsupported, packet=packet)


@torch.inference_mode()
def receive_packet(model, packet):
    # This function never sees source text, its IDs, or gold semantic labels.
    vectors = unpack_vectors(packet)
    predictions = model.decode(vectors).argmax(-1).tolist()
    return [{field: vocab[index] for field, vocab, index in zip(FIELDS, VOCABS, row)}
            for row in predictions]


def load_checkpoint(folder, seed=11):
    checkpoint = torch.load(Path(folder) / f'seed-{seed}.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'avl-semantic-study-v1' or checkpoint.get('seed') != seed:
        raise ValueError('unsupported checkpoint')
    if checkpoint.get('config', {}).get('smoke_only') is not False:
        raise ValueError('a full study checkpoint is required for this demo')
    model = SemanticNet()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint')
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('antlab/runs/semantic-v1-20261006'))
    parser.add_argument('--seed', type=int, default=11, choices=(11, 22, 33))
    parser.add_argument('--text', action='append', required=True, help='Repeat for explicitly separated segments.')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_checkpoint(args.checkpoint, args.seed)
    sent = transmit_segments(model, args.text)
    received = receive_packet(model, sent['packet']) if sent['packet'] is not None else []
    records = [dict(index=i, status='model_prediction', source=sent['sources'][i], meaning=frame)
               for i, frame in zip(sent['supported_indices'], received)]
    records.extend(sent['unsupported'])
    records.sort(key=lambda r: r['index'])
    print(json.dumps(dict(records=records, wire_bytes=len(sent['packet']) if sent['packet'] else 0,
                          payload_bytes=64*len(received), sources_retained_locally=True,
                          scope='Controlled grammar only; predictions are not a general understanding guarantee.'),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
