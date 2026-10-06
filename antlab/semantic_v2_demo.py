"""Experimental v2 word-to-vector interface, with explicit unsupported fallback."""
import argparse
import json
from pathlib import Path
import torch
from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_v2_data import FIELDS, VOCABS, is_supported
from .semantic_v2_model import SemanticNet, tokenize


@torch.inference_mode()
def transmit_segments(model, texts):
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
        raise ValueError('supply a nonempty list of original source segments')
    supported = [i for i, text in enumerate(texts) if is_supported(text)]
    unsupported = [dict(index=i, status='unsupported', text=text)
                   for i, text in enumerate(texts) if i not in supported]
    packet = None
    if supported:
        tokens, lengths = tokenize([texts[i] for i in supported])
        packet = pack_vectors(model.encode(tokens, lengths))
    return dict(sources=list(texts), supported_indices=supported,
                unsupported=unsupported, packet=packet)


@torch.inference_mode()
def receive_packet(model, packet):
    # The receiver never receives source text, admission results or target labels.
    predictions = model.decode(unpack_vectors(packet)).argmax(-1).tolist()
    return [{field: vocab[index] for field, vocab, index in zip(FIELDS, VOCABS, row)}
            for row in predictions]


def load_checkpoint(folder, seed=44):
    checkpoint = torch.load(Path(folder) / f'seed-{seed}.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'avl-semantic-study-v2' or checkpoint.get('seed') != seed:
        raise ValueError('unsupported checkpoint format or seed')
    if checkpoint.get('config', {}).get('smoke_only') is not False:
        raise ValueError('full v2 study checkpoint required')
    model = SemanticNet()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint weights')
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--seed', type=int, default=44, choices=(44, 55, 66))
    parser.add_argument('--text', action='append', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_checkpoint(args.checkpoint, args.seed)
    sent = transmit_segments(model, args.text)
    frames = receive_packet(model, sent['packet']) if sent['packet'] is not None else []
    records = [dict(index=i, status='model_prediction', source=sent['sources'][i], meaning=frame)
               for i, frame in zip(sent['supported_indices'], frames)] + sent['unsupported']
    records.sort(key=lambda row: row['index'])
    print(json.dumps(dict(records=records, wire_bytes=len(sent['packet']) if sent['packet'] else 0,
                          payload_bytes=64 * len(frames), sources_retained_locally=True,
                          scope='Finite v2 grammar only; model predictions are not verified general understanding.'),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
