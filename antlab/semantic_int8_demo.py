"""Experimental int8 wire demo on the frozen controlled English grammar."""
import argparse
import json
from pathlib import Path
import torch
from .semantic_codec_int8 import pack_vectors, unpack_vectors
from .semantic_v2_data import FIELDS, VOCABS, is_supported
from .semantic_v2_model import tokenize
from .semantic_v2_demo import load_checkpoint
from .semantic_v2_run import _mask


@torch.inference_mode()
def receive(model, packet):
    """Receiver input is bytes only; fixed field vocabularies are shared."""
    model.eval()
    predictions = _mask(model.decode(unpack_vectors(packet))).argmax(-1).tolist()
    return [{field: vocab[index] for field, vocab, index in zip(FIELDS, VOCABS, row)}
            for row in predictions]


@torch.inference_mode()
def transmit(model, texts):
    if not isinstance(texts, list) or not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError('supply a nonempty list of source strings')
    model.eval()
    supported = [index for index, text in enumerate(texts) if is_supported(text)]
    packet = None
    frames = []
    if supported:
        tokens, lengths = tokenize([texts[index] for index in supported])
        packet = pack_vectors(model.encode(tokens, lengths))
        frames = receive(model, packet)
    by_index = dict(zip(supported, frames))
    records = [dict(index=index, source=text, status='model_prediction', meaning=by_index[index])
               if index in by_index else dict(index=index, source=text, status='unsupported')
               for index, text in enumerate(texts)]
    return dict(records=records, wire_bytes=len(packet) if packet else 0,
                payload_bytes=20*len(frames), codec='AVQ1-experimental-lossy-int8',
                scope='Finite grammar, same checkpoint only. Original wording is not restored.',
                source_retained_locally=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--seed', type=int, choices=(44, 55, 66), default=44)
    parser.add_argument('--text', action='append', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(transmit(load_checkpoint(args.checkpoint, args.seed), args.text),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
