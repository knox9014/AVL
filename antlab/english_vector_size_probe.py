"""Measure fixed vector payload on natural long English inputs, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from .english_connected_model import encode_local_texts
from .english_connected_smoke import load_model


SECTIONS = [
    'After dinner, Maya entered the study, placed her notebook beside the window, '
    'and prepared to review the notes from her afternoon meeting before going to bed.',
    'The room contains a wooden desk, two shelves of reference books, a blue chair, '
    'and a small lamp that is connected to a wall socket near the floor. '
    'A stack of letters lies beside the notebook, and the curtains are closed.',
    'Earlier that day, Maya had discussed a community garden with several neighbors. '
    'They compared the cost of seeds, the available space behind the library, '
    'and the amount of water that would be needed during the summer. '
    'She recorded the decisions in her notebook and promised to bring a revised '
    'schedule to the next meeting. These details describe her work rather than '
    'the current state of the lamp.',
    'Before beginning her review, she sorted the letters by date and placed the '
    'oldest ones in a folder. She then checked a shopping list, added bread and '
    'apples, and moved the list into her coat pocket. Outside, a bus passed the '
    'building while a neighbor walked a dog along the pavement. The study remained '
    'quiet, and Maya decided to leave the window closed because the evening air '
    'was cold. None of these actions changed the electrical switch on the desk '
    'lamp. For the question about its present state, use the final explicit fact '
    'rather than guessing from the time of day or from the other objects in the room.',
]


def probe(checkpoint_folder):
    torch.set_num_threads(2)
    model = load_model(checkpoint_folder)
    records = []
    for count in range(1, len(SECTIONS) + 1):
        prefix = ' '.join(SECTIONS[:count])
        rows = [['Question: Is the lamp on?', prefix + ' The lamp is ' + state + '.']
                for state in ('on', 'off')]
        tokens, lengths = encode_local_texts(rows)
        with torch.inference_mode():
            result = model.generate(tokens, lengths, rounds=2, max_new_tokens=8)
        pair = []
        for i, state in enumerate(('on', 'off')):
            message = result['messages'][i, 0, 1].detach().cpu().contiguous()
            payload = message.numpy().tobytes()
            original = len(rows[i][1].encode('utf-8'))
            size = len(payload)
            assert size == message.numel() * message.element_size() == 64
            pair.append(dict(fact_state=state, text=rows[i][1], original_utf8_bytes=original,
                             characters=len(rows[i][1]), message_dimensions=message.numel(),
                             message_dtype=str(message.dtype), message_bytes=size,
                             raw_size_ratio=original/size,
                             raw_size_reduction_percent=100*(1-size/original),
                             first_remote_message=message.tolist(), message_payload_hex=payload.hex(),
                             generated_text=result['texts'][i], generated_token_ids=result['token_ids'][i],
                             expected='YES' if state == 'on' else 'NO'))
        records.append(dict(section_count=count, cases=pair,
                            first_remote_message_pair_maxabs=float(
                                (result['messages'][0, 0, 1]-result['messages'][1, 0, 1]).abs().max()),
                            two_units_two_rounds_bytes_per_sample=result['logical_payload_bytes']//2))
    checkpoint = Path(checkpoint_folder) / 'checkpoint.pt'
    return dict(format='act-long-english-vector-size-1', training_steps=0,
                checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('english_connected_model.py', 'english_connected_smoke.py',
                                            'english_vector_size_probe.py')},
                interpretation='Raw representation-size comparison only. No lossless reconstruction, '
                               'semantic retention, compression codec, or network speed claim.',
                query=rows[0][0], records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('antlab/runs/english-body-smoke-20261005'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('probe output already exists')
    result = probe(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for record in result['records']:
        case = record['cases'][0]
        print(json.dumps({k: case[k] for k in ('original_utf8_bytes', 'message_bytes',
                                             'raw_size_ratio', 'raw_size_reduction_percent')}, allow_nan=False))


if __name__ == '__main__':
    main()
