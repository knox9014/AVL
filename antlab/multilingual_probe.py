"""Post-hoc synthetic translation/gateway probe, not a multilingual benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .local_translation import M2M100Translator
from .semantic_v2_demo import load_checkpoint
from .semantic_render import render_meaning
from .translation_gateway import TranslationGateway, TranslationError

ROUTES = dict(English='en', Korean='ko', Japanese='ja', Chinese='zh', Spanish='es',
              French='fr', German='de', Russian='ru', Arabic='ar')
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('antlab/local_translation.py', 'antlab/translation_gateway.py',
           'antlab/multilingual_probe.py', 'antlab/semantic_render.py',
           'antlab/semantic_v2_demo.py', 'antlab/semantic_v2_model.py')


def probe(model_directory, checkpoint, fixture):
    fixture = Path(fixture)
    source = json.loads(fixture.read_text(encoding='utf-8'))
    # Gold frames are used only for this post-hoc comparison, never as encoder
    # inputs, translation constraints or receiver side information.
    expected = source['expected_for_english_control_only']
    translator = M2M100Translator(model_directory)
    gateway = TranslationGateway(load_checkpoint(checkpoint, seed=44), translator)
    started = time.perf_counter()
    rows = []
    for language, code in ROUTES.items():
        sent = gateway.encode(source['cases'][language], code)
        received = gateway.decode(sent['packet'], 'en') if sent['packet'] is not None else None
        predictions = dict(zip(sent['supported_indices'], received['frames'])) if received else {}
        records = []
        for record in sent['records']:
            index = record['index']
            frame = predictions.get(index)
            records.append(dict(**record, predicted_frame=frame, intended_frame=expected[index],
                                exact_frame_if_encoded=frame == expected[index] if frame is not None else None))
        rows.append(dict(language=language, code=code, records=records,
                         sender_metadata=sent['metadata']))
    reverse = []
    for code in ('ko', 'ja', 'es'):
        for frame in expected:
            english = render_meaning(frame)
            try:
                text = translator.translate(english, 'en', code)
                status = 'translation_prediction'
            except TranslationError:
                text, status = None, 'translation_unavailable'
            reverse.append(dict(source=english, target_lang=code, translation=text,
                                status=status, semantic_preservation='not automatically verified'))
    foreign = [record for row in rows if row['code'] != 'en' for record in row['records']]
    admitted = [r for r in foreign if r['status'] == 'encoded']
    correct = sum(r['exact_frame_if_encoded'] is True for r in admitted)
    model_manifest = json.loads((Path(model_directory) / 'model_manifest.json').read_text(encoding='utf-8'))
    import torch
    import transformers
    import sentencepiece
    return dict(format='avl-multilingual-gateway-probe-v1', status='post_hoc_illustrative_probe',
                scope='Manually authored synthetic parallel examples; no native-speaker benchmark or all-language claim.',
                training_performed=False, avl_seed=44, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                source_hashes={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in SOURCES},
                avl_checkpoint_sha256=hashlib.sha256((Path(checkpoint) / 'seed-44.pt').read_bytes()).hexdigest(),
                translation_model=model_manifest, translation_parameters=sum(p.numel() for p in translator._model.parameters()),
                runtime=dict(device='cpu', threads=2, seconds=time.perf_counter()-started,
                             torch=torch.__version__, transformers=transformers.__version__, sentencepiece=sentencepiece.__version__),
                source_text_sent_to_remote_service=False, source_retained=True,
                foreign_summary=dict(languages=8, inputs=len(foreign),
                    successful_translation_calls=sum(r['translation_status'] == 'translated' for r in foreign),
                    status_counts=dict(Counter(r['status'] for r in foreign)),
                    admitted_inputs=len(admitted), exact_frames_on_admitted=correct,
                    exact_frames_per_all_inputs=correct/len(foreign),
                    exact_frame_accuracy_on_admitted=correct/len(admitted) if admitted else None,
                    translation_semantic_accuracy='not scored by a native-speaker or reference translation metric'),
                language_rows=rows, reverse_translation_predictions=reverse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('.translation-models/m2m100-418M'))
    parser.add_argument('--checkpoint', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--fixture', type=Path, default=Path('antlab/runs/semantic-v2-multilingual-probe-20261006.json'))
    parser.add_argument('--output', type=Path, default=Path('antlab/runs/multilingual-gateway-20261006.json'))
    args = parser.parse_args()
    result = probe(args.model, args.checkpoint, args.fixture)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['foreign_summary'], indent=2))


if __name__ == '__main__':
    main()
