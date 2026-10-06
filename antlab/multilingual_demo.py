"""Local translation and optional AVL English-pivot demo. Explicit languages."""
import argparse
import json
from pathlib import Path
from .local_translation import M2M100Translator, LANGUAGES
from .semantic_v2_demo import load_checkpoint
from .translation_gateway import TranslationGateway, TranslationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('.translation-models/m2m100-418M'))
    parser.add_argument('--checkpoint', type=Path, default=Path('antlab/runs/semantic-v2-20261006'))
    parser.add_argument('--source-lang', default='ko')
    parser.add_argument('--target-lang', default='en')
    parser.add_argument('--text', action='append')
    parser.add_argument('--translation-only', action='store_true')
    parser.add_argument('--list-languages', action='store_true')
    args = parser.parse_args()
    if args.list_languages:
        print(json.dumps(dict(declared_languages=list(LANGUAGES), quality='not guaranteed for every language'), indent=2))
        return
    if not args.text:
        parser.error('supply at least one --text or use --list-languages')
    backend = M2M100Translator(args.model)
    if args.translation_only:
        records = []
        for text in args.text:
            if not backend.supports(args.source_lang, args.target_lang):
                records.append(dict(source=text, status='unsupported_language', translation=None))
                continue
            try:
                translated = backend.translate(text, args.source_lang, args.target_lang)
                records.append(dict(source=text, status='translation_prediction', translation=translated))
            except TranslationError:
                records.append(dict(source=text, status='translation_unavailable', translation=None))
        result = dict(records=records, source_lang=args.source_lang, target_lang=args.target_lang,
                      backend=backend.name, source_text_sent_to_remote_service=False)
    else:
        gateway = TranslationGateway(load_checkpoint(args.checkpoint, seed=44), backend)
        sent = gateway.encode(args.text, args.source_lang)
        result = dict(sender_records=sent['records'], sender_metadata=sent['metadata'],
                      receiver=None, source_text_sent_to_remote_service=False)
        if sent['packet'] is not None:
            result['receiver'] = gateway.decode(sent['packet'], args.target_lang)
        result['scope'] = 'Translation is separate from finite AVL meaning prediction; out-of-scope input is retained.'
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
