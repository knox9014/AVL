"""Explicit public-model download. No translation or user text is sent."""
import argparse
import hashlib
import json
from pathlib import Path
from .local_translation import MODEL_ID, optional_dependencies

FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.json', 'sentencepiece.bpe.model',
         'pytorch_model.bin')


def download(output, revision=None):
    optional_dependencies()
    from huggingface_hub import HfApi, snapshot_download
    output = Path(output)
    info = HfApi(token=False).model_info(MODEL_ID, revision=revision, files_metadata=True)
    pinned = info.sha
    snapshot_download(MODEL_ID, revision=pinned, local_dir=output,
                      allow_patterns=list(FILES) + ['README.md'], token=False,
                      max_workers=2)
    record = dict(model_id=MODEL_ID, revision=pinned, declared_languages=100,
                  license_from_card='MIT', model_card='https://huggingface.co/' + MODEL_ID,
                  files={})
    for name in FILES:
        path = output / name
        if not path.exists():
            if name in ('generation_config.json', 'special_tokens_map.json'):
                continue
            raise ValueError('missing required public model file: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        record['files'][name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    record['total_bytes'] = sum(v['bytes'] for v in record['files'].values())
    (output / 'model_manifest.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.translation-models/m2m100-418M'))
    parser.add_argument('--revision')
    args = parser.parse_args()
    print(json.dumps(download(args.output, args.revision), indent=2))


if __name__ == '__main__':
    main()
