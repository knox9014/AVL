"""Isolated-process engineering measurement of serial vs batched translation.

Use the same public model, threads, decoding settings and fixture in both modes.
Model loading and one common warm-up are excluded from measured inference.
This measures throughput and exact output changes, not semantic correctness.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

from .local_translation import M2M100Translator
from .multilingual_probe import ROUTES
from .semantic_v2_data import is_supported


def peak_working_set():
    if sys.platform != 'win32':
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak',
            'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak', 'private')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    stats = Counters()
    stats.cb = ctypes.sizeof(stats)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(stats), stats.cb):
        raise OSError('process memory measurement unavailable')
    return stats.peak


def measure(mode, model_directory, batch_size=4):
    fixture = Path('antlab/runs/semantic-v2-multilingual-probe-20261006.json')
    data = json.loads(fixture.read_text(encoding='utf-8'))
    backend = M2M100Translator(model_directory, batch_size=batch_size)
    load_started = time.perf_counter()
    backend._load()
    load_seconds = time.perf_counter()-load_started
    backend.translate(data['cases']['Korean'][0], 'ko', 'en')
    generation_calls = 0
    generate = backend._model.generate
    def counted_generate(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        return generate(*args, **kwargs)
    backend._model.generate = counted_generate
    groups = [('unique_'+language, code, data['cases'][language])
              for language, code in ROUTES.items() if code != 'en']
    groups.append(('duplicate_Korean', 'ko', data['cases']['Korean'] * 4))
    rows = []
    for name, language, texts in groups:
        before = generation_calls
        started = time.perf_counter()
        output = ([backend.translate(text, language, 'en') for text in texts] if mode == 'serial'
                  else backend.translate_many(texts, language, 'en'))
        rows.append(dict(group=name, source_lang=language, inputs=texts, translations=output,
                         seconds=time.perf_counter()-started, generation_calls=generation_calls-before,
                         nonempty_outputs=sum(value is not None and bool(value.strip()) for value in output),
                         avl_admitted_outputs=sum(value is not None and is_supported(value) for value in output)))
    root = Path(__file__).resolve().parents[1]
    import torch
    import transformers
    return dict(format='avl-translation-efficiency-v1', mode=mode, batch_size=batch_size,
                device='cpu', threads=2, num_beams=3, source_text_sent_to_remote_service=False,
                model_loading_seconds=load_seconds, warmup_excluded=True,
                torch=torch.__version__, transformers=transformers.__version__,
                fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                model_revision=json.loads((Path(model_directory)/'model_manifest.json').read_text(encoding='utf-8'))['revision'],
                peak_working_set_bytes=peak_working_set(),
                source_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                               ('antlab/local_translation.py', 'antlab/translation_efficiency.py')},
                rows=rows, semantic_accuracy='not measured; nonempty translation is not semantic success')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('serial', 'batch'), required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model', type=Path, default=Path('.translation-models/m2m100-418M'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new measurement output')
    result = measure(args.mode, args.model, args.batch_size)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(mode=args.mode, seconds=sum(row['seconds'] for row in result['rows']),
                         generation_calls=sum(row['generation_calls'] for row in result['rows']),
                         peak_working_set_bytes=result['peak_working_set_bytes']), indent=2))


if __name__ == '__main__':
    main()
