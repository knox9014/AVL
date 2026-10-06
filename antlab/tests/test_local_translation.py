import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from antlab.local_translation import M2M100Translator, LANGUAGES
from antlab.translation_gateway import TranslationError
import torch


class LocalTranslationTests(unittest.TestCase):
    def test_batch_memory_failure_is_expected_but_programming_error_propagates(self):
        class Tokenizer:
            pad_token_id = 1
            def __call__(self, texts, **kwargs):
                return {'input_ids': [[10, 2] for _ in texts]}
            def pad(self, rows, **kwargs):
                return {'input_ids': torch.tensor([row['input_ids'] for row in rows])}
            def get_lang_id(self, language):
                return 77
        backend = M2M100Translator('unused')
        backend._tokenizer = Tokenizer()
        def fail(*args, **kwargs):
            raise torch.OutOfMemoryError('private resource details')
        backend._model = SimpleNamespace(generate=fail)
        self.assertEqual(backend.translate_many(['원문'], 'ko', 'en'), [None])
        def bug(*args, **kwargs):
            raise RuntimeError('programming error')
        backend._model.generate = bug
        with self.assertRaises(RuntimeError):
            backend.translate_many(['원문'], 'ko', 'en')

    def test_batch_preserves_order_rejects_long_rows_and_ignores_output_padding(self):
        generated = []
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                ids = {'short': [10, 2], 'long': [11] * 6, 'other': [12, 2], 'limit': [13, 2]}
                return {'input_ids': [ids[text] for text in texts],
                        'attention_mask': [[1] * len(ids[text]) for text in texts]}
            def pad(self, rows, **kwargs):
                length = max(len(row['input_ids']) for row in rows)
                return {key: torch.tensor([row[key] + [0] * (length-len(row[key])) for row in rows])
                        for key in ('input_ids', 'attention_mask')}
            def get_lang_id(self, language):
                return 77
            pad_token_id = 1
            def batch_decode(self, rows, **kwargs):
                return [f'translation-{row[2]}' for row in rows.tolist()]
        def generate(input_ids, **kwargs):
            generated.extend(input_ids[:, 0].tolist())
            return torch.tensor([[2, 77, marker, marker, marker, 2] if marker == 13
                                 else [2, 77, marker, 2, 1, 1] for marker in input_ids[:, 0].tolist()])
        backend = M2M100Translator('unused', max_input_tokens=4, max_new_tokens=5)
        backend._tokenizer = Tokenizer()
        backend._model = SimpleNamespace(generate=generate)
        self.assertTrue(callable(getattr(backend, 'translate_many', None)), 'batch translation missing')
        result = backend.translate_many(['short', 'long', 'other', 'short', 'limit'], 'ko', 'en')
        self.assertEqual(result, ['translation-10', None, 'translation-12', 'translation-10', None])
        self.assertEqual(generated, [10, 12, 13])

    def test_identity_batch_never_loads_model_and_validates_count(self):
        backend = M2M100Translator('nonexistent')
        self.assertTrue(callable(getattr(backend, 'translate_many', None)), 'batch translation missing')
        self.assertEqual(backend.translate_many(['하나', '둘'], 'ko', 'ko'), ['하나', '둘'])
        self.assertEqual(backend.translate_many([], 'ko', 'en'), [])
        with self.assertRaises(TranslationError):
            backend.translate_many(['하나'], 'unknown', 'en')
    def test_corrupt_local_files_preserve_expected_failure_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('config.json', 'vocab.json', 'sentencepiece.bpe.model', 'pytorch_model.bin'):
                (Path(folder) / name).write_bytes(b'broken')
            for error in (OSError('private path must not be exposed'), ValueError('bad config')):
                tokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: (_ for _ in ()).throw(error))
                fake = SimpleNamespace(M2M100Tokenizer=tokenizer, M2M100ForConditionalGeneration=object())
                backend = M2M100Translator(folder)
                with patch('antlab.local_translation.optional_dependencies'), patch.dict('sys.modules', {'transformers': fake}):
                    with self.assertRaises(TranslationError):
                        backend.translate('안녕하세요.', 'ko', 'en')
                self.assertIsNone(backend._model)
                self.assertIsNone(backend._tokenizer)

    def test_declared_routes_and_local_only_unavailable(self):
        self.assertEqual(len(LANGUAGES), 100)
        with tempfile.TemporaryDirectory() as folder:
            backend = M2M100Translator(Path(folder))
            self.assertTrue(backend.supports('ko', 'en'))
            self.assertTrue(backend.supports('en', 'ja'))
            self.assertFalse(backend.supports('unknown', 'en'))
            self.assertFalse(backend.supports('en', 'unknown'))
            with self.assertRaises(TranslationError):
                backend.translate('안녕하세요.', 'ko', 'en')

    def test_no_download_for_identity_and_bad_inputs(self):
        backend = M2M100Translator('nonexistent-local-model')
        text = '원문 17 USD.'
        self.assertEqual(backend.translate(text, 'ko', 'ko'), text)
        for text, source, target in [('', 'ko', 'en'), ('hello', 'unknown', 'en')]:
            with self.assertRaises(TranslationError):
                backend.translate(text, source, target)
        with self.assertRaises(ValueError):
            M2M100Translator('unused', max_input_tokens=0)


if __name__ == '__main__':
    unittest.main()
