import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from antlab.local_translation import M2M100Translator, LANGUAGES
from antlab.translation_gateway import TranslationError


class LocalTranslationTests(unittest.TestCase):
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
