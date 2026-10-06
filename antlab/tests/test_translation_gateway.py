"""Gateway boundary tests use fake translators, not multilingual validation."""
import unittest
from unittest.mock import patch

import torch

from antlab.semantic_codec import unpack_vectors
from antlab.semantic_v2_model import SemanticNet
from antlab.translation_gateway import TranslationError, TranslationGateway


class FakeTranslator:
    """Deterministic test double; no real translation ability is claimed."""
    name = 'fake-test-translator'

    def __init__(self, translations=None, pairs=None):
        self.translations = translations or {}
        self.pairs = pairs if pairs is not None else {('ko', 'en'), ('en', 'ko')}
        self.calls = []

    def supports(self, source_lang, target_lang):
        return (source_lang, target_lang) in self.pairs

    def translate(self, text, source_lang, target_lang):
        self.calls.append((text, source_lang, target_lang))
        result = self.translations[(text, source_lang, target_lang)]
        if isinstance(result, Exception):
            raise result
        return result


FRAME = dict(kind='fact', subject='lamp', predicate='on', polarity='negative',
             certainty='possible', condition='rain', amount='none', comparator='none', unit='none')
CANONICAL = 'When it rains, it is possible that the lamp is not on.'


class TranslationGatewayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(44)
        self.model = SemanticNet().eval()

    def test_duplicate_segments_translate_once_but_keep_every_record(self):
        backend = FakeTranslator({('원문', 'ko', 'en'): CANONICAL})
        gateway = TranslationGateway(self.model, backend)
        sent = gateway.encode(['원문', '원문'], 'ko')
        self.assertEqual(sent['translations'], [CANONICAL, CANONICAL])
        self.assertEqual(sent['supported_indices'], [0, 1])
        self.assertEqual(unpack_vectors(sent['packet']).shape, (2, 16))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(sent['metadata']['translation_reused_segments'], 1)
        gateway.encode(['원문'], 'ko')
        self.assertEqual(len(backend.calls), 2)  # No persistent source-text cache.

    def test_optional_batch_backend_keeps_order_and_per_segment_failure(self):
        class BatchBackend:
            name = 'test-batch'
            def supports(self, source, target):
                return True
            def translate_many(self, texts, source, target):
                return [CANONICAL if text == '성공' else None for text in texts]
            def translate(self, *args):
                raise AssertionError('scalar path would waste work')
        sent = TranslationGateway(self.model, BatchBackend()).encode(['성공', '실패', '성공'], 'ko')
        self.assertEqual([r['status'] for r in sent['records']], ['encoded', 'translation_unavailable', 'encoded'])
        self.assertEqual(sent['translations'], [CANONICAL, None, CANONICAL])
        self.assertEqual(sent['metadata']['translation_batch_calls'], 1)
        self.assertEqual(sent['metadata']['translation_calls'], 1)
        self.assertEqual(sent['sources'], ['성공', '실패', '성공'])

    def test_target_duplicate_frames_reuse_request_local_translation(self):
        backend = FakeTranslator({(CANONICAL, 'en', 'ko'): '번역'})
        gateway = TranslationGateway(self.model, backend)
        packet = gateway.encode([CANONICAL, CANONICAL], 'en')['packet']
        with patch('antlab.translation_gateway.receive_packet', return_value=[dict(FRAME), dict(FRAME)]):
            result = gateway.decode(packet, 'ko')
        self.assertEqual(result['texts'], ['번역', '번역'])
        self.assertEqual(len(backend.calls), 1)

    def test_original_translation_retention_and_conservative_admission(self):
        originals = ['  비가 오면 램프가 켜져 있지 않을 수도 있다.\n', '예산 17달러', '램프가 켜져 있다.']
        translated = [CANONICAL, 'It is certain that the budget limit is exactly 17 USD.', 'The lamp is on.']
        backend = FakeTranslator({(source, 'ko', 'en'): target for source, target in zip(originals, translated)})
        sent = TranslationGateway(self.model, backend).encode(originals, source_lang='ko')
        self.assertEqual(sent['sources'], originals)
        self.assertEqual(sent['translations'], translated)
        self.assertEqual(sent['supported_indices'], [0])
        self.assertEqual([row['status'] for row in sent['records']], ['encoded', 'out_of_scope', 'out_of_scope'])
        self.assertEqual(unpack_vectors(sent['packet']).shape, (1, 16))
        self.assertEqual(backend.calls, [(text, 'ko', 'en') for text in originals])
        self.assertEqual(sent['metadata']['translation_calls'], 3)
        self.assertEqual(sent['records'][1]['translation'], translated[1])

    def test_english_bypass_and_no_out_of_scope_vector(self):
        backend = FakeTranslator()
        gateway = TranslationGateway(self.model, backend)
        sources = [CANONICAL, 'The lamp is on.', 'Unlisted entity is off.']
        sent = gateway.encode(sources, 'en')
        self.assertEqual(backend.calls, [])
        self.assertEqual(sent['translations'], sources)
        self.assertEqual(sent['supported_indices'], [0])
        self.assertEqual(sent['metadata']['translation_calls'], 0)
        self.assertIsNone(gateway.encode(sources[1:], 'en')['packet'])

    def test_unavailable_languages_and_expected_errors_preserve_source(self):
        source = '예산은 20달러 이하로 해 주세요.'
        sent = TranslationGateway(self.model).encode([source], 'ko')
        self.assertEqual(sent['records'][0]['status'], 'translation_unavailable')
        self.assertEqual(sent['sources'], [source])
        self.assertIsNone(sent['packet'])
        sent = TranslationGateway(self.model, FakeTranslator(pairs=set())).encode([source], 'xx')
        self.assertEqual(sent['records'][0]['status'], 'unsupported_language')
        backend = FakeTranslator({(source, 'ko', 'en'): TranslationError('sensitive backend details')})
        sent = TranslationGateway(self.model, backend).encode([source], 'ko')
        self.assertEqual(sent['records'][0]['status'], 'translation_unavailable')
        self.assertNotIn('sensitive', repr(sent))
        backend.translations[(source, 'ko', 'en')] = RuntimeError('programming failure')
        with self.assertRaises(RuntimeError):
            TranslationGateway(self.model, backend).encode([source], 'ko')

    def test_empty_invalid_translation_does_not_fabricate_a_vector(self):
        for output in ('', ' \n ', None, 123):
            backend = FakeTranslator({('입력', 'ko', 'en'): output})
            sent = TranslationGateway(self.model, backend).encode(['입력'], 'ko')
            self.assertEqual(sent['records'][0]['status'], 'translation_unavailable')
            self.assertIsNone(sent['packet'])

    def test_decoder_receives_only_packet_and_translates_predicted_rendering(self):
        target = '비가 오면 램프가 켜져 있지 않을 수도 있다.'
        backend = FakeTranslator({(CANONICAL, 'en', 'ko'): target})
        gateway = TranslationGateway(self.model, backend)
        packet = gateway.encode([CANONICAL], 'en')['packet']
        with patch('antlab.translation_gateway.receive_packet', return_value=[dict(FRAME)]) as receive:
            result = gateway.decode(packet, 'ko')
        receive.assert_called_once_with(self.model, packet)
        self.assertEqual(result['frames'], [FRAME])
        self.assertEqual(result['canonical_english'], [CANONICAL])
        self.assertEqual(result['texts'], [target])
        self.assertEqual(result['records'][0]['status'], 'model_prediction')
        self.assertEqual(backend.calls, [(CANONICAL, 'en', 'ko')])
        self.assertNotIn('sources', result)
        self.assertEqual(result['metadata']['translation_calls'], 1)

    def test_target_unavailable_falls_back_to_english_and_invalid_frames_stop(self):
        gateway = TranslationGateway(self.model)
        packet = gateway.encode([CANONICAL], 'en')['packet']
        with patch('antlab.translation_gateway.receive_packet', return_value=[dict(FRAME)]):
            result = gateway.decode(packet, 'ko')
        self.assertEqual(result['texts'], [CANONICAL])
        self.assertEqual(result['records'][0]['status'], 'target_translation_unavailable')
        for backend in (FakeTranslator(pairs=set()),
                        FakeTranslator({(CANONICAL, 'en', 'ko'): TranslationError('private details')})):
            with patch('antlab.translation_gateway.receive_packet', return_value=[dict(FRAME)]):
                result = TranslationGateway(self.model, backend).decode(packet, 'ko')
            self.assertEqual(result['records'][0]['status'], 'target_translation_unavailable')
            self.assertEqual(result['records'][0]['output_lang'], 'en')
            self.assertEqual(result['texts'], [CANONICAL])
            self.assertNotIn('private details', repr(result))
        bypass = FakeTranslator()
        with patch('antlab.translation_gateway.receive_packet', return_value=[dict(FRAME)]):
            result = TranslationGateway(self.model, bypass).decode(packet, 'en')
        self.assertEqual(bypass.calls, [])
        self.assertEqual(result['records'][0]['status'], 'model_prediction')
        bad = dict(FRAME, predicate='limit')
        with patch('antlab.translation_gateway.receive_packet', return_value=[bad]):
            result = gateway.decode(packet, 'en')
        self.assertEqual(result['records'][0]['status'], 'invalid_semantic_prediction')
        self.assertEqual(result['texts'], [None])
        self.assertEqual(result['frames'], [bad])

    def test_invalid_input_and_packets_fail_explicitly(self):
        gateway = TranslationGateway(self.model)
        for texts in ([], CANONICAL, [None]):
            with self.assertRaises(ValueError):
                gateway.encode(texts, 'en')
        for language in ('', '   ', None, 123):
            with self.assertRaises(ValueError):
                gateway.encode([CANONICAL], language)
            with self.assertRaises(ValueError):
                gateway.decode(b'bad', language)
        with self.assertRaises(ValueError):
            gateway.decode(b'bad', 'en')


if __name__ == '__main__':
    unittest.main()
