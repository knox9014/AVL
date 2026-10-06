import unittest
import torch
from antlab.semantic_model import SemanticNet
from antlab.semantic_demo import transmit_segments, receive_packet


class SemanticDemoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.model = SemanticNet().eval()

    def test_supported_sequence_and_unsupported_source_retained(self):
        texts = ['It is certain that the lamp is on.', 'Unknown text about a spaceship.',
                 'When it rains, it is possible that the door is not closed.']
        sent = transmit_segments(self.model, texts)
        self.assertEqual(sent['sources'], texts)
        self.assertEqual(sent['supported_indices'], [0, 2])
        self.assertEqual(sent['unsupported'], [{'index': 1, 'status': 'unsupported', 'text': texts[1]}])
        self.assertEqual(len(sent['packet']), 12+2*64)
        received = receive_packet(self.model, sent['packet'])
        self.assertEqual(len(received), 2)
        self.assertEqual(set(received[0]), {'kind', 'subject', 'predicate', 'polarity', 'certainty',
                                         'condition', 'amount', 'comparator', 'unit'})

    def test_case_and_whitespace_same_packet_originals_unchanged(self):
        text = 'It is certain that the lamp is on.'
        variant = ' IT\tis CERTAIN that\n the LAMP is ON. '
        a, b = transmit_segments(self.model, [text]), transmit_segments(self.model, [variant])
        self.assertEqual(a['packet'], b['packet'])
        self.assertEqual(b['sources'], [variant])

    def test_all_unsupported_has_no_fake_vector_and_invalid_empty_rejected(self):
        result = transmit_segments(self.model, ['The budget is 17 USD.'])
        self.assertIsNone(result['packet'])
        self.assertEqual(result['supported_indices'], [])
        for invalid in ([], 'a', [None]):
            with self.assertRaises(ValueError):
                transmit_segments(self.model, invalid)


if __name__ == '__main__':
    unittest.main()
