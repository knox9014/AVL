import unittest
from unittest.mock import patch

import torch

from antlab.semantic_v2_crossplay import evaluate_pair


class FakeModel:
    def __init__(self, marker):
        self.marker = marker

    def encode(self, tokens, lengths):
        return torch.full((len(tokens), 16), float(self.marker), dtype=torch.float32)

    def decode(self, vectors):
        logits = torch.zeros(len(vectors), 9, 8)
        for i, vector in enumerate(vectors):
            logits[i, :, 0 if vector[0].item() == self.marker else 1] = 1
        return logits


class SemanticV2CrossplayTests(unittest.TestCase):
    def test_crossplay_uses_sender_vectors_and_independent_receiver(self):
        rows = [{'text': 'fixture', 'labels': [0] * 9} for _ in range(3)]
        def batch(samples):
            return torch.zeros(len(samples), 1, dtype=torch.long), torch.ones(len(samples), dtype=torch.long)
        with patch('antlab.semantic_v2_crossplay._batch', side_effect=batch):
            same = evaluate_pair(FakeModel(44), FakeModel(44), rows, batch_size=2)
            different = evaluate_pair(FakeModel(44), FakeModel(55), rows, batch_size=2)
        self.assertEqual(same['exact_frame_accuracy'], 1)
        self.assertEqual(different['exact_frame_accuracy'], 0)
        self.assertEqual(same['wire_bytes'], 3 * 64 + 2 * 12)
        self.assertEqual(same['packets'], 2)
        self.assertEqual(set(same['fields']), {'kind', 'subject', 'predicate', 'polarity', 'certainty', 'condition', 'amount', 'comparator', 'unit'})
        self.assertNotIn('predictions', same)
        self.assertTrue(all(f['accuracy'] == 0 for f in different['fields'].values()))


if __name__ == '__main__':
    unittest.main()
