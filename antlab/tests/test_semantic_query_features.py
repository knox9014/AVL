import unittest
import torch

from antlab.semantic_query_features import features
from antlab.semantic_query_pilot import QueryReceiver, query_features


class FeatureTests(unittest.TestCase):
    def test_raw_features_retain_vector_without_extra_information(self):
        value = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        result = features(None, value, 'raw_padded')
        self.assertTrue(torch.equal(result[:, :16], value))
        self.assertEqual(result[:, 16:].count_nonzero(), 0)
        self.assertEqual(result.shape, (2, 72))

    def test_probability_features_are_label_free_and_normalized(self):
        class Decoder:
            def decode(self, vectors):
                return vectors[:, :1, None].expand(-1, 9, 8)
        result = features(Decoder(), torch.zeros(2, 16), 'frozen_probabilities')
        self.assertTrue(torch.equal(result, torch.full((2, 72), 1 / 8)))
        self.assertTrue(torch.equal(result.reshape(2, 9, 8).sum(-1), torch.ones(2, 9)))
        with self.assertRaises(ValueError):
            features(None, torch.zeros(1, 16), 'unknown')

    def test_matched_receiver_capacity_and_output(self):
        receiver = QueryReceiver(vector_width=72)
        self.assertEqual(sum(p.numel() for p in receiver.parameters()), 10349)
        ids, candidates = query_features(1)
        result = receiver(torch.zeros(24, 72), ids, candidates)
        self.assertEqual(result.shape, (24, 13))
        with self.assertRaises(ValueError):
            receiver(torch.zeros(24, 16), ids, candidates)


if __name__ == '__main__':
    unittest.main()
