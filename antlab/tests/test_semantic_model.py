import unittest
from unittest.mock import patch
import torch
from antlab.english_connected_model import encode_local_texts
from antlab.semantic_model import SemanticNet


class SemanticModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.model = SemanticNet()
        t, l = encode_local_texts([['The lamp is on.'], ['The lamp is not on.']])
        self.tokens, self.lengths = t[:, 0], l[:, 0]

    def test_count_shapes_and_valid_classes(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 98511)
        out = self.model(self.tokens, self.lengths)
        self.assertEqual(out['vectors'].shape, (2, 16))
        self.assertEqual(out['logits'].shape, (2, 9, 8))
        for field, count in enumerate((2, 5, 5, 2, 2, 4, 5, 4, 2)):
            self.assertTrue(torch.isfinite(out['logits'][:, field, :count]).all())
            self.assertTrue((out['logits'][:, field, count:] < -1000).all())

    def test_invalid_classes_cannot_win_for_large_negative_valid_scores(self):
        with patch.object(self.model.receiver, 'forward', return_value=torch.full((1, 9, 8), -1e6)):
            scores = self.model.decode(torch.zeros(1, 16))
        self.assertTrue(torch.equal(scores.argmax(-1), torch.zeros(1, 9, dtype=torch.long)))

    def test_padding_independent_and_gradient_reaches_encoder_sender_receiver(self):
        expected = self.model(self.tokens, self.lengths)
        padded = torch.cat([self.tokens, torch.full((2, 7), 88)], 1)
        torch.testing.assert_close(expected['logits'], self.model(padded, self.lengths)['logits'])
        loss = sum(expected['logits'][:, i, :n].square().mean()
                   for i, n in enumerate((2, 5, 5, 2, 2, 4, 5, 4, 2)))
        loss.backward()
        for name in ('embedding', 'encoder', 'sender', 'field_embedding', 'receiver'):
            grads = [p.grad for p in getattr(self.model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_receiver_only_vector_dependency_and_batch_independence(self):
        vectors = self.model.encode(self.tokens, self.lengths)
        scores = self.model.decode(vectors)
        torch.testing.assert_close(scores, self.model.decode(vectors.clone()))
        torch.testing.assert_close(scores[0], self.model.decode(vectors[:1])[0])
        zero = self.model.decode(torch.zeros_like(vectors))
        differences = (scores-zero)[~self.model.invalid_classes[None].expand_as(scores)]
        self.assertGreater(differences.abs().max(), 1e-6)
        torch.testing.assert_close(scores[0], self.model.decode(vectors.flip(0))[1])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.model.encode(self.tokens, torch.zeros_like(self.lengths))
        with self.assertRaises(ValueError):
            self.model.decode(torch.zeros(2, 15))
        with self.assertRaises(ValueError):
            self.model.decode(torch.full((2, 16), float('nan')))


if __name__ == '__main__':
    unittest.main()
