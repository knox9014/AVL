import unittest
from unittest.mock import patch
import torch
from antlab.semantic_v2_model import SemanticNet, tokenize


class SemanticV2ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(44)
        self.model = SemanticNet()

    def test_token_normalization_preserves_word_order_and_negation(self):
        tokens, lengths = tokenize(['It is certain that the lamp is on.', 'It is certain that the lamp is not on.'])
        changed, sizes = tokenize(['  IT\tis certain, that THE lamp is on! ', 'it is certain that the lamp is not on'])
        self.assertTrue(torch.equal(tokens, changed))
        self.assertTrue(torch.equal(lengths, sizes))
        self.assertEqual(lengths[1].item(), lengths[0].item() + 1)
        reordered, _ = tokenize(['on is lamp the that certain is it'])
        self.assertFalse(torch.equal(tokens[:1, :lengths[0]], reordered))
        with self.assertRaises(ValueError):
            tokenize(['unknownobject is on'])
        with self.assertRaises(ValueError):
            tokenize([''])

    def test_vectors_only_receiver_and_parameter_budget(self):
        count = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(count, 90000)
        self.assertLessEqual(count, 110000)
        tokens, lengths = tokenize(['It is certain that the lamp is on.', 'It is certain that the lamp is not on.'])
        out = self.model(tokens, lengths)
        self.assertEqual(out['vectors'].shape, (2, 16))
        self.assertEqual(out['logits'].shape, (2, 9, 8))
        logits = self.model.decode(out['vectors'].clone())
        torch.testing.assert_close(out['logits'], logits)
        torch.testing.assert_close(logits[0], self.model.decode(out['vectors'][:1])[0])
        self.assertGreater((out['vectors'][0] - out['vectors'][1]).abs().max().item(), 1e-6)

    def test_padding_mask_and_encoder_gradients(self):
        tokens, lengths = tokenize(['It is certain that the lamp is on.', 'It is possible that the door is not closed.'])
        original = self.model(tokens, lengths)
        padded = torch.cat([tokens, torch.ones(2, 7, dtype=torch.long)], dim=1)
        torch.testing.assert_close(original['vectors'], self.model.encode(padded, lengths))
        counts = (2, 5, 5, 2, 2, 4, 5, 4, 2)
        loss = sum(original['logits'][:, i, :n].square().mean() for i, n in enumerate(counts))
        loss.backward()
        for name in ('embedding', 'encoder', 'attention', 'sender', 'field_embedding', 'receiver'):
            gradients = [p.grad for p in getattr(self.model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_invalid_classes_and_inputs(self):
        with patch.object(self.model.receiver, 'forward', return_value=torch.full((1, 9, 8), -1e6)):
            self.assertEqual(self.model.decode(torch.zeros(1, 16)).argmax(-1).tolist(), [[0] * 9])
        with self.assertRaises(ValueError):
            self.model.decode(torch.zeros(1, 15))
        with self.assertRaises(ValueError):
            self.model.decode(torch.full((1, 16), float('nan')))
        tokens, lengths = tokenize(['It is certain that the lamp is on.'])
        with self.assertRaises(ValueError):
            self.model.encode(tokens, lengths * 0)


if __name__ == '__main__':
    unittest.main()
