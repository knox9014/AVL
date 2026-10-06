import unittest
from unittest.mock import patch
import torch
from antlab.english_connected_model import EnglishConnectedNet, encode_local_texts


class EnglishConnectedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(105)
        self.model = EnglishConnectedNet()
        self.tokens, self.lengths = encode_local_texts([
            ['Question: Is the lamp on?', 'The lamp is on.', 'No other facts.'],
            ['Question: Is the lamp on?', 'The lamp is off.', 'No other facts.']])
        self.decoder = torch.tensor([[1, 89], [1, 78]])

    def call(self, **kw):
        args = dict(tokens=self.tokens, lengths=self.lengths, decoder_inputs=self.decoder)
        args.update(kw)
        return self.model(**args)

    def test_shared_size_and_payload(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 98928)
        for n in (1, 2, 4, 8):
            t, l = encode_local_texts([['Read this fact.'] * n])
            out = self.model(t, l, torch.tensor([[1]]))
            self.assertEqual(tuple(out['logits'].shape), (1, 1, 128))
            self.assertEqual(tuple(out['messages'].shape), (1, 2, n, 16))
            self.assertEqual(out['logical_payload_bytes'], 2*n*(n-1)*16*4)

    def test_serial_and_permutation(self):
        expected = self.call()['logits']
        torch.testing.assert_close(expected, self.call(mode='serial')['logits'])
        perm = [2, 0, 1]
        actual = self.call(tokens=self.tokens[:, perm], lengths=self.lengths[:, perm],
                           output_units=torch.tensor([1, 1]))['logits']
        torch.testing.assert_close(expected, actual)

    def test_remote_information_is_blocked(self):
        altered = self.tokens.clone()
        altered[:, 1:] = ord('X')
        for kw in (dict(mode='blocked'), dict(rounds=0)):
            before, after = self.call(**kw), self.call(tokens=altered, **kw)
            torch.testing.assert_close(before['logits'], after['logits'], rtol=0, atol=0)
            self.assertEqual(before['logical_payload_bytes'], 0)
        self.assertGreater((self.call()['logits']-self.call(tokens=altered)['logits']).abs().max().item(), 1e-6)

    def test_padding_and_gradient_paths(self):
        extra = torch.full((2, 3, 5), ord('X'))
        torch.testing.assert_close(self.call()['logits'],
                                  self.call(tokens=torch.cat([self.tokens, extra], -1))['logits'])
        self.call()['logits'].square().mean().backward()
        for name in ('encoder', 'sender', 'receiver', 'query', 'key', 'update', 'decoder'):
            grads = [p.grad for p in getattr(self.model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)

    def test_generation_and_validation(self):
        result = self.model.generate(self.tokens, self.lengths, max_new_tokens=5)
        self.assertEqual(len(result['texts']), 2)
        self.assertTrue(all(len(ids) <= 5 for ids in result['token_ids']))
        for invalid in ([], [[]], [['a'], ['b', 'c']], [['한글']]):
            with self.assertRaises(ValueError):
                encode_local_texts(invalid)
        with self.assertRaises(ValueError):
            self.call(rounds=-1)
        with self.assertRaises(ValueError):
            self.call(lengths=torch.zeros_like(self.lengths))

    def test_eos_and_autoregressive_feedback(self):
        scripted = []
        for pair in ((2, ord('Y')), (127, ord('E')), (ord('X'), 2)):
            logits = torch.full((2, 1, 128), -100.)
            for b, token in enumerate(pair):
                logits[b, 0, token] = 100.
            scripted.append(logits)
        inputs = []
        hook = self.model.embedding.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].clone()))
        try:
            with patch.object(self.model.head, 'forward', side_effect=scripted):
                result = self.model.generate(self.tokens, self.lengths, max_new_tokens=5)
        finally:
            hook.remove()
        self.assertEqual(result['token_ids'], [[2], [ord('Y'), ord('E'), 2]])
        self.assertEqual(result['texts'], ['', 'YE'])
        self.assertEqual([t[:, 0].tolist() for t in inputs[1:]], [[1, 1], [2, ord('Y')], [127, ord('E')]])
        # DEL is retained in raw token records, but is not printable English.
        with patch.object(self.model.head, 'forward', side_effect=scripted[1:]):
            result = self.model.generate(self.tokens, self.lengths, max_new_tokens=2)
        self.assertEqual(result['texts'][0], 'X')


if __name__ == '__main__':
    unittest.main()
