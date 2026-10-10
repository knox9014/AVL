import struct
import unittest
import torch
from antlab.semantic_codec import pack_vectors
from antlab.semantic_named_values import (
    MAX_UINT64, HEADER, canonicalize_named, pack_named, unpack_named,
    query_named_amount, send_named_segments, receive_named_packets)
from antlab.semantic_v2_data import VOCABS
from antlab.semantic_named_values_audit import audit_rows

class NamedValuesTests(unittest.TestCase):
    def inner(self):
        return pack_vectors(torch.zeros((1, 16), dtype=torch.float32))

    def test_exact_uint64_roundtrip(self):
        for amount in (None, 0, 9007199254740993, MAX_UINT64):
            packet = pack_named(self.inner(), 'budget-east-7', amount)
            self.assertEqual(unpack_named(packet), (self.inner(), 'budget-east-7', amount))
            self.assertEqual(len(packet), 12+13+76+(8 if amount is not None else 0))

    def test_reject_malformed_packets(self):
        packet = pack_named(self.inner(), 'x', 17)
        bad = [packet[:-1], packet+b'x', b'', bytearray(packet)]
        for offset, value in ((0, 0), (4, 2), (5, 2), (6, 0), (8, 75)):
            changed = bytearray(packet)
            changed[offset] = value
            bad.append(bytes(changed))
        bad.append(packet[:12]+b'!'+packet[13:])
        bad.append(packet[:-76]+b'NOPE'+packet[-72:])
        for value in bad:
            with self.subTest(value=value[:12]):
                with self.assertRaises(ValueError):
                    unpack_named(value)

    def test_reject_invalid_literal_and_name(self):
        for value in (-1, MAX_UINT64+1, True, 1., '1'):
            with self.assertRaises(ValueError):
                pack_named(self.inner(), 'x', value)
        for name in ('', '-x', 'a b', 'é', 'x'*65):
            with self.assertRaises(ValueError):
                pack_named(self.inner(), name)
        with self.assertRaises(ValueError):
            pack_named(pack_vectors(torch.zeros((2,16))), 'x')

    def test_canonicalization_preserves_relation(self):
        text = 'At night, it is possible that the lamp named "lamp-A7" is not on.'
        self.assertEqual(canonicalize_named(text),
                         ('At night, it is possible that the lamp is not on.', 'lamp-A7', None))
        text = 'It is certain that the budget named "B7" limit is at least 9007199254740993 USD.'
        self.assertEqual(canonicalize_named(text),
                         ('It is certain that the budget limit is at least 20 USD.', 'B7', 9007199254740993))

    def test_unsupported_grammar_is_local(self):
        class NeverEncode:
            def encode(self, *args):
                raise AssertionError('unsupported raw text reached encoder')
        texts = ['the spaceship named "x" flies.',
                 'It is certain that the budget named "x" limit is exactly 18446744073709551616 USD.',
                 'It is certain that the lamp named "-bad" is on.',
                 'It is certain that the lamp is on.']
        sent = send_named_segments(NeverEncode(), texts)
        self.assertEqual(sent['packets'], [])
        self.assertEqual([row['text'] for row in sent['unsupported']], texts)

    def test_integer_difference_precedes_float_conversion(self):
        class Capture:
            def __init__(self):
                self.features = None
            def __call__(self, features):
                self.features = features
                return torch.tensor([[0., 1., 0.]])
        for amount in (9007199254740993, MAX_UINT64):
            meaning = dict(subject='budget', amount=str(amount), comparator='at-most')
            operator = Capture()
            answer = query_named_amount(operator, meaning, amount-1)
            self.assertEqual(answer['answer'], 'allowed')
            self.assertEqual(operator.features[0].tolist(), [1., 1., 0., 0., 1., 1.])
            self.assertEqual(query_named_amount(operator, meaning, amount-102)['status'],
                             'unsupported_numeric_distance')
        with self.assertRaises(ValueError):
            query_named_amount(Capture(), meaning, True)

    def test_receiver_uses_vector_and_checks_literal_role(self):
        class Receiver:
            def __init__(self, budget):
                self.budget = budget
            def decode(self, vectors):
                self.seen = vectors.clone()
                values = ('fact', 'budget', 'limit', 'positive', 'certain', 'none',
                          '20', 'exact', 'USD') if self.budget else (
                          'fact', 'lamp', 'on', 'negative', 'possible', 'night',
                          'none', 'none', 'none')
                logits = torch.full((len(vectors), 9, 8), -100.)
                for column, (vocab, value) in enumerate(zip(VOCABS, values)):
                    logits[:, column, vocab.index(value)] = 100.
                return logits
        vector = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        packet = pack_named(pack_vectors(vector), 'B7', MAX_UINT64)
        receiver = Receiver(True)
        meaning = receive_named_packets(receiver, [packet])[0]
        self.assertTrue(torch.equal(receiver.seen, vector))
        self.assertEqual(meaning['amount'], str(MAX_UINT64))
        self.assertEqual(meaning['entity'], 'B7')
        with self.assertRaises(ValueError):
            receive_named_packets(Receiver(False), [packet])
        with self.assertRaises(ValueError):
            receive_named_packets(Receiver(True), [pack_named(self.inner(), 'x')])

    def test_preregistered_dataset_and_adapter(self):
        rows = audit_rows()
        self.assertEqual(len(rows), 2560)
        self.assertEqual(len({row['text'] for row in rows}), 2560)
        for row in rows:
            _, name, amount = canonicalize_named(row['text'])
            self.assertEqual(name, row['meaning']['entity'])
            if amount is not None:
                self.assertEqual(str(amount), row['meaning']['amount'])

if __name__ == '__main__':
    unittest.main()
