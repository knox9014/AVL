import hashlib
import itertools
import json
import unittest

from antlab.semantic_data import FIELDS, VOCABS, is_supported, make_data, normalize_text


class SemanticDataTests(unittest.TestCase):
    def test_fixed_vocabulary(self):
        self.assertEqual(FIELDS, ('kind', 'subject', 'predicate', 'polarity', 'certainty', 'condition', 'amount', 'comparator', 'unit'))
        self.assertEqual(VOCABS, (('fact', 'request'), ('lamp', 'heater', 'fan', 'door', 'budget'), ('on', 'off', 'open', 'closed', 'limit'), ('positive', 'negative'), ('certain', 'possible'), ('none', 'rain', 'cold', 'night'), ('none', '10', '20', '50', '100'), ('none', 'at-most', 'at-least', 'exact'), ('none', 'USD')))

    def test_regeneration_native_rows_and_disjoint_sources(self):
        data = make_data()
        self.assertEqual(data, make_data())
        self.assertEqual(data, json.loads(json.dumps(data)))
        self.assertEqual(set(data), {'train', 'validation', 'combination', 'phrasing'})
        seen = set()
        for rows in data.values():
            for row in rows:
                self.assertEqual(set(row), {'id', 'text', 'labels', 'frame'})
                row['text'].encode('ascii')
                normalized = normalize_text(row['text'])
                self.assertNotIn(normalized, seen)
                seen.add(normalized)
                self.assertEqual(row['id'], hashlib.sha256(normalized.encode('ascii')).hexdigest())
                self.assertEqual(row['labels'], [v.index(row['frame'][f]) for f, v in zip(FIELDS, VOCABS)])
                self.assertIs(is_supported(row['text']), True)

    def test_frame_balance_rules_and_holdout(self):
        data = make_data()
        key = lambda row: tuple(row['frame'][f] for f in FIELDS)
        train_frames = {key(r) for r in data['train']}
        heldout = {key(r) for r in data['combination']}
        self.assertFalse(train_frames & heldout)
        self.assertEqual({key(r) for r in data['validation']}, train_frames)
        self.assertEqual({key(r) for r in data['phrasing']}, train_frames)
        self.assertEqual(len(data['train']), 3 * len(train_frames))
        frames = train_frames | heldout
        self.assertEqual(len(frames), 448)
        self.assertEqual(sum(f[1] == 'budget' for f in frames), 192)
        self.assertEqual(sum(f[1] != 'budget' for f in frames), 256)
        for values in frames:
            f = dict(zip(FIELDS, values))
            if f['subject'] == 'budget':
                self.assertEqual((f['predicate'], f['polarity'], f['unit']), ('limit', 'positive', 'USD'))
                self.assertNotEqual(f['amount'], 'none')
                self.assertNotEqual(f['comparator'], 'none')
            else:
                self.assertIn(f['predicate'], ('open', 'closed') if f['subject'] == 'door' else ('on', 'off'))
                self.assertEqual((f['amount'], f['comparator'], f['unit']), ('none', 'none', 'none'))
        for index, vocab in enumerate(VOCABS):
            counts = [sum(f[index] == value for f in frames) for value in vocab]
            self.assertTrue(all(counts))
        for f in frames:
            digest = hashlib.sha256(json.dumps(dict(zip(FIELDS, f)), sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
            self.assertEqual(f in heldout, int(digest, 16) % 5 == 0)

    def test_negation_scope_and_request_fixtures(self):
        rows = list(itertools.chain.from_iterable(make_data().values()))
        def fixture(**values):
            return next(r for r in rows if all(r['frame'][k] == v for k, v in values.items()))
        negative = fixture(kind='fact', subject='door', predicate='closed', polarity='negative', certainty='possible', condition='rain')
        self.assertIn('possible', negative['text'].lower())
        self.assertIn('the door is not closed', negative['text'].lower())
        self.assertTrue(negative['text'].startswith('When it rains, '))
        self.assertEqual(negative['labels'][2], VOCABS[2].index('closed'))
        request = fixture(kind='request', subject='budget', certainty='possible', amount='20', comparator='at-least')
        self.assertIn('consider', request['text'].lower())
        self.assertIn('at least 20 USD', request['text'])
        certain = fixture(kind='request', subject='lamp', certainty='certain')
        self.assertIn('please', certain['text'].lower())

    def test_unsupported_and_boolean_only_admission(self):
        for text in ('The lamp is on and the door is open.', 'The budget limit is 17 USD.', 'The lamp must not possibly be on.', 'Turn on the spaceship.', '', '\u00e9', None):
            self.assertIs(is_supported(text), False)
        original = '  IT IS CERTAIN THAT THE LAMP IS ON.  '
        self.assertIs(is_supported(original), True)
        self.assertEqual(original, '  IT IS CERTAIN THAT THE LAMP IS ON.  ')


if __name__ == '__main__':
    unittest.main()
