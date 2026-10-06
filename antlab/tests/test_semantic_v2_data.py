import hashlib
import itertools
import json
import re
import unittest

from antlab.semantic_data import FIELDS as V1_FIELDS, VOCABS as V1_VOCABS
from antlab.semantic_v2_data import FIELDS, VOCABS, WORDS, is_supported, make_data, normalize_text


class SemanticV2DataTests(unittest.TestCase):
    def test_fixed_vocabulary_and_native_regeneration(self):
        self.assertEqual(FIELDS, V1_FIELDS)
        self.assertEqual(VOCABS, V1_VOCABS)
        data = make_data()
        self.assertEqual(data, make_data())
        self.assertEqual(data, json.loads(json.dumps(data)))
        self.assertEqual(set(data), {'train', 'validation', 'combination', 'phrasing', 'joint'})
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

    def test_frames_partitions_and_rules(self):
        data = make_data()
        key = lambda row: tuple(row['frame'][f] for f in FIELDS)
        regular = {key(r) for r in data['train']}
        reserved = {key(r) for r in data['combination']}
        self.assertFalse(regular & reserved)
        self.assertEqual({key(r) for r in data['validation']}, regular)
        self.assertEqual({key(r) for r in data['phrasing']}, regular)
        self.assertEqual({key(r) for r in data['joint']}, reserved)
        self.assertEqual(len(data['train']), 8 * len(regular))
        self.assertEqual(len(data['validation']), 2 * len(regular))
        self.assertEqual(len(data['phrasing']), 2 * len(regular))
        self.assertEqual(len(data['combination']), 2 * len(reserved))
        self.assertEqual(len(data['joint']), 2 * len(reserved))
        frames = regular | reserved
        self.assertEqual(len(frames), 448)
        self.assertEqual(sum(f[1] == 'budget' for f in frames), 192)
        for values in frames:
            frame = dict(zip(FIELDS, values))
            canonical = json.dumps(frame, sort_keys=True, separators=(',', ':'))
            digest = hashlib.sha256(('avl-v2:' + canonical).encode('ascii')).hexdigest()
            self.assertEqual(values in reserved, int(digest, 16) % 5 == 0)
            if frame['subject'] == 'budget':
                self.assertEqual((frame['predicate'], frame['polarity'], frame['unit']), ('limit', 'positive', 'USD'))
                self.assertNotEqual(frame['amount'], 'none')
                self.assertNotEqual(frame['comparator'], 'none')
            else:
                self.assertIn(frame['predicate'], ('open', 'closed') if frame['subject'] == 'door' else ('on', 'off'))
                self.assertEqual((frame['amount'], frame['comparator'], frame['unit']), ('none', 'none', 'none'))

    def test_words_derived_only_from_training_and_cover_holdouts(self):
        data = make_data()
        tokens = lambda rows: {word for row in rows for word in re.findall(r'[a-z]+|[0-9]+', normalize_text(row['text']))}
        self.assertIsInstance(WORDS, tuple)
        self.assertEqual(WORDS, tuple(sorted(tokens(data['train']))))
        for split in ('validation', 'combination', 'phrasing', 'joint'):
            self.assertFalse(tokens(data[split]) - set(WORDS))

    def test_heldout_word_sequences_are_distinct_after_punctuation_removal(self):
        data = make_data()
        sequences = lambda rows: {
            tuple(re.findall(r'[a-z]+|[0-9]+', normalize_text(row['text'])))
            for row in rows
        }
        training = sequences(data['train'])
        for split in ('validation', 'combination', 'phrasing', 'joint'):
            with self.subTest(split=split):
                self.assertFalse(training & sequences(data[split]))

    def test_negation_modality_scope_and_unsupported(self):
        rows = list(itertools.chain.from_iterable(make_data().values()))
        negative = [r for r in rows if all(r['frame'][k] == v for k, v in dict(kind='fact', subject='door', predicate='closed', polarity='negative', certainty='possible', condition='rain').items())]
        self.assertTrue(negative)
        for row in negative:
            self.assertIn('the door is not closed', row['text'].lower())
            self.assertIn('possible', row['text'].lower())
            self.assertIn('when it rains', row['text'].lower())
            self.assertNotIn('may not', row['text'].lower())
        self.assertTrue(any(r['text'].startswith('When it rains, ') for r in negative))
        self.assertTrue(any(r['text'].endswith(', when it rains.') for r in negative))
        for row in rows:
            if row['frame']['kind'] == 'request':
                text = row['text'].lower()
                if row['frame']['certainty'] == 'certain': self.assertIn('please', text)
                else: self.assertTrue('consider' in text or 'think about' in text)
        for text in ('The lamp is on and the door is open.', 'Please ensure that the budget limit is exactly 17 USD.', 'It is possible that the spaceship is on.', 'It may not be possible that the lamp is on.', '', None, '\u00e9'):
            self.assertIs(is_supported(text), False)
        original = '  IT IS CERTAIN THAT THE LAMP IS ON.  '
        self.assertIs(is_supported(original), True)
        self.assertEqual(normalize_text(original), 'it is certain that the lamp is on.')
        self.assertEqual(original, '  IT IS CERTAIN THAT THE LAMP IS ON.  ')


if __name__ == '__main__':
    unittest.main()
