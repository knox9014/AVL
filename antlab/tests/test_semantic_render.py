import unittest

from antlab.semantic_render import render_meaning
from antlab.semantic_v2_data import FIELDS, is_supported, make_data


class SemanticRenderTests(unittest.TestCase):
    def test_all_allowed_frames_render_to_distinct_admitted_canonical_text(self):
        frames = {}
        for rows in make_data().values():
            for row in rows:
                frames.setdefault(tuple(row['frame'][field] for field in FIELDS), row)
        self.assertEqual(len(frames), 448)
        rendered = set()
        for row in frames.values():
            frame = row['frame']
            original = dict(frame)
            text = render_meaning(frame)
            self.assertEqual(frame, original)
            self.assertTrue(is_supported(text))
            self.assertNotIn(text, rendered)
            rendered.add(text)
            # First gold row for each frame uses canonical training template 0.
            # Match the independent text fixture, not merely grammar membership.
            # This frame-rendering roundtrip does not evaluate a trained model.
            self.assertEqual(text, row['text'])

    def test_explicit_scope_and_request_fixtures(self):
        frame = dict(zip(FIELDS, ('fact', 'door', 'closed', 'negative', 'possible', 'rain', 'none', 'none', 'none')))
        self.assertEqual(render_meaning(frame), 'When it rains, it is possible that the door is not closed.')
        frame.update(kind='request', certainty='certain', condition='cold')
        self.assertEqual(render_meaning(frame), 'When it is cold, please make sure that the door is not closed.')
        frame.update(subject='budget', predicate='limit', polarity='positive', certainty='possible', condition='night', amount='20', comparator='at-least', unit='USD')
        self.assertEqual(render_meaning(frame), 'At night, consider making sure that the budget limit is at least 20 USD.')

    def test_invalid_schema_classes_and_semantic_combinations_raise(self):
        valid = dict(zip(FIELDS, ('fact', 'lamp', 'on', 'positive', 'certain', 'none', 'none', 'none', 'none')))
        invalid = [None, [], {k: v for k, v in valid.items() if k != 'unit'}, dict(valid, extra='none')]
        invalid.extend(dict(valid, **change) for change in (
            {'subject': 'spaceship'}, {'certainty': None}, {'amount': 10},
            {'predicate': 'open'}, {'subject': 'door'}, {'amount': '10'},
            {'comparator': 'exact'}, {'unit': 'USD'}, {'subject': 'budget'},
        ))
        budget = dict(valid, subject='budget', predicate='limit', amount='20', comparator='exact', unit='USD')
        invalid.extend(dict(budget, **change) for change in (
            {'polarity': 'negative'}, {'amount': 'none'}, {'comparator': 'none'},
            {'unit': 'none'}, {'predicate': 'off'},
        ))
        for frame in invalid:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                render_meaning(frame)


if __name__ == '__main__':
    unittest.main()
