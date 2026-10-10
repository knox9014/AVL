import unittest

from antlab.semantic_queries import answer_query
from antlab.semantic_v2_data import FIELDS, _frames


def device(**changes):
    frame = dict(zip(FIELDS, ('fact', 'lamp', 'on', 'positive', 'certain',
                             'none', 'none', 'none', 'none')))
    frame.update(changes)
    return frame


def proposition(subject='lamp', predicate='on'):
    return dict(type='proposition', subject=subject, predicate=predicate)


def budget(**changes):
    return device(subject='budget', predicate='limit', amount='20',
                  comparator='at-most', unit='USD', **changes)


class SemanticQueryTests(unittest.TestCase):
    def test_positive_and_negative_claims(self):
        self.assertEqual(answer_query(device(), proposition()), 'entailed')
        self.assertEqual(answer_query(device(polarity='negative'), proposition()),
                         'contradicted')

    def test_request_is_not_world_evidence(self):
        for polarity in ('positive', 'negative'):
            self.assertEqual(answer_query(device(kind='request', polarity=polarity),
                                          proposition()), 'undetermined')

    def test_possibility_is_not_certainty(self):
        for polarity in ('positive', 'negative'):
            self.assertEqual(answer_query(device(certainty='possible', polarity=polarity),
                                          proposition()), 'undetermined')

    def test_condition_without_world_context_is_unknown(self):
        for condition in ('rain', 'cold', 'night'):
            self.assertEqual(answer_query(device(condition=condition), proposition()),
                             'undetermined')

    def test_no_antonym_or_other_subject_inference(self):
        self.assertEqual(answer_query(device(polarity='negative'),
                                      proposition(predicate='off')), 'undetermined')
        self.assertEqual(answer_query(device(), proposition(subject='fan')),
                         'undetermined')

    def test_metadata_describes_message_not_world(self):
        frame = device(kind='request', certainty='possible', condition='cold')
        for name, expected in (('kind', 'request'), ('certainty', 'possible'),
                               ('condition', 'cold')):
            self.assertEqual(answer_query(frame, dict(type=name)), expected)

    def test_numeric_boundaries_are_inclusive(self):
        expected = {
            'at-most': ('allowed', 'allowed', 'disallowed'),
            'at-least': ('disallowed', 'allowed', 'allowed'),
            'exact': ('disallowed', 'allowed', 'disallowed'),
        }
        for comparator, answers in expected.items():
            frame = budget()
            frame['comparator'] = comparator
            for candidate, target in zip((19, 20, 21), answers):
                self.assertEqual(answer_query(frame, dict(type='constraint_allows',
                                 candidate=candidate, unit='USD')), target)

    def test_constraint_interpretation_is_not_enforcement(self):
        frame = budget(kind='request', certainty='possible', condition='rain')
        self.assertEqual(answer_query(frame, dict(type='constraint_allows',
                         candidate=19, unit='USD')), 'allowed')
        self.assertEqual(answer_query(frame, proposition()), 'undetermined')
        self.assertEqual(answer_query(device(), dict(type='constraint_allows',
                         candidate=19, unit='USD')), 'undetermined')

    def test_invalid_queries_and_frames_are_rejected(self):
        queries = (None, [], {}, dict(type='unknown'), dict(type='kind', extra=1),
                   dict(type='proposition', subject=[], predicate='on'),
                   dict(type='proposition', subject='lamp', predicate='open'),
                   dict(type='constraint_allows', candidate=True, unit='USD'),
                   dict(type='constraint_allows', candidate=-1, unit='USD'),
                   dict(type='constraint_allows', candidate=1.5, unit='USD'),
                   dict(type='constraint_allows', candidate=10, unit='EUR'),
                   dict(type='constraint_allows', candidate=1000001, unit='USD'))
        for query in queries:
            with self.subTest(query=query), self.assertRaises(ValueError):
                answer_query(device(), query)
        with self.assertRaises(ValueError):
            answer_query(device(predicate='open'), dict(type='kind'))

    def test_entire_finite_grammar_and_inputs_are_unchanged(self):
        frames = list(_frames())
        self.assertEqual(len(frames), 448)
        for frame in frames:
            original = dict(frame)
            for key in ('kind', 'certainty', 'condition'):
                self.assertEqual(answer_query(frame, dict(type=key)), frame[key])
            if frame['subject'] != 'budget':
                query = proposition(frame['subject'], frame['predicate'])
                result = answer_query(frame, query)
                self.assertIn(result, ('entailed', 'contradicted', 'undetermined'))
            else:
                for candidate in (0, 10, 20, 50, 100, 101):
                    self.assertIn(answer_query(frame, dict(type='constraint_allows',
                                  candidate=candidate, unit='USD')),
                                  ('allowed', 'disallowed'))
            self.assertEqual(frame, original)


if __name__ == '__main__':
    unittest.main()
