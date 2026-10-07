import unittest

import torch

from antlab.semantic_query_pilot import (ANSWERS, QUESTIONS, QueryReceiver,
                                         labels_for, measure, query_features)
from antlab.semantic_v2_data import make_data


class QueryPilotTests(unittest.TestCase):
    def test_query_features_share_numeric_identifier(self):
        ids, values = query_features(2)
        self.assertEqual(tuple(ids.shape), (48,))
        self.assertEqual(ids[:11].tolist(), list(range(11)))
        self.assertEqual(ids[11:24].tolist(), [11] * 13)
        self.assertTrue(torch.equal(values[:24], values[24:]))
        self.assertAlmostEqual(float(values[12]), .09, places=6)
        self.assertEqual(len(QUESTIONS), 24)

    def test_receiver_masks_only_question_categories(self):
        receiver = QueryReceiver()
        ids, candidates = query_features(1)
        logits = receiver(torch.zeros(24, 16), ids, candidates)
        self.assertEqual(tuple(logits.shape), (24, 13))
        self.assertTrue(torch.isneginf(logits[0, 2:]).all())
        self.assertTrue(torch.isneginf(logits[3, :8]).all())
        self.assertTrue(torch.isfinite(logits[3, 8:11]).all())
        self.assertTrue(torch.isfinite(logits[11:, 10:13]).all())
        self.assertTrue(torch.isneginf(logits[11:, :10]).all())

    def test_receiver_rejects_invalid_features(self):
        receiver = QueryReceiver()
        good = (torch.zeros(1, 16), torch.zeros(1, dtype=torch.long), torch.zeros(1, 1))
        bad = ((torch.zeros(1, 15), good[1], good[2]),
               (torch.full((1, 16), float('nan')), good[1], good[2]),
               (good[0].double(), good[1], good[2]),
               (good[0], torch.tensor([12]), good[2]),
               (good[0], good[1], torch.tensor([[float('inf')]])))
        for args in bad:
            with self.assertRaises(ValueError):
                receiver(*args)

    def test_metrics_penalize_unknown_only_predictions(self):
        # Two established outcomes and one unknown, in every proposition column.
        target = torch.zeros(3, 24, dtype=torch.long)
        for column, query in enumerate(QUESTIONS):
            if query['type'] == 'kind':
                target[:, column] = ANSWERS.index('fact')
            elif query['type'] == 'certainty':
                target[:, column] = ANSWERS.index('certain')
            elif query['type'] == 'condition':
                target[:, column] = ANSWERS.index('none')
            elif query['type'] == 'proposition':
                target[:, column] = torch.tensor([8, 9, 10])
            else:
                target[:, column] = ANSWERS.index('undetermined')
        pred = target.clone()
        pred[:, 3:11] = ANSWERS.index('undetermined')
        values = measure(target, pred)['proposition']
        self.assertAlmostEqual(values['accuracy'], 1 / 3)
        self.assertAlmostEqual(values['macro_recall'], 1 / 3)
        self.assertIsNone(values['recall'][0])

    def test_no_frame_overlap_for_query_training(self):
        data = make_data()
        train = {tuple(row['labels']) for row in data['train']}
        test = {tuple(row['labels']) for row in data['joint']}
        self.assertFalse(train & test)
        self.assertEqual(labels_for(data['joint']).shape, (202, 24))


if __name__ == '__main__':
    unittest.main()
