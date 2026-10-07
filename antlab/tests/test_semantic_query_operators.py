from collections import Counter
import unittest

import torch

from antlab.semantic_query_operators import (
    Operator, curricula, numeric_features, proposition_features, receive, train_operator)


class OperatorTests(unittest.TestCase):
    def test_curriculum_counts_and_labels(self):
        data = curricula()
        self.assertEqual(len(data['proposition']['inputs']), 64)
        self.assertEqual(Counter(data['proposition']['targets']), {0: 62, 1: 1, 2: 1})
        self.assertEqual(len(data['numeric']['inputs']), 1218)
        self.assertEqual(Counter(data['numeric']['targets']), {0: 609, 1: 205, 2: 404})
        lookup = dict(zip(map(tuple, data['proposition']['inputs']), data['proposition']['targets']))
        self.assertEqual(lookup[(1, 1, 1, 1, 1, 1)], 1)
        self.assertEqual(lookup[(1, 1, 1, 1, 1, 0)], 2)
        self.assertEqual(lookup[(0, 1, 1, 1, 1, 1)], 0)
        self.assertEqual(lookup[(1, 0, 1, 1, 1, 1)], 0)
        self.assertEqual(lookup[(1, 1, 0, 1, 1, 1)], 0)

    def test_numeric_curriculum_boundary_fixtures(self):
        table = curricula()['numeric']
        lookup = dict(zip(map(tuple, table['inputs']), table['targets']))
        self.assertEqual(lookup[(1, 1, 0, 0, 0, 0)], 1)
        self.assertEqual(lookup[(1, 0, 1, 0, 0, 0)], 1)
        self.assertEqual(lookup[(1, 0, 0, 1, 0, 0)], 1)
        self.assertEqual(lookup[(1, 1, 0, 0, -1, 1)], 2)
        self.assertEqual(lookup[(1, 0, 1, 0, 1, 1)], 2)
        self.assertEqual(lookup[(1, 0, 0, 1, 1, 1)], 2)
        self.assertEqual(lookup[(0, 1, 0, 0, 0, 0)], 0)

    def test_proposition_features_preserve_scope_and_query_binding(self):
        # Predicted fact/lamp/on/positive/certain/unconditional.
        values = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
        query = dict(subject='lamp', predicate='on')
        self.assertEqual(proposition_features(values, query).tolist(), [[1] * 6])
        self.assertEqual(proposition_features(values, dict(subject='fan', predicate='off')).tolist(),
                         [[1, 1, 1, 0, 0, 1]])
        values[0, 0], values[0, 3], values[0, 4], values[0, 5] = 1, 1, 1, 1
        self.assertEqual(proposition_features(values, query).tolist(), [[0, 0, 0, 1, 1, 0]])

    def test_numeric_features_use_predicted_amount_and_query_candidate(self):
        values = torch.tensor([[1, 4, 4, 0, 1, 1, 2, 1, 1]])
        self.assertEqual(numeric_features(values, dict(candidate=19)).tolist(),
                         [[1, 1, 0, 0, 1, 1]])
        self.assertEqual(numeric_features(values, dict(candidate=21)).tolist(),
                         [[1, 1, 0, 0, -1, 1]])

    def test_parameter_budget_and_input_validation(self):
        operator = Operator()
        self.assertEqual(sum(p.numel() for p in operator.parameters()), 1379)
        for value in (torch.zeros(1, 5), torch.zeros(0, 6),
                      torch.zeros(1, 6).double(), torch.full((1, 6), float('nan'))):
            with self.assertRaises(ValueError):
                operator(value)

    def test_receiver_routes_vector_decisions_to_learned_modules(self):
        class Decoder:
            def decode(self, vectors):
                logits = torch.full((len(vectors), 9, 8), -torch.inf)
                logits[:, :, 0] = 0.
                return logits
        class CapturingOperator:
            def __init__(self):
                self.inputs = []
            def __call__(self, value):
                self.inputs.append(value.clone())
                return torch.zeros(len(value), 3)
        prop, numeric = CapturingOperator(), CapturingOperator()
        result = receive(Decoder(), prop, numeric, torch.zeros(2, 16))
        self.assertEqual(result.shape, (2, 24))
        self.assertEqual(result[0, :3].tolist(), [0, 2, 4])
        self.assertTrue((result[:, 3:] == 10).all())
        self.assertEqual(len(prop.inputs), 8)
        self.assertEqual(len(numeric.inputs), 13)
        self.assertEqual(prop.inputs[0][0].tolist(), [1] * 6)
        with self.assertRaises(ValueError):
            receive(Decoder(), prop, numeric, torch.zeros(1, 15))

    def test_smoke_training_updates_operator(self):
        torch.set_num_threads(2)
        data = curricula()['proposition']
        torch.manual_seed(7)
        initial = Operator()
        learned, report = train_operator(data, 7, lambda: None, steps=2)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(initial.parameters(), learned.parameters())))
        self.assertEqual(report['steps'], 2)
        self.assertEqual(report['parameters'], 1379)
        self.assertTrue(torch.isfinite(torch.tensor(report['final_loss'])))


if __name__ == '__main__':
    unittest.main()
