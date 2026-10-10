import unittest
from collections import Counter, defaultdict
from antlab.semantic_binding_data import make_data, canonicalize, counterpart, table_bytes

class BindingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data()

    def test_inverse_and_counterfactual_labels(self):
        row = self.data['joint'][0]
        inverse = counterpart(row, 'equivalent')
        reverse = counterpart(row, 'opposite')
        reordered = counterpart(row, 'table')
        self.assertEqual(inverse['target'], row['target'])
        self.assertNotEqual(reverse['target'], row['target'])
        self.assertNotEqual(reordered['target'], row['target'])
        self.assertEqual(inverse['scene'], row['scene'])
        self.assertEqual(reordered['scene'], row['scene'])
        self.assertEqual(canonicalize('x is left of y.', ['x','y']),
                         ['slot0','is','left','of','slot1'])
        self.assertEqual(canonicalize('y is right of x.', ['x','y']),
                         ['slot1','is','right','of','slot0'])

    def test_semantic_split_isolation(self):
        groups = {split: {tuple(row['pair']) for row in rows}
                  for split, rows in self.data.items()}
        self.assertFalse(groups['train'] & groups['pair'])
        self.assertFalse(groups['train'] & groups['joint'])
        self.assertFalse(groups['new_names'] & groups['train'])
        self.assertEqual(groups['pair'], groups['joint'])

    def test_per_table_context_balance_and_counts(self):
        expected = dict(train=2880, validation=1440, pair=960, joint=480,
                        new_names=448, phrasing=1440)
        for split, rows in self.data.items():
            self.assertEqual(len(rows), expected[split])
            groups = defaultdict(Counter)
            for row in rows:
                groups[(tuple(row['table']),row['template'],row['orientation'])][row['target']] += 1
            for counts in groups.values():
                self.assertEqual(set(counts), {0,1,2,3})
                self.assertEqual(len(set(counts.values())), 1)

    def test_canonical_overlap_is_disclosed(self):
        known = {tuple(canonicalize(r['text'],r['table'])) for r in self.data['train']}
        self.assertEqual(len(known), 16)
        for split in ('pair','new_names'):
            self.assertTrue(all(tuple(canonicalize(r['text'],r['table'])) in known
                                for r in self.data[split]) if split=='pair' else
                            all('slot0' in canonicalize(r['text'],r['table']) for r in self.data[split]))
        self.assertTrue(all(tuple(canonicalize(r['text'],r['table'])) not in known
                            for r in self.data['joint']))

    def test_malformed_and_missing_entities_rejected(self):
        for table in (['x','x'],['x'],['x','y','z'],['x','a b'],['slot0','y']):
            with self.assertRaises(ValueError):
                canonicalize('x is above y.',table)
        for text in ('z is above y.','x is above x.',''):
            with self.assertRaises(ValueError):
                canonicalize(text,['x','y'])
        self.assertEqual(table_bytes(['x','yy']),7)

if __name__ == '__main__':
    unittest.main()
