import itertools
import unittest
from antlab.semantic_composition_data import make_data, make_row, edge_rows, infer_chain, table_size

class CompositionDataTests(unittest.TestCase):
    def test_partition_and_heldout_structure(self):
        data=make_data()
        self.assertEqual({k:len(v) for k,v in data.items()},
                         dict(development=1152,joint=384,phrasing=1152,new_names=1536))
        dev={tuple(r['relations']) for r in data['development']}
        joint={tuple(r['relations']) for r in data['joint']}
        self.assertFalse(dev & joint)
        self.assertEqual(len(dev|joint),16)
        self.assertEqual({r[i] for r in dev for i in range(2)},set(range(4)))

    def test_slot_binding_tracks_orientation_order_and_clause_reversal(self):
        for orientation,order,reverse in itertools.product(range(4),range(6),range(2)):
            row=make_row(('obj00','obj01','obj02'),(0,3),orientation,order,3,reverse)
            edges=edge_rows(row['text'],row['table'])
            self.assertEqual(len(edges),2)
            for e in edges:
                self.assertLess(*e['slots'])
                self.assertEqual(e['table'],[row['table'][i] for i in e['slots']])
                self.assertEqual(sorted(e['table']),sorted(
                    row['names'][:2] if e['clause_index']==(1 if reverse else 0) else row['names'][1:]))
            self.assertEqual(table_size(row['table']),21)

    def test_queries_and_rejection(self):
        for i in range(4):
            self.assertEqual(infer_chain(i,i),i)
            self.assertEqual(infer_chain(i,i^1),4)
        self.assertEqual(infer_chain(0,2),4)
        row=make_row(('obj00','obj01','obj02'),(0,2),0,0,0)
        for text,table in [(row['text'],['obj00']*3),
                           ('obj00 is left of obj01.',row['table']),
                           (row['text'].replace('obj02','obj00'),row['table']),
                           (row['text'],['slot2','obj00','obj01'])]:
            with self.assertRaises(ValueError):
                edge_rows(text,table)

if __name__=='__main__':
    unittest.main()
