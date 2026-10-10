import itertools
import unittest
from antlab.semantic_shared_operator_data import (make_data,make_chain,edge_rows,
    expand_queries,answer,find_route,validate_edges,select_pairs)

class SharedOperatorDataTests(unittest.TestCase):
    def test_fresh_four_node_support_and_train_length(self):
        data=make_data()
        self.assertEqual({k:len(expand_queries(v)) for k,v in data.items()},
                         dict(train=6912,joint=2304,phrasing=6912,new_joint=2304,known4=12288,new4=12288))
        self.assertTrue(all(len(r['names'])==3 for r in data['train']))
        qs=expand_queries(data['known4'])
        self.assertEqual(sum(q['distance']==3 for q in qs),2048)
        self.assertEqual([sum(q['distance']==3 and q['target']==i for q in qs) for i in range(5)],
                         [32,32,32,32,1920])
        self.assertEqual(len(select_pairs(data['known4'])),64)
        self.assertEqual(len(select_pairs(data['joint'])),16)

    def test_route_and_binding_for_all_table_permutations(self):
        names=('obj00','obj01','obj02','obj03')
        for order,reverse in itertools.product(range(24),range(2)):
            row=make_chain(names,(0,0,0),7,order,3,reverse)
            edges=edge_rows(row['text'],row['table'])
            path=find_route([e['slots'] for e in edges],row['table'].index(names[3]),
                            row['table'].index(names[0]),4)
            self.assertEqual(len(path),3)
            self.assertEqual(answer(row,(0,3)),0)
            self.assertEqual(answer(row,(3,0)),1)
        self.assertEqual(find_route([[0,1],[1,2],[2,3]],3,0,4),[(2,True),(1,True),(0,True)])

    def test_unknown_and_topology_rejection(self):
        row=make_chain(('obj00','obj01','obj02','obj03'),(0,2,0),0,0,3)
        self.assertEqual(answer(row,(0,3)),4)
        self.assertEqual(answer(row,(0,1)),0)
        for edges,n in [([[0,1],[0,1]],3),([[0,1],[0,2],[0,3]],4),
                         ([[0,1],[1,2],[0,2]],4),([[1,0],[1,2]],3)]:
            with self.assertRaises(ValueError):
                validate_edges(edges,n)

if __name__=='__main__':
    unittest.main()
