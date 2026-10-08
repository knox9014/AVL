import unittest
from antlab.semantic_learned_query_data import make_data, expand_queries, semantic_signature, answer, HELDOUT
from antlab.semantic_composition_data import make_row

class LearnedQueryDataTests(unittest.TestCase):
    def test_split_semantically_disjoint_and_positive_support(self):
        data=make_data()
        self.assertEqual({k:len(expand_queries(v)) for k,v in data.items()},
                         dict(train=6912,joint=2304,phrasing=6912,new_names=9216,new_joint=2304))
        train={semantic_signature(data['train'][q['row']],q['pair']) for q in expand_queries(data['train'])}
        joint={semantic_signature(data['joint'][q['row']],q['pair']) for q in expand_queries(data['joint'])}
        self.assertFalse(train & joint)
        self.assertTrue(all((b^1,a^1) in HELDOUT for a,b in HELDOUT))
        positive=[q['target'] for q in expand_queries(data['joint']) if q['kind']=='path' and q['target']!=4]
        self.assertEqual(set(positive),{0,1})
        self.assertEqual({q['target'] for q in expand_queries(data['train'])},set(range(5)))

    def test_signature_quotients_names_and_chain_reversal(self):
        a=make_row(('obj00','obj01','obj02'),(0,0),0,0,0)
        b=make_row(('new00','new01','new02'),(1,1),3,5,3,1)
        self.assertEqual(semantic_signature(a,(0,2)),semantic_signature(b,(2,0)))
        self.assertNotEqual(semantic_signature(a,(0,2)),semantic_signature(a,(2,0)))

    def test_answers_preserve_direction_and_uncertainty(self):
        row=make_row(('obj00','obj01','obj02'),(0,0),0,0,0)
        self.assertEqual([answer(row,pair) for pair in ((0,1),(1,0),(0,2),(2,0))],[0,1,0,1])
        mixed=make_row(('obj00','obj01','obj02'),(0,2),0,0,0)
        self.assertEqual(answer(mixed,(0,2)),4)
        for pair in ((0,0),(0,3),(True,1)):
            with self.assertRaises(ValueError):
                answer(row,pair)

if __name__=='__main__':
    unittest.main()
