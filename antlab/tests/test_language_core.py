"""Language contracts checked against an independent coordinate oracle."""
import itertools
import json
import unittest
from antlab.language_core import parse, evaluate, dumps, loads, execute


class LanguageCoreTests(unittest.TestCase):
    names = ['a', 'b', 'c']

    def run_source(self, source):
        return execute(source, self.names)['status']

    def test_transitivity(self):
        self.assertEqual(self.run_source('assert a left b; assert b left c; ask a left c;'), 'entailed')

    def test_four_name_long_chain_and_mixed_negation(self):
        self.assertEqual(execute('assert a below b; assert b below c; assert c below d; ask not a above d;', ['a','b','c','d'])['status'], 'entailed')
        self.assertEqual(execute('assert not a left b; assert not b left c; assert c left a; ask a above d;', ['a','b','c','d'])['status'], 'undetermined')

    def test_refuted(self):
        self.assertEqual(self.run_source('assert a left b; ask a right b;'), 'refuted')

    def test_negation_includes_equality(self):
        self.assertEqual(self.run_source('assert not a left b; ask a right b;'), 'undetermined')
        self.assertEqual(self.run_source('assert not a left b; assert not a right b; ask a left b;'), 'refuted')

    def test_orthogonal_unknown(self):
        self.assertEqual(self.run_source('assert a left b; ask a above b;'), 'undetermined')

    def test_inconsistent_cycle_no_explosion(self):
        self.assertEqual(self.run_source('assert a left b; assert b left c; assert c left a; ask a above b;'), 'inconsistent')

    def test_negated_query(self):
        self.assertEqual(self.run_source('assert a left b; ask not a right b;'), 'entailed')

    def test_strict_source(self):
        for source in ['a is left b.', 'assert a left b ask a left b;', 'ask a left a;', 'ask a left z;', 'ask a near b;', 'ask a left b; garbage', 'ask a left b; ask a left b;', 'ask  a left b;', 'ASSERT a left b;', '']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse(source, self.names)

    def test_json_roundtrip(self):
        program = parse('assert not a below b; ask a above b;', self.names)
        self.assertEqual(loads(dumps(program)), program)
        self.assertEqual(evaluate(loads(dumps(program))), evaluate(program))

    def test_invalid_ast(self):
        program = parse('ask a left b;', self.names)
        for mutate in [lambda p:p.update(extra=1), lambda p:p.update(version='AVL-L2'), lambda p:p['query'].update(negated=1), lambda p:p['query'].update(extra=1), lambda p:p.update(names=['a','a'])]:
            p=json.loads(json.dumps(program)); mutate(p)
            with self.assertRaises(ValueError): evaluate(p)
        with self.assertRaises(ValueError): loads('{"version":"AVL-L1","version":"AVL-L1"}')

    def test_bounds(self):
        for names in [['a'], ['a','b','c','d','e'], ['a','ask'], ['a','é']]:
            with self.assertRaises(ValueError): parse('ask a left b;',names)
        with self.assertRaises(ValueError): parse('assert a left b; '*33+'ask a left b;', ['a','b'])

    def test_exhaustive_two_name_oracle(self):
        # Independent full 2-D coordinates, not the implementation's axis factoring.
        atoms = [(s,r,o,n) for s,o in [('a','b'),('b','a')] for r in ['left','right','above','below'] for n in [False,True]]
        def text(atom):
            s,r,o,n=atom; return ('not ' if n else '')+f'{s} {r} {o}'
        def truth(atom, world):
            s,r,o,n=atom; x,y=world[s]; u,v=world[o]
            value={'left':x<u,'right':x>u,'above':y>v,'below':y<v}[r]
            return not value if n else value
        worlds=[dict(zip(['a','b'],zip(coords[:2],coords[2:]))) for coords in itertools.product(range(2),repeat=4)]
        count=0
        for assertions in itertools.combinations_with_replacement(atoms,2):
            valid=[w for w in worlds if all(truth(a,w) for a in assertions)]
            for query in atoms:
                possible={truth(query,w) for w in valid}
                expected={frozenset():'inconsistent',frozenset([True]):'entailed',frozenset([False]):'refuted',frozenset([False,True]):'undetermined'}[frozenset(possible)]
                source=''.join('assert '+text(a)+'; ' for a in assertions)+'ask '+text(query)+';'
                result=execute(source,['a','b'])
                self.assertEqual(result['status'],expected,source)
                for value,key in [(True,'true_model'),(False,'false_model')]:
                    witness=result[key]
                    self.assertEqual(witness is not None,value in possible)
                    if witness is not None:
                        self.assertTrue(all(truth(a,witness) for a in assertions))
                        self.assertEqual(truth(query,witness),value)
                count+=1
        self.assertEqual(count,2176)


if __name__ == '__main__': unittest.main()

