import unittest
import torch
from antlab.semantic_binding_data import make_data
from antlab.semantic_binding_pilot import BindingNet, encode_texts, wire_vectors, score, paired_score

class BindingPilotTests(unittest.TestCase):
    def test_wire_roundtrip_and_finite_validation(self):
        vectors=torch.arange(48,dtype=torch.float32).reshape(3,16)/48
        self.assertTrue(torch.equal(wire_vectors(vectors),vectors))
        with self.assertRaises(ValueError):
            wire_vectors(torch.full((1,16),float('nan')))
        with self.assertRaises(ValueError):
            wire_vectors(torch.zeros((1,15)))

    def test_all_classes_and_paired_correctness(self):
        targets=torch.tensor([0,1,2,3])
        metrics=score(targets,torch.tensor([0,1,2,2]))
        self.assertEqual(metrics['accuracy'],.75)
        self.assertEqual(metrics['macro_recall'],.75)
        self.assertEqual(metrics['support'],[1,1,1,1])
        self.assertEqual(paired_score(targets,targets,torch.tensor([0,1,3,2]),targets),.5)
        with self.assertRaises(ValueError):
            score(torch.tensor([0,0]),torch.tensor([0,0]))

    def test_sender_uses_grammar_order(self):
        data=make_data()
        vocabulary=sorted({token for row in data['train']
                           for token in __import__('antlab.semantic_binding_data',fromlist=['canonicalize']).canonicalize(row['text'],row['table'])})
        x,lengths=encode_texts(data['train'][:8],vocabulary)
        model=BindingNet(len(vocabulary))
        vectors=model.encode(x,lengths)
        self.assertEqual(vectors.shape,(8,16))
        self.assertTrue(torch.isfinite(vectors).all())
        logits=model.receive(wire_vectors(vectors))
        self.assertEqual(logits.shape,(8,4))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([r['target'] for r in data['train'][:8]]))
        loss.backward()
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()),0.)

if __name__=='__main__':
    unittest.main()
