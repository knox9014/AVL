import unittest
import torch
from antlab.semantic_composition_pilot import pack_message
from antlab.semantic_learned_query_pilot import (pack_query, unpack_query, features,
                                               QueryReceiver, rule_predict, balanced_indices, score_queries)

class LearnedQueryReceiverTests(unittest.TestCase):
    def graph(self):
        vectors=torch.zeros(2,16)
        return pack_message(['obj00','obj01','obj02'],[[0,1],[1,2]],vectors)

    def test_wire_features_and_malformed_queries(self):
        message=self.graph()
        packet=pack_query(message,0,2)
        self.assertEqual(len(packet),181)
        decoded=unpack_query(packet)
        self.assertEqual(decoded['query'],[0,2])
        x=features([packet])
        self.assertEqual(x.shape,(1,50))
        self.assertEqual(x.dtype,torch.float32)
        self.assertEqual(x[0,44:].tolist(),[1,0,0,0,0,1])
        for bad in (packet[:-1],packet+b'x',b'BAD!'+packet[4:],packet[:-2]+bytes([0,0]),
                    packet[:-2]+bytes([0,3])):
            with self.assertRaises(ValueError):
                unpack_query(bad)

    def test_raw_receiver_trainable_and_no_oracle_features(self):
        x=features([pack_query(self.graph(),0,2)])
        model=QueryReceiver()
        self.assertEqual(sum(p.numel() for p in model.parameters()),7749)
        logits=model(x)
        self.assertEqual(logits.shape,(1,5))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(1,49))
        with self.assertRaises(ValueError):
            model(torch.full((1,50),float('nan')))

    def test_empty_intervention_subset_is_not_a_perfect_score(self):
        metrics=score_queries([dict(target=0,kind='direct'),dict(target=4,kind='path')],[0,4])
        self.assertEqual(metrics['all']['accuracy'],1)
        self.assertEqual(metrics['positive_path']['records'],0)
        self.assertIsNone(metrics['positive_path']['accuracy'])
        self.assertEqual(metrics['positive_path']['support'],[0]*5)
        self.assertEqual(metrics['positive_path']['observed_classes'],[])

    def test_rule_control_and_balanced_sampling(self):
        class ConstantLeft:
            def receive(self,v):
                result=torch.zeros(len(v),4)
                result[:,0]=1
                return result
        packets=[pack_query(self.graph(),0,2),pack_query(self.graph(),2,0)]
        self.assertEqual(rule_predict(ConstantLeft(),packets),[0,1])
        indices=balanced_indices(torch.tensor([0,1,2,3,4]),torch.Generator().manual_seed(7),128)
        self.assertEqual(indices.shape,(128,))
        self.assertEqual(set(indices.tolist()),set(range(5)))
        with self.assertRaises(ValueError):
            balanced_indices(torch.tensor([0,1,2,3]),torch.Generator().manual_seed(7),128)

if __name__=='__main__':
    unittest.main()
