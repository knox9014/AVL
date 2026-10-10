import itertools
import unittest
import torch
from antlab.semantic_shared_operator_pilot import (pack_graph,unpack_graph,pack_query,unpack_query,
                                                  prepare,SharedOperator)
class SharedOperatorTests(unittest.TestCase):
    def graph(self,n=4):
        vectors=torch.zeros(n-1,16)
        return pack_graph(['obj'+str(i).zfill(2) for i in range(n)],
                          [[i,i+1] for i in range(n-1)],vectors)

    def test_wire_sizes_and_invalid_graphs(self):
        for n,size in ((3,181),(4,254)):
            graph=self.graph(n)
            packet=pack_query(graph,0,n-1)
            self.assertEqual(len(packet),size)
            self.assertEqual(unpack_query(packet)['query'],[0,n-1])
            self.assertTrue(torch.equal(unpack_graph(graph)['vectors'],torch.zeros(n-1,16)))
            for bad in (packet[:-1],packet+b'x',b'BAD!'+packet[4:],packet[:-2]+bytes([0,0])):
                with self.assertRaises(ValueError):
                    unpack_query(bad)
        with self.assertRaises(ValueError):
            pack_graph(['obj00','obj01','obj02','obj03'],[[0,1],[0,2],[0,3]],torch.zeros(3,16))

    def test_relation_channel_equivariance_and_recursion(self):
        torch.manual_seed(7)
        model=SharedOperator()
        self.assertEqual(sum(p.numel() for p in model.parameters()),66)
        p=torch.softmax(torch.randn(8,3,4),dim=-1)
        lengths=torch.tensor([1,2,3,1,2,3,2,3])
        base=model(p,lengths)
        for permutation in itertools.permutations(range(4)):
            result=model(p[:,:,list(permutation)],lengths)
            self.assertTrue(torch.allclose(result[:,:4],base[:,list(permutation)],atol=1e-6))
            self.assertTrue(torch.allclose(result[:,4],base[:,4],atol=1e-6))
        expected=model.combine(torch.softmax(model.combine(p[:,0],p[:,1]),dim=-1)[:,:4],p[:,2])
        self.assertTrue(torch.allclose(model(p,torch.full((8,),3)),expected,atol=1e-6))
        torch.nn.functional.cross_entropy(base,torch.arange(8)%5).backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.full((8,3,4),float('nan')),lengths)

    def test_transmitted_route_inverts_direction(self):
        class Primitive:
            def receive(self,v):
                result=torch.zeros(len(v),4)
                result[:,0]=20
                return result
        p,lengths=prepare(Primitive(),[pack_query(self.graph(),3,0)])
        self.assertEqual(lengths.tolist(),[3])
        self.assertEqual(p.argmax(-1).tolist(),[[1,1,1]])
        zero,zero_lengths=prepare(Primitive(),[pack_query(self.graph(),0,3)],zero=True)
        self.assertEqual(zero_lengths.tolist(),[3])
        self.assertTrue(torch.isfinite(zero).all())

if __name__=='__main__':
    unittest.main()
