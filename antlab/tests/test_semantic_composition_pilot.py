import struct
import unittest
import torch
from antlab.semantic_composition_pilot import pack_message, unpack_message, physical_meaning, classification
from antlab.semantic_composition_data import infer_chain

class CompositionWireTests(unittest.TestCase):
    def test_roundtrip_and_exact_bytes(self):
        vectors=torch.arange(32,dtype=torch.float32).reshape(2,16)/10
        packet=pack_message(['obj00','obj01','obj02'],[[0,1],[1,2]],vectors)
        self.assertEqual(len(packet),171)
        decoded=unpack_message(packet)
        self.assertEqual(decoded['table'],['obj00','obj01','obj02'])
        self.assertEqual(decoded['edges'],[[0,1],[1,2]])
        self.assertTrue(torch.equal(decoded['vectors'],vectors))
        for bad in (packet[:-1],packet+b'x',b'BAD!'+packet[4:],packet[:4]+b'\\x04'+packet[5:],
                    packet[:6]+struct.pack('<H',65535)+packet[8:]):
            with self.assertRaises(ValueError):
                unpack_message(bad)

    def test_binding_not_labels_and_rejection(self):
        decoded=dict(table=['obj02','obj00','obj01'],edges=[[1,2],[0,2]],labels=[0,2])
        self.assertEqual(physical_meaning(decoded,['obj00','obj01','obj02']),(0,3))
        with self.assertRaises(ValueError):
            physical_meaning(decoded,['obj00','obj01','missing'])
        for edges,vectors in [([[0,1],[0,1]],torch.zeros(2,16)),
                              ([[1,0],[1,2]],torch.zeros(2,16)),
                              ([[0,1],[1,2]],torch.zeros(1,16)),
                              ([[0,1],[1,2]],torch.full((2,16),float('nan')))]:
            with self.assertRaises(ValueError):
                pack_message(['obj00','obj01','obj02'],edges,vectors)

    def test_observed_class_metrics_do_not_hide_support(self):
        result=classification([0,0,3,3],[0,0,0,3],16)
        self.assertEqual(result['accuracy'],.75)
        self.assertEqual(result['macro_recall_observed'],.75)
        self.assertEqual(result['observed_classes'],[0,3])
        self.assertEqual(result['support'][1],0)
        self.assertIsNone(result['recall'][1])
        self.assertEqual(infer_chain(0,2),4)

if __name__=='__main__':
    unittest.main()
