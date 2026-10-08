import math
import struct
import unittest
from antlab.semantic_grounded_codec import anchors, pack, unpack


class GroundedCodecTests(unittest.TestCase):
    def test_anchors_and_inversion(self):
        for width in (2,16):
            points=anchors(width)
            for i in range(4):
                self.assertEqual(points[i^1],[-x for x in points[i]])
            self.assertEqual(len({tuple(x) for x in points}),4)

    def test_wire_float32_and_cost(self):
        for width in (2,16):
            packet=pack(anchors(width),1)
            decoded=unpack(packet,1,width)
            self.assertEqual(len(packet),12+4*width*4)
            for a,b in zip(decoded,anchors(width)):
                for x,y in zip(a,b): self.assertAlmostEqual(x,y,places=6)

    def test_reject_malformed_and_contract_mismatch(self):
        good=pack(anchors(2),1)
        for packet in [good[:-1],good+b'x',b'',b'BAD!'+good[4:],good[:4]+bytes([2])+good[5:],good[:12]+struct.pack('<f',math.nan)+good[16:]]:
            with self.assertRaises(ValueError): unpack(packet,1,2)
        with self.assertRaises(ValueError): unpack(good,0,2)
        with self.assertRaises(ValueError): unpack(good,1,16)
        for vectors,contract in [([],1),([[1,2]],0),([[math.inf,0]],1),([[True,0]],1),([[0]*3],1),([[0,0]]*1025,1)]:
            with self.assertRaises(ValueError): pack(vectors,contract)


if __name__=='__main__': unittest.main()

