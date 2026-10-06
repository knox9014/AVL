import unittest
import torch
from antlab.semantic_codec import pack_vectors, unpack_vectors


class SemanticCodecTests(unittest.TestCase):
    def test_exact_transport_and_segment_order(self):
        vectors = torch.arange(48, dtype=torch.float32).reshape(3, 16)/7
        payload = pack_vectors(vectors)
        self.assertEqual(len(payload), 12+3*64)
        torch.testing.assert_close(unpack_vectors(payload), vectors, rtol=0, atol=0)
        self.assertEqual(pack_vectors(unpack_vectors(payload)), payload)

    def test_malformed_header_and_size_rejected(self):
        packet = pack_vectors(torch.ones(1, 16))
        for bad in (packet[:5], packet[:-1], packet+b'X', b'NOPE'+packet[4:],
                    packet[:4]+b'\x02'+packet[5:], packet[:5]+b'\x02'+packet[6:]):
            with self.assertRaises(ValueError):
                unpack_vectors(bad)

    def test_bad_dtype_shape_nonfinite_and_empty(self):
        for vectors in (torch.zeros(0, 16), torch.zeros(2, 15), torch.zeros(1, 16).double(),
                        torch.full((1, 16), float('inf'))):
            with self.assertRaises(ValueError):
                pack_vectors(vectors)


if __name__ == '__main__':
    unittest.main()
