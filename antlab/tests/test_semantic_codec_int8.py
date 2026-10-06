import struct
import unittest
import torch
from antlab import semantic_codec_int8 as codec


class Int8CodecTests(unittest.TestCase):
    def test_real_bytes_order_zero_row_and_bounded_quantization_error(self):
        vectors = torch.stack([torch.zeros(16), torch.linspace(-4, 4, 16)])
        packet = codec.pack_vectors(vectors)
        self.assertIsInstance(packet, bytes)
        self.assertEqual(len(packet), 52)  # 12-byte header + 2 * (4 + 16).
        received = codec.unpack_vectors(packet)
        self.assertTrue(torch.equal(received[0], vectors[0]))
        self.assertTrue(torch.all((received[1]-vectors[1]).abs() <= 4/127/2 + 1e-6))
        self.assertEqual(received.dtype, torch.float32)
        self.assertLess(received[1, 0].item(), 0)
        self.assertGreater(received[1, -1].item(), 0)

    def test_malformed_scale_range_and_packet_length_rejected(self):
        header = struct.pack('<4sBBHI', b'AVQ1', 1, 2, 16, 1)
        for scale in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                codec.unpack_vectors(header + struct.pack('<f16b', scale, *([0]*16)))
        with self.assertRaises(ValueError):
            codec.unpack_vectors(header + struct.pack('<f16b', 1., -128, *([0]*15)))
        for packet in (b'bad', header, header + bytes(21), bytearray(header + bytes(20))):
            with self.assertRaises(ValueError):
                codec.unpack_vectors(packet)

    def test_nonfinite_invalid_shape_dtype_and_overflow_rejected(self):
        for vector in (torch.zeros(0, 16), torch.zeros(1, 15), torch.zeros(1, 16, dtype=torch.float64),
                       torch.full((1, 16), float('nan')), torch.zeros(1025, 16)):
            with self.assertRaises(ValueError):
                codec.pack_vectors(vector)
        header = struct.pack('<4sBBHI', b'AVQ1', 1, 2, 16, 1)
        with self.assertRaises(ValueError):
            codec.unpack_vectors(header + struct.pack('<f16b', 3e38, *([127]*16)))


if __name__ == '__main__':
    unittest.main()
