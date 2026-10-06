"""Experimental lossy int8 vectors; independent from frozen float32 packets.

Each ordered width-16 vector carries a float32 positive scale and 16 signed
integers. The packet contains no text, semantic labels or checkpoint identity.
Only compatible same-checkpoint receivers should use the reconstructed vectors.
"""
import struct
import torch

HEADER = struct.Struct('<4sBBHI')
ROW = struct.Struct('<f16b')
MAX_SEGMENTS = 1024


def pack_vectors(vectors):
    if vectors.ndim != 2 or vectors.shape[1] != 16 or not 1 <= vectors.shape[0] <= MAX_SEGMENTS:
        raise ValueError('expected 1..1024 ordered vectors of width16')
    if vectors.dtype != torch.float32 or not torch.isfinite(vectors).all():
        raise ValueError('finite float32 vectors required')
    values = vectors.detach().cpu().contiguous()
    scales = (values.abs().amax(dim=1)/127).clamp_min(torch.finfo(torch.float32).tiny)
    scales = torch.where(values.abs().amax(dim=1) == 0, torch.ones_like(scales), scales)
    quantized = (values/scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    reconstructed = quantized.float()*scales[:, None]
    if not torch.isfinite(reconstructed).all():
        raise ValueError('quantization overflow')
    rows = [ROW.pack(scale, *row) for scale, row in zip(scales.tolist(), quantized.tolist())]
    return HEADER.pack(b'AVQ1', 1, 2, 16, values.shape[0]) + b''.join(rows)


def unpack_vectors(packet):
    if not isinstance(packet, bytes) or len(packet) < HEADER.size:
        raise ValueError('truncated or invalid packet')
    magic, version, dtype, width, count = HEADER.unpack_from(packet)
    if (magic, version, dtype, width) != (b'AVQ1', 1, 2, 16) or not 1 <= count <= MAX_SEGMENTS:
        raise ValueError('unsupported int8 vector packet header')
    if len(packet) != HEADER.size + count*ROW.size:
        raise ValueError('int8 vector packet size mismatch')
    rows = [ROW.unpack_from(packet, HEADER.size + index*ROW.size) for index in range(count)]
    scales = torch.tensor([row[0] for row in rows], dtype=torch.float32)
    integers = torch.tensor([row[1:] for row in rows], dtype=torch.float32)
    if not torch.isfinite(scales).all() or (scales <= 0).any() or (integers == -128).any():
        raise ValueError('invalid quantization scale or integer range')
    received = integers*scales[:, None]
    if not torch.isfinite(received).all():
        raise ValueError('nonfinite reconstructed vector')
    return received
