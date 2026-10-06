"""Versioned, ordered float32 vector packets. No text or labels on the wire."""
import struct
import torch

HEADER = struct.Struct('<4sBBHI')
MAX_SEGMENTS = 1024


def pack_vectors(vectors):
    if vectors.ndim != 2 or vectors.shape[1] != 16 or not 1 <= vectors.shape[0] <= MAX_SEGMENTS:
        raise ValueError('expected 1..1024 ordered vectors of width16')
    if vectors.dtype != torch.float32 or not torch.isfinite(vectors).all():
        raise ValueError('finite float32 vectors required')
    values = vectors.detach().cpu().contiguous().reshape(-1).tolist()
    return HEADER.pack(b'AVL1', 1, 1, 16, vectors.shape[0]) + struct.pack('<'+'f'*len(values), *values)


def unpack_vectors(packet):
    if not isinstance(packet, bytes) or len(packet) < HEADER.size:
        raise ValueError('truncated or invalid packet')
    magic, version, dtype, width, count = HEADER.unpack_from(packet)
    if (magic, version, dtype, width) != (b'AVL1', 1, 1, 16) or not 1 <= count <= MAX_SEGMENTS:
        raise ValueError('unsupported vector packet header')
    if len(packet) != HEADER.size + count*64:
        raise ValueError('vector packet size mismatch')
    values = struct.unpack_from('<'+'f'*(count*16), packet, HEADER.size)
    result = torch.tensor(values, dtype=torch.float32).reshape(count, 16)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite received vector')
    return result
