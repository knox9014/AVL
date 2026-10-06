"""Ordered word encoder; the receiver's only evidence is a sixteen-float vector."""
import re
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .semantic_v2_data import WORDS, normalize_text
from .semantic_model import CARDINALITIES

WORD_IDS = {word: index + 2 for index, word in enumerate(WORDS)}


def tokenize(texts):
    """Lexical tokenization only, with no semantic-frame lookup or labels.

    Vocabulary comes from training inputs only. Case and whitespace are
    normalized; punctuation is ignored. Public grammar admission is separate.
    Unknown words fail explicitly, rather than silently dropping meaning.
    """
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) or not t.isascii() for t in texts):
        raise ValueError('supply nonempty ASCII text list')
    rows = []
    for text in texts:
        words = re.findall(r'[a-z]+|[0-9]+', normalize_text(text))
        if not words or any(word not in WORD_IDS for word in words):
            raise ValueError('empty text or word outside training vocabulary')
        rows.append([WORD_IDS[word] for word in words])
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    tokens = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long)
    for index, row in enumerate(rows):
        tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long)
    return tokens, lengths


class SemanticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(WORDS) + 2, 64, padding_idx=0)
        self.encoder = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.sender = nn.Sequential(nn.Linear(128, 16), nn.LayerNorm(16))
        self.field_embedding = nn.Embedding(9, 16)
        self.receiver = nn.Sequential(nn.Linear(32, 192), nn.GELU(),
                                      nn.Linear(192, 192), nn.GELU(), nn.Linear(192, 8))
        mask = torch.arange(8)[None, :] >= torch.tensor(CARDINALITIES)[:, None]
        self.register_buffer('invalid_classes', mask, persistent=False)

    def encode(self, tokens, lengths):
        if tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) < 1:
            raise ValueError('tokens must be nonempty long matrix')
        if lengths.shape != (tokens.shape[0],) or lengths.dtype != torch.long:
            raise ValueError('invalid lengths')
        if (lengths < 1).any() or (lengths > tokens.shape[1]).any():
            raise ValueError('invalid lengths')
        if (tokens < 0).any() or (tokens >= self.embedding.num_embeddings).any():
            raise ValueError('word token outside vocabulary')
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        contextual, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(contextual, batch_first=True)
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None].to(hidden.device)
        attention = self.attention(hidden).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        attended = (hidden * attention[..., None]).sum(1)
        average = (hidden * mask[..., None]).sum(1) / lengths[:, None].to(hidden.device)
        return self.sender((attended + average) * .5)

    def decode(self, vectors):
        if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] != 16:
            raise ValueError('receiver requires nonempty [segments,16] vectors')
        if vectors.dtype != torch.float32 or not torch.isfinite(vectors).all():
            raise ValueError('receiver requires finite float32 vectors')
        fields = self.field_embedding(torch.arange(9, device=vectors.device))
        inputs = torch.cat([vectors[:, None, :].expand(-1, 9, -1),
                            fields[None].expand(vectors.shape[0], -1, -1)], -1)
        logits = self.receiver(inputs)
        if not torch.isfinite(logits.masked_select(~self.invalid_classes[None])).all():
            raise ValueError('nonfinite valid receiver logits')
        return logits.masked_fill(self.invalid_classes[None], -torch.inf)

    def forward(self, tokens, lengths):
        vectors = self.encode(tokens, lengths)
        return dict(vectors=vectors, logits=self.decode(vectors))
