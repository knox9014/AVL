"""Learn a bounded English meaning channel; receiver accepts vectors only."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

CARDINALITIES = (2, 5, 5, 2, 2, 4, 5, 4, 2)


class SemanticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 48, padding_idx=0)
        self.encoder = nn.GRU(48, 64, batch_first=True)
        self.sender = nn.Sequential(nn.Linear(64, 16), nn.LayerNorm(16))
        self.field_embedding = nn.Embedding(9, 16)
        self.receiver = nn.Sequential(nn.Linear(32, 243), nn.GELU(),
                                      nn.Linear(243, 243), nn.GELU(), nn.Linear(243, 8))
        mask = torch.arange(8)[None, :] >= torch.tensor(CARDINALITIES)[:, None]
        self.register_buffer('invalid_classes', mask, persistent=False)

    def encode(self, tokens, lengths):
        if tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) < 1:
            raise ValueError('tokens must be nonempty long [segments, characters]')
        if lengths.shape != (tokens.shape[0],) or lengths.dtype != torch.long:
            raise ValueError('invalid lengths')
        if (lengths < 1).any() or (lengths > tokens.shape[1]).any():
            raise ValueError('invalid lengths')
        if (tokens < 0).any() or (tokens >= 128).any():
            raise ValueError('invalid ASCII token')
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return self.sender(hidden[0])

    def decode(self, vectors):
        if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] != 16:
            raise ValueError('receiver requires nonempty [segments,16] vectors')
        if not vectors.is_floating_point() or not torch.isfinite(vectors).all():
            raise ValueError('receiver requires finite floating vectors')
        fields = self.field_embedding(torch.arange(9, device=vectors.device))
        inputs = torch.cat([vectors[:, None, :].expand(-1, 9, -1),
                            fields[None, :, :].expand(vectors.shape[0], -1, -1)], -1)
        logits = self.receiver(inputs)
        if not torch.isfinite(logits.masked_select(~self.invalid_classes[None])).all():
            raise ValueError('nonfinite valid receiver logits')
        return logits.masked_fill(self.invalid_classes[None], -torch.inf)

    def forward(self, tokens, lengths):
        vectors = self.encode(tokens, lengths)
        return dict(vectors=vectors, logits=self.decode(vectors))
