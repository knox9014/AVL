"""Shared local English encoders with synchronous learned vector communication."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def encode_local_texts(rows):
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('expected nonempty rectangular local text rows')
    if any(not isinstance(s, str) or not s or any(ord(c) < 32 or ord(c) > 126 for c in s)
           for row in rows for s in row):
        raise ValueError('local texts must be nonempty printable ASCII')
    lengths = torch.tensor([[len(s) for s in row] for row in rows])
    tokens = torch.zeros(len(rows), len(rows[0]), int(lengths.max()), dtype=torch.long)
    for b, row in enumerate(rows):
        for n, s in enumerate(row):
            tokens[b, n, :len(s)] = torch.tensor([ord(c) for c in s])
    return tokens, lengths


class EnglishConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 48, padding_idx=0)
        self.encoder = nn.GRU(48, 64, batch_first=True)
        self.sender = nn.Linear(64, 16)
        self.query = nn.Linear(64, 16)
        self.key = nn.Linear(16, 16)
        self.receiver = nn.Linear(16, 64)
        self.update = nn.GRUCell(128, 64)
        self.decoder = nn.GRU(48, 64, batch_first=True)
        self.head = nn.Linear(64, 128)

    def communicate(self, tokens, lengths, rounds=2, mode='connected', output_units=None):
        if mode not in ('connected', 'blocked', 'serial') or type(rounds) is not int or rounds < 0:
            raise ValueError('invalid communication configuration')
        if tokens.ndim != 3 or tokens.dtype != torch.long or min(tokens.shape) < 1:
            raise ValueError('tokens must be nonempty long [batch, units, length]')
        b, n, l = tokens.shape
        if lengths.shape != (b, n) or lengths.dtype != torch.long or (lengths < 1).any() or (lengths > l).any():
            raise ValueError('invalid local lengths')
        if (tokens < 0).any() or (tokens >= 128).any():
            raise ValueError('token outside ASCII vocabulary')
        if output_units is None:
            output_units = torch.zeros(b, dtype=torch.long, device=tokens.device)
        if output_units.shape != (b,) or output_units.dtype != torch.long or (output_units < 0).any() or (output_units >= n).any():
            raise ValueError('invalid output unit')
        packed = pack_padded_sequence(self.embedding(tokens.reshape(b*n, l)),
                                     lengths.reshape(-1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        context = hidden[0].reshape(b, n, 64)
        states = context
        messages = []
        for _ in range(rounds):
            message = torch.tanh(self.sender(states))
            messages.append(message)
            if mode == 'blocked' or n == 1:
                inbox = torch.zeros_like(states)
            else:
                scores = self.query(states) @ self.key(message).transpose(1, 2) / math.sqrt(16)
                scores = scores.masked_fill(torch.eye(n, device=tokens.device, dtype=torch.bool), -torch.inf)
                inbox = self.receiver(scores.softmax(-1) @ message)
            inputs = torch.cat([context, inbox], -1)
            if mode == 'serial':
                states = torch.stack([self.update(inputs[:, i], states[:, i]) for i in range(n)], 1)
            else:
                states = self.update(inputs.reshape(b*n, 128), states.reshape(b*n, 64)).reshape(b, n, 64)
        output = states[torch.arange(b, device=tokens.device), output_units]
        return dict(state=output, node_states=states,
                    messages=torch.stack(messages, 1) if messages else states.new_empty(b, 0, n, 16),
                    logical_payload_bytes=0 if mode == 'blocked' else b*rounds*n*(n-1)*16*states.element_size())

    def forward(self, tokens, lengths, decoder_inputs, rounds=2, mode='connected', output_units=None):
        if decoder_inputs.ndim != 2 or decoder_inputs.shape[0] != tokens.shape[0] or decoder_inputs.shape[1] < 1 or decoder_inputs.dtype != torch.long:
            raise ValueError('invalid decoder inputs')
        if (decoder_inputs < 0).any() or (decoder_inputs >= 128).any():
            raise ValueError('decoder token outside vocabulary')
        result = self.communicate(tokens, lengths, rounds, mode, output_units)
        outputs, _ = self.decoder(self.embedding(decoder_inputs), result['state'].unsqueeze(0))
        result['logits'] = self.head(outputs)
        return result

    @torch.no_grad()
    def generate(self, tokens, lengths, max_new_tokens=32, **kw):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('positive generation limit required')
        result = self.communicate(tokens, lengths, **kw)
        hidden = result['state'].unsqueeze(0)
        current = torch.ones(tokens.shape[0], 1, dtype=torch.long, device=tokens.device)
        ids = [[] for _ in range(tokens.shape[0])]
        finished = [False] * tokens.shape[0]
        for _ in range(max_new_tokens):
            outputs, hidden = self.decoder(self.embedding(current), hidden)
            current = self.head(outputs).argmax(-1)
            for b, token in enumerate(current[:, 0].tolist()):
                if not finished[b]:
                    ids[b].append(token)
                    finished[b] = token == 2
            if all(finished):
                break
        result.update(token_ids=ids, texts=[''.join(chr(t) for t in row if 32 <= t <= 126) for row in ids])
        return result
