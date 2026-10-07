"""Experimental exact named literals around a frozen learned AVL relation vector."""
import re
import struct
import torch
from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_v2_data import FIELDS, VOCABS, is_supported
from .semantic_v2_model import tokenize
from .semantic_render import render_meaning

MAX_UINT64 = (1 << 64) - 1
HEADER = struct.Struct('<4sBBHI')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
SUBJECT = re.compile(r'the (lamp|heater|fan|door|budget) named "([^"]*)"', re.I)
AMOUNT = re.compile(r'(the budget limit is (?:at most|at least|exactly) )([0-9]+)( USD)', re.I)

def uint64(value):
    if type(value) is not int or not 0 <= value <= MAX_UINT64:
        raise ValueError('expected uint64 integer')
    return value

def canonicalize_named(text):
    if not isinstance(text, str) or not text.isascii():
        raise ValueError('expected ASCII controlled text')
    matches = list(SUBJECT.finditer(text))
    if len(matches) != 1 or not NAME.fullmatch(matches[0][2]):
        raise ValueError('expected one valid named subject')
    match = matches[0]
    entity = match[2]
    canonical = text[:match.start()] + 'the ' + match[1] + text[match.end():]
    amount = None
    if match[1].lower() == 'budget':
        amounts = list(AMOUNT.finditer(canonical))
        if len(amounts) != 1 or len(amounts[0][2]) > 20:
            raise ValueError('expected one uint64 budget literal')
        literal = amounts[0]
        amount = uint64(int(literal[2]))
        canonical = canonical[:literal.start(2)] + '20' + canonical[literal.end(2):]
    if not is_supported(canonical):
        raise ValueError('unsupported controlled grammar')
    return canonical, entity, amount

def pack_named(inner, entity, amount=None):
    if not isinstance(entity, str) or not NAME.fullmatch(entity):
        raise ValueError('invalid entity identifier')
    if not isinstance(inner, bytes) or len(inner) != 76 or unpack_vectors(inner).shape != (1, 16):
        raise ValueError('expected one inner AVL1 vector')
    name = entity.encode('ascii')
    literal = b'' if amount is None else struct.pack('<Q', uint64(amount))
    return HEADER.pack(b'AVN1', 1, int(amount is not None), len(name), len(inner)) + name + literal + inner

def unpack_named(packet):
    if not isinstance(packet, bytes) or len(packet) < HEADER.size:
        raise ValueError('truncated named packet')
    magic, version, flags, length, inner_length = HEADER.unpack_from(packet)
    if magic != b'AVN1' or version != 1 or flags not in (0, 1) or inner_length != 76:
        raise ValueError('unsupported named header')
    if not 1 <= length <= 64 or len(packet) != 12 + length + 8 * flags + inner_length:
        raise ValueError('invalid named packet length')
    try:
        entity = packet[12:12+length].decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('non-ASCII entity') from error
    if not NAME.fullmatch(entity):
        raise ValueError('invalid entity identifier')
    position = 12 + length
    amount = struct.unpack_from('<Q', packet, position)[0] if flags else None
    inner = packet[position+8*flags:]
    if unpack_vectors(inner).shape != (1, 16):
        raise ValueError('expected one inner vector')
    return inner, entity, amount

@torch.inference_mode()
def send_named_segments(model, texts):
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
        raise ValueError('expected nonempty source text list')
    plans, unsupported = [], []
    for index, text in enumerate(texts):
        try:
            canonical, entity, amount = canonicalize_named(text)
            plans.append((index, canonical, entity, amount))
        except ValueError:
            unsupported.append(dict(index=index, status='unsupported', text=text))
    unique = list(dict.fromkeys(p[1] for p in plans))
    encoded = {}
    for start in range(0, len(unique), 128):
        batch = unique[start:start+128]
        tokens, lengths = tokenize(batch)
        vectors = model.encode(tokens, lengths)
        encoded.update((text, pack_vectors(vector.unsqueeze(0))) for text, vector in zip(batch, vectors))
    return dict(sources=list(texts), supported_indices=[p[0] for p in plans],
                unsupported=unsupported,
                packets=[pack_named(encoded[text], entity, amount) for _, text, entity, amount in plans])

@torch.inference_mode()
def receive_named_packets(model, packets):
    if not isinstance(packets, list) or not packets:
        raise ValueError('expected nonempty named packet list')
    parsed = [unpack_named(packet) for packet in packets]
    frames = []
    for start in range(0, len(parsed), 128):
        batch = parsed[start:start+128]
        vectors = torch.cat([unpack_vectors(item[0]) for item in batch])
        decisions = model.decode(vectors).argmax(-1).tolist()
        for decision, (_, entity, amount) in zip(decisions, batch):
            frame = {field: vocab[index] for field, vocab, index in zip(FIELDS, VOCABS, decision)}
            render_meaning(frame)
            budget = frame['subject'] == 'budget'
            if budget != (amount is not None) or (budget and frame['amount'] != '20'):
                raise ValueError('literal and decoded relation disagree')
            if budget:
                frame['amount'] = str(amount)
            frame['entity'] = entity
            frames.append(frame)
    return frames

@torch.inference_mode()
def query_named_amount(operator, meaning, candidate):
    uint64(candidate)
    if meaning.get('subject') != 'budget':
        return dict(status='unsupported_subject')
    literal = meaning.get('amount')
    if not isinstance(literal, str) or not literal.isascii() or not literal.isdigit() or len(literal) > 20:
        raise ValueError('invalid restored integer')
    amount = uint64(int(literal))
    comparator = meaning.get('comparator')
    if comparator not in ('at-most', 'at-least', 'exact'):
        raise ValueError('invalid comparator')
    delta = amount - candidate
    if not -101 <= delta <= 101:
        return dict(status='unsupported_numeric_distance')
    features = torch.tensor([[1., float(comparator == 'at-most'), float(comparator == 'at-least'),
                              float(comparator == 'exact'), float(delta), float(abs(delta))]], dtype=torch.float32)
    prediction = operator(features).argmax(-1).item()
    return dict(status='model_prediction', answer=('undetermined', 'allowed', 'disallowed')[prediction])
