"""Frozen synthetic grammar for AVL v1; admission never supplies labels.

The source defines the entire grammar before training. Each text expresses one
complete frame, with explicit certainty and proposition-level negation. A
leading condition applies to the complete fact or request. Generated labels
are training/evaluation targets only, never an inference admission result.
"""

import hashlib
import itertools
import json
from functools import lru_cache


FIELDS = ('kind', 'subject', 'predicate', 'polarity', 'certainty', 'condition',
          'amount', 'comparator', 'unit')
VOCABS = (
    ('fact', 'request'),
    ('lamp', 'heater', 'fan', 'door', 'budget'),
    ('on', 'off', 'open', 'closed', 'limit'),
    ('positive', 'negative'),
    ('certain', 'possible'),
    ('none', 'rain', 'cold', 'night'),
    ('none', '10', '20', '50', '100'),
    ('none', 'at-most', 'at-least', 'exact'),
    ('none', 'USD'),
)

# Each heldout template rearranges words already present in training templates.
# Template indices 0/1/2 are training, 3 validation, and 4 phrasing test.
TEMPLATES = {
    'fact': (
        'It is {certainty} that {proposition}.',
        'Now, it is {certainty} that {proposition}.',
        'It is {certainty} now that {proposition}.',
        'It is {certainty} that, now, {proposition}.',
        'It is {certainty}, now, that {proposition}.',
    ),
    'request-certain': (
        'Please make sure that {proposition}.',
        'Please, now make sure that {proposition}.',
        'Now make sure that {proposition}, please.',
        'Make sure that {proposition} now, please.',
        'Now, please make sure that {proposition}.',
    ),
    'request-possible': (
        'Consider making sure that {proposition}.',
        'Consider, now, making sure that {proposition}.',
        'Now consider making sure that {proposition}.',
        'Consider making sure that {proposition} now.',
        'Now, consider making sure that {proposition}.',
    ),
}
CONDITION_PREFIXES = {
    'none': '', 'rain': 'When it rains, ',
    'cold': 'When it is cold, ', 'night': 'At night, ',
}
COMPARATOR_WORDS = {'at-most': 'at most', 'at-least': 'at least', 'exact': 'exactly'}


def normalize_text(text):
    """Canonical ASCII source identity: lowercase and collapse whitespace."""
    return ' '.join(text.lower().split())


def _frames():
    for kind, subject, polarity, certainty, condition in itertools.product(
            VOCABS[0], VOCABS[1][:-1], VOCABS[3], VOCABS[4], VOCABS[5]):
        predicates = ('open', 'closed') if subject == 'door' else ('on', 'off')
        for predicate in predicates:
            yield dict(zip(FIELDS, (kind, subject, predicate, polarity, certainty,
                                   condition, 'none', 'none', 'none')))
    for kind, certainty, condition, amount, comparator in itertools.product(
            VOCABS[0], VOCABS[4], VOCABS[5], VOCABS[6][1:], VOCABS[7][1:]):
        yield dict(zip(FIELDS, (kind, 'budget', 'limit', 'positive', certainty,
                               condition, amount, comparator, 'USD')))


def _combination_holdout(frame):
    canonical = json.dumps(frame, sort_keys=True, separators=(',', ':'))
    return int(hashlib.sha256(canonical.encode('ascii')).hexdigest(), 16) % 5 == 0


def _render(frame, template_index):
    if frame['subject'] == 'budget':
        proposition = 'the budget limit is {} {} USD'.format(
            COMPARATOR_WORDS[frame['comparator']], frame['amount'])
    else:
        proposition = 'the {} is {}{}'.format(
            frame['subject'], 'not ' if frame['polarity'] == 'negative' else '',
            frame['predicate'])
    template_key = ('fact' if frame['kind'] == 'fact'
                    else 'request-' + frame['certainty'])
    body = TEMPLATES[template_key][template_index].format(
        certainty=frame['certainty'], proposition=proposition)
    return CONDITION_PREFIXES[frame['condition']] + body


def _row(frame, template_index):
    text = _render(frame, template_index)
    return {
        'id': hashlib.sha256(normalize_text(text).encode('ascii')).hexdigest(),
        'text': text,
        'labels': [vocab.index(frame[field]) for field, vocab in zip(FIELDS, VOCABS)],
        'frame': dict(frame),
    }


def make_data():
    """Return fresh JSON-native rows with fixed frame and phrasing partitions.

    Combination tests use training template 0 on frames absent from every other
    split. Validation/phrasing contain all nonreserved frames exactly once.
    Train contains each nonreserved frame three times with distinct surfaces.
    """
    data = {name: [] for name in ('train', 'validation', 'combination', 'phrasing')}
    seen = set()
    for frame in _frames():
        assignments = (('combination', 0),) if _combination_holdout(frame) else (
            ('train', 0), ('train', 1), ('train', 2), ('validation', 3), ('phrasing', 4))
        for split, template_index in assignments:
            row = _row(frame, template_index)
            if row['id'] in seen:
                raise ValueError('Frozen grammar produced duplicate normalized source text')
            seen.add(row['id'])
            data[split].append(row)
    return data


@lru_cache(maxsize=1)
def _admitted_texts():
    # This stores only normalized strings, never a mapping to semantic frames.
    return frozenset(normalize_text(_render(frame, index))
                     for frame in _frames() for index in range(5))


def is_supported(text):
    """Boolean grammar admission, with no semantic labels returned to inference.

    The caller retains the original source unchanged. Case/whitespace variants
    are admitted; arbitrary prose, new entities/numbers and compound scopes are
    rejected. Admission covers the frozen grammar, including unseen frames.
    """
    return (isinstance(text, str) and text.isascii()
            and normalize_text(text) in _admitted_texts())
