"""Frozen AVL v2 finite grammar, targets and training-only lexical vocabulary.

Single-proposition meaning rules and field classes remain those of v1. This
study broadens surface forms and condition placement; it does not parse general
English. Admission stores strings only and returns no semantic interpretation.
"""

import hashlib
import itertools
import json
import re
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

# Indices 0..7: training; 8/9: validation; 10/11: phrasing test.
# Combination-reserved frames use training indices 0/1 for combination, and
# heldout indices 10/11 for the joint meaning-and-phrasing test. All heldout words
# occur in training; generalization concerns surface placement and combinations.
TEMPLATES = {
    'fact': (
        'It is {certainty} that {proposition}.',
        'Now, it is {certainty} that {proposition}.',
        'That {proposition} is {certainty}.',
        'The statement that {proposition} is {certainty}.',
        'It is {certainty} that {proposition} now.',
        'The statement that {proposition} is now {certainty}.',
        'Now it is {certainty} that {proposition}.',
        '{certainty_title}: {proposition}.',
        'Now, the statement that {proposition} is {certainty}.',
        'It is now {certainty} that {proposition}.',
        'That {proposition} is now {certainty}.',
        'The statement that {proposition} is {certainty} now.',
    ),
    'request-certain': (
        'Please make sure that {proposition}.',
        'Please ensure that {proposition}.',
        'Make sure that {proposition}, please.',
        'Ensure that {proposition}, please.',
        'Now, please make sure that {proposition}.',
        'Please make sure that {proposition} now.',
        'Now, please ensure that {proposition}.',
        'Please ensure that {proposition} now.',
        'Now make sure that {proposition}, please.',
        'Now ensure that {proposition}, please.',
        'Please, now make sure that {proposition}.',
        'Please, now ensure that {proposition}.',
    ),
    'request-possible': (
        'Consider making sure that {proposition}.',
        'Think about ensuring that {proposition}.',
        'Consider ensuring that {proposition}.',
        'Think about making sure that {proposition}.',
        'Now, consider making sure that {proposition}.',
        'Consider making sure that {proposition} now.',
        'Now, think about ensuring that {proposition}.',
        'Think about ensuring that {proposition} now.',
        'Now consider ensuring that {proposition}.',
        'Now think about making sure that {proposition}.',
        'Consider, now, making sure that {proposition}.',
        'Think about, now, ensuring that {proposition}.',
    ),
}
CONDITIONS = {'none': '', 'rain': 'when it rains', 'cold': 'when it is cold',
              'night': 'at night'}
COMPARATOR_WORDS = {'at-most': 'at most', 'at-least': 'at least', 'exact': 'exactly'}


def normalize_text(text):
    """Lowercase and collapse whitespace without changing the retained source."""
    return ' '.join(text.lower().split())


def _frames():
    for kind, subject, polarity, certainty, condition in itertools.product(
            VOCABS[0], VOCABS[1][:-1], VOCABS[3], VOCABS[4], VOCABS[5]):
        for predicate in (('open', 'closed') if subject == 'door' else ('on', 'off')):
            yield dict(zip(FIELDS, (kind, subject, predicate, polarity, certainty,
                                   condition, 'none', 'none', 'none')))
    for kind, certainty, condition, amount, comparator in itertools.product(
            VOCABS[0], VOCABS[4], VOCABS[5], VOCABS[6][1:], VOCABS[7][1:]):
        yield dict(zip(FIELDS, (kind, 'budget', 'limit', 'positive', certainty,
                               condition, amount, comparator, 'USD')))


def _reserved(frame):
    canonical = json.dumps(frame, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(('avl-v2:' + canonical).encode('ascii')).hexdigest()
    return int(digest, 16) % 5 == 0


def _render(frame, index):
    if frame['subject'] == 'budget':
        proposition = 'the budget limit is {} {} USD'.format(
            COMPARATOR_WORDS[frame['comparator']], frame['amount'])
    else:
        proposition = 'the {} is {}{}'.format(
            frame['subject'], 'not ' if frame['polarity'] == 'negative' else '',
            frame['predicate'])
    family = 'fact' if frame['kind'] == 'fact' else 'request-' + frame['certainty']
    body = TEMPLATES[family][index].format(
        proposition=proposition, certainty=frame['certainty'],
        certainty_title=frame['certainty'].capitalize())
    condition = CONDITIONS[frame['condition']]
    if not condition:
        return body
    # Either position governs the entire statement, never a nested proposition.
    if index % 2 == 0:
        return condition.capitalize() + ', ' + body[0].lower() + body[1:]
    return body[:-1] + ', ' + condition + '.'


def _row(frame, index):
    text = _render(frame, index)
    return dict(id=hashlib.sha256(normalize_text(text).encode('ascii')).hexdigest(),
                text=text,
                labels=[vocab.index(frame[field]) for field, vocab in zip(FIELDS, VOCABS)],
                frame=dict(frame))


def make_data():
    """Return fresh JSON-native rows with frozen frame/surface partitions."""
    result = {split: [] for split in ('train', 'validation', 'combination', 'phrasing', 'joint')}
    seen = set()
    for frame in _frames():
        assignments = (('combination', 0), ('combination', 1),
                       ('joint', 10), ('joint', 11)) if _reserved(frame) else (
            tuple(('train', i) for i in range(8)) +
            (('validation', 8), ('validation', 9), ('phrasing', 10), ('phrasing', 11)))
        for split, index in assignments:
            row = _row(frame, index)
            if row['id'] in seen:
                raise ValueError('Frozen v2 grammar duplicated normalized full text')
            seen.add(row['id'])
            result[split].append(row)
    return result


# No validation, combination or phrasing examples participate in vocabulary
# construction. A word encoder can reserve its own padding/OOV indices.
WORDS = tuple(sorted({word for row in make_data()['train']
                      for word in re.findall(r'[a-z]+|[0-9]+', normalize_text(row['text']))}))


@lru_cache(maxsize=1)
def _admitted_texts():
    return frozenset(normalize_text(_render(frame, index))
                     for frame in _frames() for index in range(12))


def is_supported(text):
    """Return only boolean membership in the generated grammar, never labels."""
    return (isinstance(text, str) and text.isascii()
            and normalize_text(text) in _admitted_texts())
