"""Deterministic semantic-query baseline; not a learned AI receiver.

Accepts validated predicted or gold frames, never parses source text.
Gold use is a label oracle; predicted-frame use is a decoder-plus-rule baseline.
"""
from .semantic_render import render_meaning

SUBJECT_PREDICATES = {
    'lamp': ('on', 'off'), 'heater': ('on', 'off'),
    'fan': ('on', 'off'), 'door': ('open', 'closed'),
}
ANSWERS = ('entailed', 'contradicted', 'undetermined')


def answer_query(frame, query):
    """Answer an explicit bounded question without assuming world facts.

    proposition: is the positive subject/predicate atom established now?
    Requests, possibilities and conditional statements cannot establish it.
    Different predicates are unknown; no antonym equivalence is assumed.

    constraint_allows: does a candidate meet the described numeric constraint?
    This interprets the constraint, not its truth, applicability or enforcement.
    """
    render_meaning(frame)  # Reject inconsistent frames rather than repair them.
    if not isinstance(query, dict) or not isinstance(query.get('type'), str):
        raise ValueError('expected a typed query')
    kind = query['type']
    if kind in ('kind', 'certainty', 'condition'):
        if set(query) != {'type'}:
            raise ValueError('unexpected query fields')
        return frame[kind]
    if kind == 'proposition':
        if set(query) != {'type', 'subject', 'predicate'}:
            raise ValueError('expected proposition query fields')
        subject, predicate = query['subject'], query['predicate']
        if (not isinstance(subject, str) or subject not in SUBJECT_PREDICATES
                or not isinstance(predicate, str)
                or predicate not in SUBJECT_PREDICATES[subject]):
            raise ValueError('unsupported proposition query')
        if (frame['kind'] != 'fact' or frame['certainty'] != 'certain'
                or frame['condition'] != 'none'
                or frame['subject'] != subject or frame['predicate'] != predicate):
            return 'undetermined'
        return 'entailed' if frame['polarity'] == 'positive' else 'contradicted'
    if kind == 'constraint_allows':
        if set(query) != {'type', 'candidate', 'unit'}:
            raise ValueError('expected numeric query fields')
        candidate = query['candidate']
        if type(candidate) is not int or not 0 <= candidate <= 1000000:
            raise ValueError('candidate must be a bounded nonnegative integer')
        if query['unit'] != 'USD':
            raise ValueError('unsupported candidate unit')
        if frame['subject'] != 'budget':
            return 'undetermined'
        amount = int(frame['amount'])
        comparator = frame['comparator']
        allowed = (candidate <= amount if comparator == 'at-most' else
                   candidate >= amount if comparator == 'at-least' else
                   candidate == amount)
        return 'allowed' if allowed else 'disallowed'
    raise ValueError('unsupported query type')
