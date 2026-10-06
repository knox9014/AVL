"""Render a predicted frame as canonical English without original text.

This deterministic presentation layer preserves the supplied frame, not the
original wording. It cannot establish whether model predictions are correct.
No source corpus, label mapping, checkpoint or inference model is consulted.
"""


_CLASSES = {
    'kind': ('fact', 'request'),
    'subject': ('lamp', 'heater', 'fan', 'door', 'budget'),
    'predicate': ('on', 'off', 'open', 'closed', 'limit'),
    'polarity': ('positive', 'negative'),
    'certainty': ('certain', 'possible'),
    'condition': ('none', 'rain', 'cold', 'night'),
    'amount': ('none', '10', '20', '50', '100'),
    'comparator': ('none', 'at-most', 'at-least', 'exact'),
    'unit': ('none', 'USD'),
}
_CONDITIONS = {'none': '', 'rain': 'When it rains, ',
               'cold': 'When it is cold, ', 'night': 'At night, '}
_COMPARATORS = {'at-most': 'at most', 'at-least': 'at least', 'exact': 'exactly'}


def render_meaning(frame):
    """Validate all nine fields and render their complete meaning or raise.

    Impossible cross-field combinations are rejected, never repaired or partly
    discarded. Possible negation describes possibility of a negated proposition
    rather than prohibition or impossibility. Conditions govern the full segment.
    """
    if not isinstance(frame, dict) or frame.keys() != _CLASSES.keys():
        raise ValueError('expected exactly the nine semantic frame fields')
    for field, classes in _CLASSES.items():
        if not isinstance(frame[field], str) or frame[field] not in classes:
            raise ValueError('invalid semantic class for ' + field)
    if frame['subject'] == 'budget':
        if (frame['predicate'] != 'limit' or frame['polarity'] != 'positive'
                or frame['amount'] == 'none' or frame['comparator'] == 'none'
                or frame['unit'] != 'USD'):
            raise ValueError('inconsistent budget frame')
        proposition = 'the budget limit is {} {} USD'.format(
            _COMPARATORS[frame['comparator']], frame['amount'])
    else:
        predicates = ('open', 'closed') if frame['subject'] == 'door' else ('on', 'off')
        if (frame['predicate'] not in predicates or any(
                frame[field] != 'none' for field in ('amount', 'comparator', 'unit'))):
            raise ValueError('inconsistent device frame')
        proposition = 'the {} is {}{}'.format(
            frame['subject'], 'not ' if frame['polarity'] == 'negative' else '',
            frame['predicate'])
    if frame['kind'] == 'fact':
        body = 'It is {} that {}.'.format(frame['certainty'], proposition)
    elif frame['certainty'] == 'certain':
        body = 'Please make sure that {}.'.format(proposition)
    else:
        body = 'Consider making sure that {}.'.format(proposition)
    prefix = _CONDITIONS[frame['condition']]
    return prefix + body[0].lower() + body[1:] if prefix else body
