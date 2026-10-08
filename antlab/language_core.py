"""AVL-L1: exact bounded spatial semantics; independent of learned vectors.

Finite rank assignments represent every weak order on each real coordinate
axis. Negation is comparison complement, including ties, never inversion.
"""
import itertools
import json
import re

VERSION = 'AVL-L1'
RELATIONS = ('left', 'right', 'above', 'below')
IDENTIFIER = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}\Z')
RESERVED = {'assert', 'ask', 'not', *RELATIONS}
ATOM = re.compile(r'(not )?([A-Za-z][A-Za-z0-9_-]{0,63}) (left|right|above|below) ([A-Za-z][A-Za-z0-9_-]{0,63})\Z')
MAX_BYTES = 16384


def validate(program):
    """Reject unsupported structures before any interpretation."""
    if type(program) is not dict or set(program) != {'version','names','assertions','query'}:
        raise ValueError('expected exact AVL-L1 program fields')
    if program['version'] != VERSION:
        raise ValueError('unsupported version')
    names = program['names']
    if (type(names) is not list or not 2 <= len(names) <= 4
            or any(type(n) is not str or not IDENTIFIER.fullmatch(n)
                   or n.lower() in RESERVED for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('expected 2-4 distinct non-reserved ASCII names')
    assertions = program['assertions']
    if type(assertions) is not list or len(assertions) > 32:
        raise ValueError('expected at most 32 assertions')
    for atom in assertions + [program['query']]:
        if type(atom) is not dict or set(atom) != {'subject','relation','object','negated'}:
            raise ValueError('expected exact atom fields')
        if (type(atom['subject']) is not str or type(atom['object']) is not str
                or atom['subject'] not in names or atom['object'] not in names
                or atom['subject'] == atom['object']
                or type(atom['relation']) is not str or atom['relation'] not in RELATIONS
                or type(atom['negated']) is not bool):
            raise ValueError('unsupported atom')
    return program


def parse(source, names):
    """Strict textual frontend. Unsupported input raises ValueError."""
    if type(source) is not str or not source.isascii() or len(source) > MAX_BYTES:
        raise ValueError('expected bounded ASCII source')
    source = source.strip()
    if not source.endswith(';'):
        raise ValueError('expected terminal semicolon')
    statements = source[:-1].split(';')
    if not 1 <= len(statements) <= 33:
        raise ValueError('expected assertions then exactly one query')
    atoms = []
    for index, statement in enumerate(statements):
        statement = statement.strip()
        prefix = 'ask ' if index == len(statements)-1 else 'assert '
        if not statement.startswith(prefix):
            raise ValueError('expected assertions then exactly one query')
        match = ATOM.fullmatch(statement[len(prefix):])
        if not match:
            raise ValueError('unsupported grammar')
        negated, subject, relation, obj = match.groups()
        atoms.append(dict(subject=subject, relation=relation, object=obj, negated=bool(negated)))
    return validate(dict(version=VERSION, names=names, assertions=atoms[:-1], query=atoms[-1]))


def _truth(atom, ranks, indices):
    a, b = ranks[indices[atom['subject']]], ranks[indices[atom['object']]]
    value = a < b if atom['relation'] in ('left','below') else a > b
    return not value if atom['negated'] else value


def _axis(atom):
    return 0 if atom['relation'] in ('left','right') else 1


def evaluate(program):
    """Return semantic status and witnesses; never infer from inconsistency."""
    validate(program)
    names = program['names']; indices = {name:i for i,name in enumerate(names)}
    models = []
    for axis in range(2):
        constraints = [a for a in program['assertions'] if _axis(a) == axis]
        models.append([ranks for ranks in itertools.product(range(len(names)),repeat=len(names))
                       if all(_truth(a,ranks,indices) for a in constraints)])
    result = dict(version=VERSION, status='inconsistent', true_model=None, false_model=None)
    if not all(models):
        return result
    axis = _axis(program['query'])
    for ranks in models[axis]:
        key = 'true_model' if _truth(program['query'],ranks,indices) else 'false_model'
        if result[key] is None:
            x,y = (ranks,models[1][0]) if axis == 0 else (models[0][0],ranks)
            result[key] = {name:[x[i],y[i]] for i,name in enumerate(names)}
    result['status'] = ('undetermined' if result['true_model'] is not None and result['false_model'] is not None
                        else 'entailed' if result['true_model'] is not None else 'refuted')
    return result


def dumps(program):
    """Canonical symbolic transport; this is not the AVL1 vector codec."""
    validate(program)
    return json.dumps(program,sort_keys=True,separators=(',',':'),ensure_ascii=True)


def loads(payload):
    if type(payload) is not str or len(payload.encode('utf-8')) > MAX_BYTES:
        raise ValueError('expected bounded UTF-8 JSON text')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    try:
        program = json.loads(payload,object_pairs_hook=unique)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid JSON') from exc
    return validate(program)


def execute(source, names):
    return evaluate(loads(dumps(parse(source,names))))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.source,args.names),sort_keys=True))
    except ValueError as exc:
        parser.exit(2, 'unsupported: '+str(exc)+'\n')

