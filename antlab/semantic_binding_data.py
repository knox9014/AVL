"""Balanced two-instance spatial binding data; literal adapter is explicit."""
import hashlib
import itertools
import re
import struct

RELATIONS = ('left of','right of','above','below')
TEMPLATES = ('{a} is {r} {b}.','{a} is {r} {b} now.',
             'Now {a} is {r} {b}.','{a} now is {r} {b}.')
IDENTIFIER = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}\Z')

def validate_table(table):
    if (not isinstance(table,(list,tuple)) or len(table)!=2 or table[0]==table[1]
            or any(not isinstance(n,str) or not IDENTIFIER.fullmatch(n)
                   or n.lower() in ('slot0','slot1','is','of','left','right','above','below','now')
                   for n in table)):
        raise ValueError('expected two distinct non-reserved ASCII identifiers')

def canonicalize(text,table):
    validate_table(table)
    if not isinstance(text,str) or not text.isascii() or len(text)>512:
        raise ValueError('expected bounded ASCII source')
    tokens = re.findall(r'[A-Za-z0-9_-]+',text)
    if any(tokens.count(name)!=1 for name in table):
        raise ValueError('each table identifier must occur once')
    mapping = {name:'slot'+str(i) for i,name in enumerate(table)}
    result = [mapping[token] if token in mapping else token.lower() for token in tokens]
    if any(token not in ('slot0','slot1','is','of','left','right','above','below','now')
           for token in result):
        raise ValueError('unsupported word')
    return result

def table_bytes(table):
    validate_table(table)
    return sum(2+len(name.encode('ascii')) for name in table)

def make_row(pair,relation,orientation,template,order):
    pair = list(pair)
    a,b = pair if orientation==0 else pair[::-1]
    spoken = relation if orientation==0 else relation^1
    table = pair if order==0 else pair[::-1]
    return dict(pair=pair, relation=relation, orientation=orientation,
                template=template, order=order, table=list(table),
                text=TEMPLATES[template].format(a=a,b=b,r=RELATIONS[spoken]),
                target=relation if order==0 else relation^1,
                scene=[*pair,relation])

def counterpart(row,kind):
    args = [row['pair'],row['relation'],row['orientation'],row['template'],row['order']]
    if kind=='equivalent':
        args[2] ^= 1
    elif kind=='opposite':
        args[1] ^= 1
    elif kind=='table':
        args[4] ^= 1
    else:
        raise ValueError('unknown intervention')
    return make_row(*args)

def make_data():
    pairs = list(itertools.combinations(['obj'+str(i).zfill(2) for i in range(16)],2))
    pairs.sort(key=lambda pair:hashlib.sha256(('avl-binding-v1:'+':'.join(pair)).encode('ascii')).hexdigest())
    reserved,train = pairs[:30],pairs[30:]
    new = list(itertools.combinations(['new'+str(i).zfill(2) for i in range(8)],2))
    partitions = dict(train=(train,(0,1)),validation=(train,(2,)),
                      pair=(reserved,(0,1)),joint=(reserved,(3,)),
                      new_names=(new,(3,)),phrasing=(train,(3,)))
    return {split:[make_row(pair,relation,orientation,template,order)
                   for pair,relation,orientation,template,order in
                   itertools.product(group,range(4),range(2),templates,range(2))]
            for split,(group,templates) in partitions.items()}
