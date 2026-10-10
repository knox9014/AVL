"""Synthetic three-entity fixtures and explicit lexical graph scaffolding."""
import hashlib
import itertools
import re
from .semantic_binding_data import TEMPLATES, RELATIONS, canonicalize, validate_table

HELDOUT={(0,2),(2,0),(1,3),(3,1)}
PERMUTATIONS=tuple(itertools.permutations(range(3)))

def validate_names(table):
    if not isinstance(table,(list,tuple)) or len(table)!=3:
        raise ValueError('expected three distinct identifiers')
    for pair in itertools.combinations(table,2):
        validate_table(pair)
    if any(name.lower() in ('slot2','and') for name in table):
        raise ValueError('reserved identifier')

def table_size(table):
    validate_names(table)
    return sum(2+len(name.encode('ascii')) for name in table)

def validate_edges(edges):
    if len(edges)!=2 or any(len(e)!=2 or any(type(i) is not int or i not in range(3) for i in e)
                           or e[0]>=e[1] for e in edges):
        raise ValueError('expected two canonical edges')
    if tuple(edges[0])==tuple(edges[1]) or set(i for e in edges for i in e)!={0,1,2}:
        raise ValueError('expected connected three-node chain')

def edge_rows(text,table):
    validate_names(table)
    if not isinstance(text,str) or not text.isascii() or len(text)>1100:
        raise ValueError('expected bounded ASCII clauses')
    clauses=text.split(' And ')
    if len(clauses)!=2:
        raise ValueError('expected two explicitly separated clauses')
    result=[]
    for index,clause in enumerate(clauses):
        words=re.findall(r'[A-Za-z0-9_-]+',clause)
        slots=[i for i,name in enumerate(table) if name in words]
        if len(slots)!=2:
            raise ValueError('expected two distinct clause endpoints')
        local=[table[i] for i in slots]
        canonicalize(clause,local)
        result.append(dict(text=clause,table=local,slots=slots,clause_index=index))
    validate_edges([e['slots'] for e in result])
    return result

def infer_chain(first,second):
    if type(first) is not int or type(second) is not int or first not in range(4) or second not in range(4):
        raise ValueError('unknown primitive relation')
    return first if first==second else 4

def make_row(names,relations,orientation,order,template,reverse=0):
    validate_names(names)
    if (len(relations)!=2 or any(type(r) is not int or r not in range(4) for r in relations)
            or orientation not in range(4) or order not in range(6)
            or template not in (0,3) or reverse not in (0,1)):
        raise ValueError('invalid fixture parameters')
    clauses=[]
    for i,relation in enumerate(relations):
        a,b=names[i:i+2]
        if orientation & (1<<i):
            a,b=b,a
            relation^=1
        clauses.append(TEMPLATES[template].format(a=a,b=b,r=RELATIONS[relation]))
    if reverse:
        clauses.reverse()
    return dict(names=list(names),relations=list(relations),orientation=orientation,
                order=order,template=template,reverse=reverse,
                table=[names[i] for i in PERMUTATIONS[order]],text=' And '.join(clauses),
                target_tuple=4*relations[0]+relations[1],query=infer_chain(*relations))

def triples(prefix):
    group=list(itertools.combinations([prefix+str(i).zfill(2) for i in range(6)],3))
    return sorted(group,key=lambda names:hashlib.sha256(
        ('avl-composition-v1:'+':'.join(names)).encode('ascii')).hexdigest())[:4]

def make_data():
    combinations=list(itertools.product(range(4),repeat=2))
    seen=[r for r in combinations if r not in HELDOUT]
    partitions=dict(development=(triples('obj'),seen,0),
                    joint=(triples('obj'),sorted(HELDOUT),3),
                    phrasing=(triples('obj'),seen,3),
                    new_names=(triples('new'),combinations,3))
    return {split:[make_row(names,relations,orientation,order,template)
                   for names,relations,orientation,order in itertools.product(group,relations,range(4),range(6))]
            for split,(group,relations,template) in partitions.items()}
