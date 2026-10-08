"""Finite directed graph queries with symmetry-closed heldout relation tuples."""
import itertools
from .semantic_composition_data import triples, make_row, infer_chain

HELDOUT={(0,0),(1,1),(0,2),(3,1)}
QUERIES=((0,1),(1,0),(1,2),(2,1),(0,2),(2,0))

def validate_pair(pair):
    if len(pair)!=2 or any(type(i) is not int or i not in range(3) for i in pair) or pair[0]==pair[1]:
        raise ValueError('expected distinct directed node indices')

def answer(row,pair):
    validate_pair(pair)
    source,target=pair
    if abs(source-target)==1:
        relation=row['relations'][min(source,target)]
    else:
        relation=infer_chain(*row['relations'])
    return relation^1 if source>target and relation!=4 else relation

def expand_queries(rows):
    return [dict(row=i,pair=list(pair),target=answer(row,pair),
                 kind='direct' if abs(pair[0]-pair[1])==1 else 'path')
            for i,row in enumerate(rows) for pair in QUERIES]

def semantic_signature(row,pair):
    validate_pair(pair)
    source,target=pair
    other=next(i for i in range(3) if i not in pair)
    mapping={source:0,target:1,other:2}
    edges=[]
    for i,relation in enumerate(row['relations']):
        a,b=mapping[i],mapping[i+1]
        if a>b:
            a,b=b,a
            relation^=1
        edges.append((a,b,relation))
    return tuple(sorted(edges))

def make_data():
    combinations=list(itertools.product(range(4),repeat=2))
    seen=[pair for pair in combinations if pair not in HELDOUT]
    partitions=dict(train=(triples('obj'),seen,0),
                    joint=(triples('obj'),sorted(HELDOUT),3),
                    phrasing=(triples('obj'),seen,3),
                    new_names=(triples('new'),combinations,3),
                    new_joint=(triples('new'),sorted(HELDOUT),3))
    return {split:[make_row(names,relations,orientation,order,template)
                   for names,relations,orientation,order in itertools.product(group,relations,range(4),range(6))]
            for split,(group,relations,template) in partitions.items()}
