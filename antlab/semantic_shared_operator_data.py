"""Bounded chain grammar, explicit routing and fresh four-node transfer fixtures."""
import hashlib
import itertools
import json
import math
import re
from .semantic_binding_data import validate_table as validate_pair_names, canonicalize, TEMPLATES, RELATIONS
from .semantic_learned_query_data import make_data as old_data

def validate_names(table):
    if not isinstance(table,(list,tuple)) or len(table) not in (3,4):
        raise ValueError('expected3or4 chain identifiers')
    for pair in itertools.combinations(table,2):
        validate_pair_names(pair)
    if any(name.lower() in ('slot2','slot3','and') for name in table):
        raise ValueError('reserved identifier')

def validate_edges(edges,n):
    if n not in (3,4) or len(edges)!=n-1:
        raise ValueError('bounded chain size mismatch')
    adjacency={i:[] for i in range(n)}
    for edge in edges:
        if len(edge)!=2 or any(type(i) is not int or i not in range(n) for i in edge) or edge[0]>=edge[1]:
            raise ValueError('expected canonical endpoints')
        a,b=edge
        if b in adjacency[a]:
            raise ValueError('duplicate edge')
        adjacency[a].append(b)
        adjacency[b].append(a)
    if any(not neighbors or len(neighbors)>2 for neighbors in adjacency.values()):
        raise ValueError('expected chain degrees')
    seen=set()
    stack=[0]
    while stack:
        node=stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(adjacency[node])
    if len(seen)!=n:
        raise ValueError('disconnected chain')

def find_route(edges,source,target,n):
    validate_edges(edges,n)
    if any(type(i) is not int or i not in range(n) for i in (source,target)) or source==target:
        raise ValueError('invalid directed query')
    stack=[(source,[],{source})]
    while stack:
        node,path,seen=stack.pop()
        if node==target:
            return path
        for index,(a,b) in enumerate(edges):
            if node==a and b not in seen:
                stack.append((b,path+[(index,False)],seen|{b}))
            elif node==b and a not in seen:
                stack.append((a,path+[(index,True)],seen|{a}))
    raise ValueError('query route absent')

def table_size(table):
    validate_names(table)
    return sum(2+len(name) for name in table)

def edge_rows(text,table):
    validate_names(table)
    if not isinstance(text,str) or not text.isascii() or len(text)>1600:
        raise ValueError('invalid bounded source')
    clauses=text.split(' And ')
    if len(clauses)!=len(table)-1:
        raise ValueError('clause count mismatch')
    rows=[]
    for clause in clauses:
        words=re.findall(r'[A-Za-z0-9_-]+',clause)
        slots=[i for i,name in enumerate(table) if name in words]
        if len(slots)!=2:
            raise ValueError('expected two clause identifiers')
        local=[table[i] for i in slots]
        canonicalize(clause,local)
        rows.append(dict(text=clause,table=local,slots=slots))
    validate_edges([r['slots'] for r in rows],len(table))
    return rows

def make_chain(names,relations,orientation,order,template,reverse=0):
    validate_names(names)
    n=len(names)
    if (len(relations)!=n-1 or any(type(r) is not int or r not in range(4) for r in relations)
            or orientation not in range(1<<(n-1)) or order not in range(math.factorial(n))
            or template not in (0,3) or reverse not in (0,1)):
        raise ValueError('invalid fixture parameters')
    clauses=[]
    for i,r in enumerate(relations):
        a,b=names[i:i+2]
        if orientation & (1<<i):
            a,b=b,a
            r^=1
        clauses.append(TEMPLATES[template].format(a=a,b=b,r=RELATIONS[r]))
    if reverse:
        clauses.reverse()
    permutation=list(itertools.permutations(range(n)))[order]
    return dict(names=list(names),relations=list(relations),orientation=orientation,
                order=order,template=template,reverse=reverse,
                table=[names[i] for i in permutation],text=' And '.join(clauses))

def answer(row,pair):
    n=len(row['names'])
    if len(pair)!=2 or any(type(i) is not int or i not in range(n) for i in pair) or pair[0]==pair[1]:
        raise ValueError('invalid fixture query')
    source,target=pair
    path=row['relations'][min(source,target):max(source,target)]
    relation=path[0] if len(set(path))==1 else 4
    return relation^1 if source>target and relation!=4 else relation

def expand_queries(rows):
    return [dict(row=i,pair=[a,b],target=answer(row,(a,b)),distance=abs(a-b),
                 kind='direct' if abs(a-b)==1 else 'path')
            for i,row in enumerate(rows) for a,b in itertools.permutations(range(len(row['names'])),2)]

def quadruples(prefix):
    names=[prefix+str(i).zfill(2) for i in range(8)]
    return sorted(itertools.combinations(names,4),key=lambda group:hashlib.sha256(
        ('avl-chain-v2:'+':'.join(group)).encode('ascii')).hexdigest())[:2]

def make_data():
    old=old_data()
    data={k:old[k] for k in ('train','joint','phrasing','new_joint')}
    for split,prefix in (('known4','obj'),('new4','new')):
        data[split]=[make_chain(names,relations,orientation,order,3)
            for names,relations,orientation,order in itertools.product(quadruples(prefix),
                itertools.product(range(4),repeat=3),(0,7),(0,9,14,23))]
    return data

def select_pairs(rows):
    groups={}
    for row in rows:
        groups.setdefault(tuple(row['relations']),[]).append(row)
    count=4 if len(rows[0]['names'])==3 else 1
    return [row for key in sorted(groups) for row in sorted(groups[key],key=lambda r:hashlib.sha256(
        json.dumps(r,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest())[:count]]

def remade(rows,kind):
    result=[]
    for row in rows:
        n=len(row['names'])
        relations=list(row['relations'])
        orientation,order,reverse=row['orientation'],row['order'],row['reverse']
        if kind=='equivalent':
            orientation^=(1<<(n-1))-1
        elif kind=='table':
            order=(order+1)%math.factorial(n)
        elif kind=='clause_order':
            reverse^=1
        elif kind=='replacement':
            relations[0]^=1
        else:
            raise ValueError('unknown intervention')
        result.append(make_chain(row['names'],relations,orientation,order,row['template'],reverse))
    return result
