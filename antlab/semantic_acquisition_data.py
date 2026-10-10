"""Training-only, balanced support selection for frozen vector acquisition."""
import hashlib
import json
import statistics
from .semantic_binding_data import canonicalize


def training_bank(rows):
    bank={}
    for row in rows:
        key=tuple(canonicalize(row['text'],row['table']))
        if type(row['target']) is not int or row['target'] not in range(4):
            raise ValueError('expected four-class training target')
        if key in bank and bank[key]['target']!=row['target']:
            raise ValueError('conflicting canonical supervision')
        bank.setdefault(key,row)
    if len(bank)!=16 or [sum(r['target']==c for r in bank.values()) for c in range(4)]!=[4]*4:
        raise ValueError('expected16 unique training forms, four per class')
    return bank


def select_support(bank,budget,seed):
    if type(budget) is not int or budget not in (4,8,16) or type(seed) is not int:
        raise ValueError('unsupported support selection')
    selected=[]
    for label in range(4):
        keys=[k for k,r in bank.items() if r['target']==label]
        keys.sort(key=lambda k:hashlib.sha256(json.dumps(['avl-acquisition-v1',seed,k],separators=(',',':')).encode('ascii')).hexdigest())
        if len(keys)!=4:
            raise ValueError('unbalanced training bank')
        selected.extend(bank[k] for k in keys[:budget//4])
    return selected


def summarize_thresholds(values):
    if not values or any(v is not None and (type(v) is not int or v<0) for v in values):
        raise ValueError('expected nonempty sampled thresholds with explicit censoring')
    reached=[v for v in values if v is not None]
    return dict(trials=len(values),reached=len(reached),fraction=len(reached)/len(values),
                conditional_median_step=float(statistics.median(reached)) if reached else None)

