"""Fresh receiver learning with frozen sender features and no pair calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import torch
from torch import nn
from .semantic_binding_data import make_data,canonicalize
from .semantic_binding_pilot import targets_for,score,digest,canonical_hash
from .semantic_grounded_pilot import ARMS,SEEDS,FINAL,VectorNet,encode,wire,run as reproduce
from .semantic_acquisition_data import training_bank,select_support,summarize_thresholds

PROBES=(0,1,5,10,20,50,100)


def fingerprint(model):
    value=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value.update(name.encode('ascii'))
        value.update(str(tuple(tensor.shape)).encode('ascii'))
        values=tensor.detach().cpu().reshape(-1).tolist()
        value.update(struct.pack('<'+'f'*len(values),*values))
    return value.hexdigest()


def new_receiver(width,seed):
    if width not in (2,16): raise ValueError('unsupported receiver width')
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,4))


def train_receiver(features,targets,validation,validation_targets,source,seed,deadline):
    if features.requires_grad or any(v.requires_grad for v in validation.values()):
        raise ValueError('sender features must be detached and frozen')
    if features.ndim!=2 or len(features)!=len(targets) or source not in validation:
        raise ValueError('inconsistent receiver training inputs')
    model=new_receiver(features.shape[1],seed)
    optim=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.0001)
    started=time.monotonic(); curve=[]
    for step in range(101):
        deadline()
        if step:
            model.train(); optim.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(features),targets)
            if not torch.isfinite(loss): raise ValueError('nonfinite receiver loss')
            loss.backward()
            if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(),1.)):
                raise ValueError('nonfinite receiver gradient')
            optim.step()
        if step in PROBES:
            model.eval()
            with torch.no_grad():
                metrics={str(s):score(validation_targets,model(z).argmax(-1)) for s,z in validation.items()}
            others=[m for s,m in metrics.items() if int(s)!=source]
            curve.append(dict(step=step,examples_seen=step*len(features),senders=metrics,
                own_macro=metrics[str(source)]['macro_recall'],
                other_macro_min=min(m['macro_recall'] for m in others),
                other_macro_mean=sum(m['macro_recall'] for m in others)/len(others)))
    return model,dict(steps=100,examples=len(features),examples_seen=100*len(features),
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,curve=curve,
        first_sampled_own_95=next((p['step'] for p in curve if p['own_macro']>=.95),None),
        first_sampled_other_95=next((p['step'] for p in curve if p['other_macro_min']>=.95),None))


def compact_report(report):
    """Lossless dictionary encoding of repeated full confusion/support metrics."""
    dictionary={}; lookup={}
    def walk(value):
        if type(value) is dict and set(value)=={'accuracy','macro_recall','support','recall','confusion'}:
            key=json.dumps(value,sort_keys=True,separators=(',',':'))
            if key not in lookup:
                label='m'+str(len(lookup)); lookup[key]=label; dictionary[label]=value
            return {'metric_id':lookup[key]}
        if type(value) is dict: return {k:walk(v) for k,v in value.items()}
        if type(value) is list: return [walk(v) for v in value]
        return value
    body=walk(report)
    return dict(format='avl-receiver-acquisition-compact-v1',metric_dictionary=dictionary,report=body)


def expand_report(packed):
    if packed.get('format')!='avl-receiver-acquisition-compact-v1':
        raise ValueError('unsupported evidence encoding')
    def walk(value):
        if type(value) is dict and set(value)=={'metric_id'}:
            return packed['metric_dictionary'][value['metric_id']]
        if type(value) is dict: return {k:walk(v) for k,v in value.items()}
        if type(value) is list: return [walk(v) for v in value]
        return value
    return walk(packed['report'])


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=900: raise TimeoutError('declared900second acquisition budget exceeded')
    torch.set_num_threads(2)
    reproduction=reproduce(output/'sender-reproduction')
    root=Path(__file__).resolve().parents[1]
    prior=json.loads((root/'antlab/runs/grounded-vector-report-20261009.json').read_text(encoding='utf-8'))
    if reproduction['source_hashes']!=prior['source_hashes'] or reproduction['dataset_sha256']!=prior['dataset_sha256']:
        raise ValueError('sender source or dataset differs from preregistered prior configuration')
    deadline()
    data=make_data(); bank=training_bank(data['train'])
    vocabulary=reproduction['vocabulary']
    paths=['antlab/semantic_acquisition_data.py','antlab/semantic_acquisition_pilot.py',
           'antlab/tests/test_semantic_acquisition_data.py','antlab/tests/test_semantic_acquisition_pilot.py',
           'docs/AVL_RECEIVER_ACQUISITION_PROTOCOL.md']
    report=dict(format='avl-receiver-acquisition-v1',completed=False,pilot_passed=False,
        scope='fresh receiver acquisition of four known classes; shared supervised grounding, finite observed fixtures',
        torch=torch.__version__,threads=2,source_hashes={p:digest(root/p) for p in paths},
        dataset_sha256=canonical_hash(data),reproduction=dict(completed=reproduction['completed'],
            pilot_passed=reproduction['pilot_passed'],source_data_match_prior=True,checkpoint_file_match_prior={}),
        selected_support={},arms={})
    for selection in (101,202,303):
        report['selected_support'][str(selection)]={str(budget):[dict(text=r['text'],table=r['table'],target=r['target'],
             canonical=canonicalize(r['text'],r['table'])) for r in select_support(bank,budget,selection)] for budget in (4,8,16)}
    report['support_sha256']=canonical_hash(report['selected_support'])
    for arm,(width,contract) in ARMS.items():
        models={}; before={}; cached={}; entry=dict(width=width,contract=contract,trials=[],budgets={})
        report['arms'][arm]=entry
        report['reproduction']['checkpoint_file_match_prior'][arm]={}
        for sender in SEEDS:
            path=output/'sender-reproduction'/f'{arm}-{sender}.pt'
            record=torch.load(path,map_location='cpu',weights_only=True)
            if (record['arm'],record['width'],record['contract'],record['vocabulary'])!=(arm,width,contract,vocabulary):
                raise ValueError('sender checkpoint contract mismatch')
            model=VectorNet(len(vocabulary),width,contract)
            model.load_state_dict(record['model_state'],strict=True); model.eval(); model.requires_grad_(False)
            if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('nonfinite frozen sender')
            models[sender]=model; before[str(sender)]=fingerprint(model)
            report['reproduction']['checkpoint_file_match_prior'][arm][str(sender)]=digest(path)==prior['arms'][arm]['seeds'][str(sender)]['checkpoint_sha256']
            for split in ('validation',*FINAL):
                cached[sender,split]=wire(encode(model,data[split],vocabulary),contract)
        for source in SEEDS:
            validation={s:cached[s,'validation'] for s in SEEDS}
            validation_targets=targets_for(data['validation'])
            for selection in (101,202,303):
                for budget in (4,8,16):
                    rows=select_support(bank,budget,selection)
                    support=wire(encode(models[source],rows,vocabulary),contract)
                    targets=targets_for(rows)
                    for receiver_seed in (401,402,403):
                        deadline()
                        receiver,training=train_receiver(support,targets,validation,validation_targets,source,receiver_seed,deadline)
                        name=f'{arm}-sender{source}-selection{selection}-n{budget}-receiver{receiver_seed}.pt'
                        torch.save(dict(format=report['format'],arm=arm,source=source,selection=selection,
                            budget=budget,receiver_seed=receiver_seed,receiver_state=receiver.state_dict()),output/name)
                        trial=dict(source=source,selection=selection,budget=budget,receiver_seed=receiver_seed,
                            training=training,checkpoint_sha256=digest(output/name),final={})
                        with torch.no_grad():
                            for split in FINAL:
                                y=targets_for(data[split]); trial['final'][split]={}
                                for sender in SEEDS:
                                    z=cached[sender,split]
                                    trial['final'][split][str(sender)]=dict(transmitted=score(y,receiver(z).argmax(-1)),
                                        zero=score(y,receiver(torch.zeros_like(z)).argmax(-1)),
                                        shuffled=score(y,receiver(z.roll(1,0)).argmax(-1)))
                        trial['source_passed']=all(trial['final'][s][str(source)]['transmitted']['macro_recall']>=.95 for s in FINAL)
                        trial['transfer_passed']=all(trial['final'][s][str(sender)]['transmitted']['macro_recall']>=.95 for s in FINAL for sender in SEEDS if sender!=source)
                        entry['trials'].append(trial)
        for budget in (4,8,16):
            trials=[t for t in entry['trials'] if t['budget']==budget]
            entry['budgets'][str(budget)]=dict(trials=len(trials),
                source_success_fraction=sum(t['source_passed'] for t in trials)/len(trials),
                transfer_success_fraction=sum(t['transfer_passed'] for t in trials)/len(trials),
                own_acquisition=summarize_thresholds([t['training']['first_sampled_own_95'] for t in trials]),
                other_acquisition=summarize_thresholds([t['training']['first_sampled_other_95'] for t in trials]),
                validation_curve=[dict(step=step,
                    own_macro_mean=sum(t['training']['curve'][i]['own_macro'] for t in trials)/len(trials),
                    other_macro_min_mean=sum(t['training']['curve'][i]['other_macro_min'] for t in trials)/len(trials)) for i,step in enumerate(PROBES)])
        entry['gates']=dict(source=all(b['source_success_fraction']>=.9 for b in entry['budgets'].values()),
                            transfer=all(b['transfer_success_fraction']>=.9 for b in entry['budgets'].values()))
        entry['frozen_before']=before
        entry['frozen_after']={str(s):fingerprint(m) for s,m in models.items()}
        if entry['frozen_before']!=entry['frozen_after']: raise ValueError('sender weights mutated')
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report.update(completed=True,pilot_passed=all(all(report['arms'][arm]['gates'].values()) for arm in ('grounded16','grounded2')),
                  total_seconds=time.monotonic()-started)
    deadline()
    packed=compact_report(report)
    if expand_report(packed)!=report: raise ValueError('evidence compression changed metrics')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'compact-report.json').write_text(json.dumps(packed,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return packed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2,allow_nan=False))

