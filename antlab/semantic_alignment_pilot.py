"""Fixed-budget supervised calibration of frozen finite AVL binding codebooks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .semantic_binding_data import make_data, canonicalize
from .semantic_binding_pilot import (BindingNet, TESTS, digest, canonical_hash, infer,
                                     targets_for, score, wire_vectors)

def select_calibration(rows,count,seed):
    if type(count) is not int or count not in (4,8,16):
        raise ValueError('expected4/8/16 examples')
    unique={}
    for row in rows:
        key=tuple(canonicalize(row['text'],row['table']))
        if key in unique and unique[key]['target']!=row['target']:
            raise ValueError('inconsistent canonical meaning')
        unique.setdefault(key,row)
    groups=[]
    for label in range(4):
        group=[row for row in unique.values() if row['target']==label]
        group.sort(key=lambda row:hashlib.sha256(
            (str(seed)+':'+ ' '.join(canonicalize(row['text'],row['table']))).encode('ascii')).hexdigest())
        if len(group)<count//4:
            raise ValueError('insufficient distinct per-class calibration examples')
        groups.append(group)
    return [groups[label][index] for index in range(count//4) for label in range(4)]

def fit_ridge(x,y):
    if (x.ndim!=2 or y.ndim!=2 or x.shape[1]!=16 or len(x)!=len(y) or len(x)<2
            or y.shape[1] not in (4,16) or x.dtype!=torch.float32 or y.dtype!=torch.float32
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
        raise ValueError('expected finite matching float32 matrices')
    design=torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),dim=1)
    regularizer=torch.eye(17,dtype=torch.float64)*.01
    regularizer[-1,-1]=0
    weights=torch.linalg.solve(design.T@design+regularizer,design.T@y.double()).float()
    if not torch.isfinite(weights).all():
        raise ValueError('nonfinite ridge weights')
    return weights

def linear_output(x,weights):
    if (x.ndim!=2 or x.shape[1]!=16 or weights.ndim!=2 or weights.shape[0]!=17
            or not torch.isfinite(x).all() or not torch.isfinite(weights).all()):
        raise ValueError('invalid affine application')
    return (torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),dim=1)@weights.double()).float()

def apply_ridge(x,weights):
    if weights.shape!=(17,16):
        raise ValueError('bridge must be17x16')
    return wire_vectors(linear_output(x,weights))

def load_frozen(study,seed,vocabulary,expected):
    path=study/('seed-'+str(seed)+'.pt')
    if digest(path)!=expected:
        raise ValueError('checkpoint hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='avl-binding-pilot-v1' or saved.get('seed')!=seed or saved.get('vocabulary')!=vocabulary:
        raise ValueError('checkpoint identity mismatch')
    model=BindingNet(len(vocabulary))
    model.load_state_dict(saved['model_state'],strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError('nonfinite frozen weights')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()

@torch.no_grad()
def run(study,output):
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=600:
            raise TimeoutError('calibration budget exceeded')
    torch.set_num_threads(2)
    study,output=Path(study),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((study/'report.json').read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[1]
    original=json.loads((root/'antlab/runs/binding-pilot-summary-20261008.json').read_text(encoding='utf-8'))
    data=make_data()
    if (previous.get('format')!='avl-binding-pilot-v1' or not previous.get('completed')
            or not previous.get('pilot_passed')
            or canonical_hash(data)!=previous['dataset_sha256']
            or previous['dataset_sha256']!=original['dataset_sha256']):
        raise ValueError('binding reproduction/data validation failed')
    for name in original['source_hashes']:
        if digest(root/name)!=original['source_hashes'][name] or previous['source_hashes'][name]!=original['source_hashes'][name]:
            raise ValueError('binding source changed')
    vocabulary=previous['vocabulary']
    checkpoints={seed:previous['seeds'][str(seed)]['checkpoint_sha256'] for seed in (44,55,66)}
    models={seed:load_frozen(study,seed,vocabulary,checkpoints[seed]) for seed in checkpoints}
    vectors={seed:{split:infer(model,data[split],vocabulary)[1] for split in TESTS}
             for seed,model in models.items()}
    selections={(count,cal):select_calibration(data['train'],count,cal)
                for count in (4,8,16) for cal in (101,202,303)}
    calibration_vectors={(seed,count,cal):infer(model,rows,vocabulary)[1]
                         for seed,model in models.items() for (count,cal),rows in selections.items()}
    paths=('antlab/semantic_alignment_pilot.py','antlab/tests/test_semantic_alignment_pilot.py',
           'docs/AVL_ALIGNMENT_PILOT_PROTOCOL.md')
    report=dict(format='avl-binding-calibration-v1',completed=False,pilot_passed=False,
                source_hashes={name:digest(root/name) for name in paths},
                binding_source_hashes=previous['source_hashes'],dataset_sha256=previous['dataset_sha256'],
                binding_report_sha256=digest(study/'report.json'),
                checkpoint_hashes={str(seed):value for seed,value in checkpoints.items()},
                reproduced_checkpoint_byte_match={str(seed):checkpoints[seed]==original['seeds'][str(seed)]['checkpoint_sha256'] for seed in checkpoints},
                frozen_weights=True,calibration_supervision='paired same-message model representations; explicit labels for fresh classifier',
                canonical_limit='16 unique train strings;8 unique heldout-template strings; no novel relation classes',
                bridge_parameters=272,bridge_raw_float32_bytes=1088,classifier_parameters=68,
                classifier_raw_float32_bytes=272,message_bytes_including_table=90,
                selections={str(count)+'/'+str(cal):[dict(text=r['text'],table=r['table'],
                    canonical=canonicalize(r['text'],r['table']),target=r['target']) for r in rows]
                    for (count,cal),rows in selections.items()},trials=[])
    saved_adapters={}
    for sender in models:
        for receiver in models:
            if sender==receiver:
                continue
            for (count,cal),rows in selections.items():
                deadline()
                x=calibration_vectors[sender,count,cal]
                y=calibration_vectors[receiver,count,cal]
                labels=targets_for(rows)
                if torch.any(labels==labels.roll(1)):
                    raise ValueError('wrong-pair control accidentally retains class')
                one_hot=torch.nn.functional.one_hot(labels,4).float()
                bridge=fit_ridge(x,y)
                wrong=fit_ridge(x,y.roll(1,0))
                fresh=fit_ridge(x,one_hot)
                wrong_fresh=fit_ridge(x,one_hot.roll(1,0))
                key=str(sender)+'/'+str(receiver)+'/'+str(count)+'/'+str(cal)
                saved_adapters[key]=dict(bridge=bridge,wrong_bridge=wrong,
                                         classifier=fresh,wrong_classifier=wrong_fresh)
                trial=dict(sender=sender,receiver=receiver,examples=count,selection_seed=cal,
                           calibration_pair_payload_bytes=166*count,
                           calibration_label_payload_bytes=91*count,evaluations={},gates={})
                for split in TESTS:
                    source=vectors[sender][split]
                    target=targets_for(data[split])
                    frozen=models[receiver]
                    predictions=dict(unadapted=frozen.receive(source).argmax(-1),
                                     bridge=frozen.receive(apply_ridge(source,bridge)).argmax(-1),
                                     wrong_bridge=frozen.receive(apply_ridge(source,wrong)).argmax(-1),
                                     fresh_classifier=linear_output(source,fresh).argmax(-1),
                                     wrong_classifier=linear_output(source,wrong_fresh).argmax(-1))
                    metrics={name:score(target,prediction) for name,prediction in predictions.items()}
                    trial['evaluations'][split]=metrics
                    if count==16:
                        trial['gates'][split]=dict(accuracy=metrics['bridge']['accuracy']>=.95,
                             macro_recall=metrics['bridge']['macro_recall']>=.95,
                             wrong_pair=metrics['wrong_bridge']['accuracy']<=.30,
                             gap=metrics['bridge']['accuracy']-metrics['wrong_bridge']['accuracy']>=.60)
                report['trials'].append(trial)
    for seed,expected in checkpoints.items():
        if digest(study/('seed-'+str(seed)+'.pt'))!=expected:
            raise ValueError('frozen checkpoint changed')
    filename=output/'adapters.pt'
    torch.save(dict(format=report['format'],adapters=saved_adapters),filename)
    report.update(completed=True,pilot_passed=all(all(all(g.values()) for g in trial['gates'].values())
                for trial in report['trials'] if trial['examples']==16),
                adapter_archive_sha256=digest(filename),adapter_archive_bytes=filename.stat().st_size,
                total_seconds=time.monotonic()-started)
    deadline()
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.study,args.output)
    summary={key:value for key,value in report.items() if key!='trials'}
    summary['trials']=[{**{key:value for key,value in trial.items() if key!='evaluations'},
                       'evaluations':{split:{name:dict(accuracy=m['accuracy'],macro_recall=m['macro_recall'],
                                   support=m['support'],recall=m['recall']) for name,m in metrics.items()}
                           for split,metrics in trial['evaluations'].items()}}
                        for trial in report['trials']]
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
