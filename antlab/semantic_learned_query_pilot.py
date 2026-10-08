"""Preregistered raw-vector neural receiver for finite three-entity AVL queries."""
import argparse
import json
from pathlib import Path
import struct
import time
import torch
from torch import nn
from .semantic_alignment_pilot import load_frozen
from .semantic_binding_pilot import digest, canonical_hash
from .semantic_composition_pilot import (encode_batch, unpack_message, classification,
                                         remade, fingerprint)
from .semantic_learned_query_data import make_data, expand_queries, semantic_signature

QUERY_HEADER=struct.Struct('<4sI')

def pack_query(message,source,target):
    unpack_message(message)
    if any(type(i) is not int or i not in range(3) for i in (source,target)) or source==target:
        raise ValueError('expected distinct query slots')
    return QUERY_HEADER.pack(b'ACQ1',len(message))+message+bytes((source,target))

def unpack_query(packet):
    if not isinstance(packet,bytes) or not 166<=len(packet)<=358:
        raise ValueError('invalid query packet length')
    magic,length=QUERY_HEADER.unpack_from(packet)
    if magic!=b'ACQ1' or len(packet)!=QUERY_HEADER.size+length+2:
        raise ValueError('invalid query framing')
    decoded=unpack_message(packet[QUERY_HEADER.size:QUERY_HEADER.size+length])
    source,target=packet[-2:]
    if source not in range(3) or target not in range(3) or source==target:
        raise ValueError('invalid directed query')
    decoded['query']=[source,target]
    return decoded

def features(packets):
    if not packets:
        raise ValueError('empty query input')
    result=torch.zeros((len(packets),50),dtype=torch.float32)
    for i,packet in enumerate(packets):
        decoded=unpack_query(packet)
        result[i,:32]=decoded['vectors'].reshape(-1)
        for j,index in enumerate(index for edge in decoded['edges'] for index in edge):
            result[i,32+3*j+index]=1
        for j,index in enumerate(decoded['query']):
            result[i,44+3*j+index]=1
    return result

class QueryReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(50,64),nn.GELU(),
                                   nn.Linear(64,64),nn.GELU(),nn.Linear(64,5))
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=50 or x.dtype!=torch.float32 or not torch.isfinite(x).all():
            raise ValueError('expected finite float32 query features width50')
        return self.network(x)

def balanced_indices(targets,generator,batch):
    if targets.ndim!=1 or targets.dtype!=torch.long or not len(targets) or batch<=0:
        raise ValueError('invalid balanced sampler inputs')
    if torch.any((targets<0)|(targets>4)):
        raise ValueError('invalid target class')
    groups=[torch.where(targets==i)[0] for i in range(5)]
    if any(not len(g) for g in groups):
        raise ValueError('all five output classes required')
    labels=torch.randint(5,(batch,),generator=generator)
    result=torch.empty(batch,dtype=torch.long)
    for label,group in enumerate(groups):
        mask=labels==label
        result[mask]=group[torch.randint(len(group),(int(mask.sum()),),generator=generator)]
    return result

@torch.no_grad()
def predict(model,x):
    return torch.cat([model(x[i:i+512]).argmax(-1) for i in range(0,len(x),512)]).tolist()

@torch.no_grad()
def rule_predict(primitive,packets):
    decoded=[unpack_query(p) for p in packets]
    vectors=torch.cat([d['vectors'] for d in decoded])
    labels=primitive.receive(vectors).argmax(-1).tolist()
    predictions=[]
    for i,d in enumerate(decoded):
        lookup={}
        for edge,label in zip(d['edges'],labels[2*i:2*i+2]):
            a,b=edge
            lookup[a,b]=label
            lookup[b,a]=label^1
        source,target=d['query']
        if (source,target) in lookup:
            prediction=lookup[source,target]
        else:
            middle=next(j for j in range(3) if j not in (source,target))
            first,second=lookup[source,middle],lookup[middle,target]
            prediction=first if first==second else 4
        predictions.append(prediction)
    return predictions

def query_packets(primitive,rows,vocabulary):
    messages=encode_batch(primitive,rows,vocabulary)
    queries=expand_queries(rows)
    packets=[pack_query(messages[q['row']],
                        rows[q['row']]['table'].index(rows[q['row']]['names'][q['pair'][0]]),
                        rows[q['row']]['table'].index(rows[q['row']]['names'][q['pair'][1]]))
             for q in queries]
    return packets,queries

def train(x,targets,seed,deadline):
    torch.manual_seed(seed)
    model=QueryReceiver()
    torch.manual_seed(seed+1000)
    baseline=QueryReceiver()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),lr=.003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    started=time.monotonic()
    for step in range(3000):
        deadline()
        indices=balanced_indices(targets,generator,128)
        batch=x[indices]
        no_vectors=batch.clone()
        no_vectors[:,:32]=0
        for receiver,opt,values in ((model,optimizer,batch),(baseline,baseline_optimizer,no_vectors)):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(receiver(values),targets[indices])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite query loss')
            loss.backward()
            if not torch.isfinite(nn.utils.clip_grad_norm_(receiver.parameters(),1.)):
                raise ValueError('nonfinite query gradients')
            opt.step()
        final_baseline_loss=float(loss.detach())
        with torch.no_grad():
            final_loss=float(nn.functional.cross_entropy(model(batch),targets[indices]))
    return model.eval(),baseline.eval(),dict(seed=seed,steps=3000,batch=128,
               parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,
               final_loss=final_loss,baseline_final_loss=final_baseline_loss,
               sampling='uniform class then uniform training example; same indices for both receivers')

def score_queries(queries,predictions):
    if len(queries)!=len(predictions):
        raise ValueError('query prediction count mismatch')
    groups=dict(all=list(range(len(queries))),
        direct=[i for i,q in enumerate(queries) if q['kind']=='direct'],
        path=[i for i,q in enumerate(queries) if q['kind']=='path'],
        positive_path=[i for i,q in enumerate(queries) if q['kind']=='path' and q['target']!=4],
        undetermined=[i for i,q in enumerate(queries) if q['target']==4])
    return {name:classification([queries[i]['target'] for i in indices],
                                [predictions[i] for i in indices],5)
            for name,indices in groups.items()}

def reversal_pairs(queries,predictions):
    eligible=[i for i,q in enumerate(queries) if q['target']!=4]
    if any(queries[i^1]['row']!=queries[i]['row'] or queries[i^1]['pair']!=queries[i]['pair'][::-1]
           for i in eligible):
        raise ValueError('query reversal indexing mismatch')
    return dict(eligible_directed_queries=len(eligible),
         both_correct=sum(predictions[i]==queries[i]['target'] and
                          predictions[i^1]==queries[i^1]['target'] for i in eligible)/len(eligible))

def run(study,output):
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=600:
            raise TimeoutError('learned query budget exceeded')
    torch.set_num_threads(2)
    study,output=Path(study),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((study/'report.json').read_text(encoding='utf-8'))
    original=json.loads((root/'antlab/runs/binding-pilot-summary-20261008.json').read_text(encoding='utf-8'))
    if (binding.get('format')!='avl-binding-pilot-v1' or not binding.get('completed')
            or not binding.get('pilot_passed') or binding['dataset_sha256']!=original['dataset_sha256']):
        raise ValueError('invalid primitive reproduction')
    for name,sha in original['source_hashes'].items():
        if binding['source_hashes'][name]!=sha or digest(root/name)!=sha:
            raise ValueError('primitive source configuration changed')
    vocabulary=binding['vocabulary']
    checkpoints={seed:binding['seeds'][str(seed)]['checkpoint_sha256'] for seed in (44,55,66)}
    primitives={seed:load_frozen(study,seed,vocabulary,sha) for seed,sha in checkpoints.items()}
    frozen={seed:fingerprint(m) for seed,m in primitives.items()}
    data=make_data()
    signatures={split:{semantic_signature(rows[q['row']],q['pair']) for q in expand_queries(rows)}
                for split,rows in data.items()}
    if signatures['train'] & (signatures['joint']|signatures['new_joint']):
        raise ValueError('semantic train/test leakage')
    paths=('antlab/semantic_learned_query_data.py','antlab/semantic_learned_query_pilot.py',
           'antlab/tests/test_semantic_learned_query_data.py','antlab/tests/test_semantic_learned_query_pilot.py',
           'antlab/semantic_composition_data.py','antlab/semantic_composition_pilot.py',
           'antlab/semantic_alignment_pilot.py','docs/AVL_LEARNED_COMPOSITION_QUERY_PROTOCOL.md',
           '.github/workflows/semantic-learned-query.yml')
    report=dict(format='avl-learned-composition-query-v1',completed=False,pilot_passed=False,
                raw_vector_neural_receiver=True,topology_and_grammar_designed=True,
                scope='tiny supervised finite three-node query task; no unseen concept or universal language claim',
                torch=torch.__version__,dataset_sha256=canonical_hash(data),
                source_hashes={p:digest(root/p) for p in paths},
                binding_source_hashes=original['source_hashes'],
                binding_report_sha256=digest(study/'report.json'),
                checkpoint_hashes=checkpoints,frozen_state_sha256=frozen,
                reproduced_checkpoint_byte_match={seed:checkpoints[seed]==original['seeds'][str(seed)]['checkpoint_sha256']
                                                  for seed in checkpoints},
                semantic_signatures={split:dict(count=len(sigs),train_overlap=len(sigs & signatures['train']),
                    values=sorted(sigs),sha256=canonical_hash(sorted(sigs))) for split,sigs in signatures.items()},
                partitions={},seeds={})
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for seed,primitive in primitives.items():
        deadline()
        train_packets,train_queries=query_packets(primitive,data['train'],vocabulary)
        x_train=features(train_packets)
        targets=torch.tensor([q['target'] for q in train_queries],dtype=torch.long)
        model,baseline,training=train(x_train,targets,seed+100,deadline)
        filename=output/('query-seed-'+str(seed)+'.pt')
        torch.save(dict(format=report['format'],sender_seed=seed,receiver_seed=seed+100,
                        receiver_state=model.state_dict(),no_vector_state=baseline.state_dict()),filename)
        saved=torch.load(filename,map_location='cpu',weights_only=True)
        loaded=QueryReceiver()
        loaded.load_state_dict(saved['receiver_state'],strict=True)
        if fingerprint(loaded)!=fingerprint(model):
            raise ValueError('receiver state serialization mismatch')
        entry=dict(training=training,checkpoint_sha256=digest(filename),checkpoint_bytes=filename.stat().st_size,
                   evaluations={},gates={})
        other=primitives[55 if seed==44 else 66 if seed==55 else 44]
        for split,rows in data.items():
            deadline()
            packets,queries=(train_packets,train_queries) if split=='train' else query_packets(primitive,rows,vocabulary)
            x=x_train if split=='train' else features(packets)
            prediction=predict(model,x)
            zero=x.clone()
            zero[:,:32]=0
            shuffled=x.clone()
            shuffled[:,:32]=x[:,:32].roll(7,0)
            no_topology=x.clone()
            no_topology[:,32:]=0
            wrong_packets,_=query_packets(other,rows,vocabulary)
            evaluation=dict(neural=score_queries(queries,prediction),
                 no_vector=score_queries(queries,predict(baseline,zero)),
                 zero=score_queries(queries,predict(model,zero)),
                 shuffled=score_queries(queries,predict(model,shuffled)),
                 no_topology_query=score_queries(queries,predict(model,no_topology)),
                 wrong_checkpoint=score_queries(queries,predict(model,features(wrong_packets))),
                 rule=score_queries(queries,rule_predict(primitive,packets)),
                 query_reversal=reversal_pairs(queries,prediction))
            if seed==44:
                transmitted=sum(map(len,packets))
                table_total=sum(3*(2+5) for _ in rows)*6
                report['partitions'][split]=dict(scenes=len(rows),queries=len(queries),
                    support=[int(sum(q['target']==i for q in queries)) for i in range(5)],
                    path_support=[int(sum(q['target']==i and q['kind']=='path' for q in queries)) for i in range(5)],
                    transmitted_bytes=transmitted,
                    symbolic_gold_bytes=43*len(queries),
                    text_reference_bytes=sum((len(row['text'].encode('ascii'))+21+10)*6 for row in rows),
                    graph_repeated_per_query=True,query_wrapper_and_indices_bytes=10*len(queries),
                    cost_scope='includes actual AVC1 graph/name/topology/vector and ACQ1 query wrapper; excludes model distribution/network framing')
                if transmitted!=181*len(queries) or table_total!=21*len(queries):
                    raise ValueError('query wire accounting mismatch')
            for kind in ('equivalent','table','clause_order','replacement'):
                changed_rows=remade(rows,kind)
                changed_packets,changed_queries=query_packets(primitive,changed_rows,vocabulary)
                changed=predict(model,features(changed_packets))
                changed_indices=[i for i,q in enumerate(queries) if q['target']!=changed_queries[i]['target']]
                evaluation[kind]=dict(metrics=score_queries(changed_queries,changed),
                    both_correct=sum(a==q['target'] and b==cq['target']
                                     for a,q,b,cq in zip(prediction,queries,changed,changed_queries))/len(queries),
                    changed_target_queries=len(changed_indices),
                    changed_target_accuracy=(sum(changed[i]==changed_queries[i]['target'] for i in changed_indices)/len(changed_indices)
                                             if changed_indices else None))
            entry['evaluations'][split]=evaluation
            if split!='train':
                neural=evaluation['neural']
                macro=neural['all']['macro_recall_observed']
                no_vector=evaluation['no_vector']['all']['macro_recall_observed']
                entry['gates'][split]=dict(accuracy=neural['all']['accuracy']>=.95,macro=macro>=.95,
                    direct=neural['direct']['accuracy']>=.95,
                    positive_path=neural['positive_path']['accuracy']>=.95,
                    undetermined=neural['undetermined']['accuracy']>=.95,
                    invariant=all(evaluation[k]['both_correct']>=.95 for k in ('equivalent','table','clause_order')),
                    replacement=evaluation['replacement']['metrics']['all']['accuracy']>=.95,
                    query_reversal=evaluation['query_reversal']['both_correct']>=.95,
                    no_vector=no_vector<=.50,gap=macro-no_vector>=.40)
            deadline()
        report['seeds'][str(seed)]=entry
    for seed,primitive in primitives.items():
        if fingerprint(primitive)!=frozen[seed] or digest(study/('seed-'+str(seed)+'.pt'))!=checkpoints[seed]:
            raise ValueError('frozen primitive changed')
    report.update(completed=True,pilot_passed=all(all(all(g.values()) for g in e['gates'].values())
                                                  for e in report['seeds'].values()),
                  total_seconds=time.monotonic()-started)
    deadline()
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.study,args.output),indent=2,allow_nan=False))

if __name__=='__main__':
    main()
