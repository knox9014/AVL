"""Shared learned relation operator on explicit bounded-chain AVL query packets."""
import argparse
import json
from pathlib import Path
import struct
import time
import torch
from torch import nn
from .semantic_codec import pack_vectors,unpack_vectors
from .semantic_binding_data import canonicalize
from .semantic_binding_pilot import digest,canonical_hash,encode_texts
from .semantic_alignment_pilot import load_frozen
from .semantic_composition_pilot import fingerprint
from .semantic_learned_query_pilot import balanced_indices,score_queries
from .semantic_learned_query_data import semantic_signature,expand_queries as old_expand
from .semantic_shared_operator_data import (validate_names,validate_edges,table_size,edge_rows,
    find_route,make_data,expand_queries,select_pairs,remade)

GRAPH_HEADER=struct.Struct('<4sBB')
QUERY_HEADER=struct.Struct('<4sI')
INVERSE=[1,0,3,2]

def pack_graph(table,edges,vectors):
    validate_names(table)
    validate_edges(edges,len(table))
    if vectors.shape!=(len(edges),16):
        raise ValueError('edge/vector count mismatch')
    names=b''.join(struct.pack('<H',len(n))+n.encode('ascii') for n in table)
    return GRAPH_HEADER.pack(b'AVC2',len(table),len(edges))+names+bytes(i for e in edges for i in e)+pack_vectors(vectors)

def unpack_graph(packet):
    if not isinstance(packet,bytes) or not 159<=len(packet)<=480:
        raise ValueError('invalid graph size')
    magic,n,e=GRAPH_HEADER.unpack_from(packet)
    if magic!=b'AVC2' or n not in (3,4) or e!=n-1:
        raise ValueError('invalid graph header')
    position=GRAPH_HEADER.size
    table=[]
    for _ in range(n):
        if position+2>len(packet):
            raise ValueError('truncated identifier')
        length=struct.unpack_from('<H',packet,position)[0]
        position+=2
        if not 1<=length<=64 or position+length>len(packet):
            raise ValueError('invalid identifier length')
        try:
            table.append(packet[position:position+length].decode('ascii'))
        except UnicodeDecodeError as error:
            raise ValueError('invalid identifier encoding') from error
        position+=length
    validate_names(table)
    if position+2*e>len(packet):
        raise ValueError('truncated topology')
    edges=[list(packet[position+2*i:position+2*i+2]) for i in range(e)]
    validate_edges(edges,n)
    vectors=unpack_vectors(packet[position+2*e:])
    if vectors.shape!=(e,16):
        raise ValueError('vector topology mismatch')
    return dict(table=table,edges=edges,vectors=vectors)

def pack_query(graph,source,target):
    decoded=unpack_graph(graph)
    n=len(decoded['table'])
    if any(type(i) is not int or i not in range(n) for i in (source,target)) or source==target:
        raise ValueError('invalid directed query')
    return QUERY_HEADER.pack(b'ACQ2',len(graph))+graph+bytes((source,target))

def unpack_query(packet):
    if not isinstance(packet,bytes) or not 169<=len(packet)<=490:
        raise ValueError('invalid query size')
    magic,length=QUERY_HEADER.unpack_from(packet)
    if magic!=b'ACQ2' or len(packet)!=QUERY_HEADER.size+length+2:
        raise ValueError('invalid query framing')
    decoded=unpack_graph(packet[QUERY_HEADER.size:QUERY_HEADER.size+length])
    source,target=packet[-2:]
    if source not in range(len(decoded['table'])) or target not in range(len(decoded['table'])) or source==target:
        raise ValueError('invalid query endpoints')
    decoded['query']=[source,target]
    return decoded

@torch.no_grad()
def graph_packets(primitive,rows,vocabulary):
    plans=[edge_rows(r['text'],r['table']) for r in rows]
    unique={}
    for plan in plans:
        for edge in plan:
            unique.setdefault(tuple(canonicalize(edge['text'],edge['table'])),edge)
    keys=list(unique)
    tokens,lengths=encode_texts(list(unique.values()),vocabulary)
    cache=dict(zip(keys,primitive.encode(tokens,lengths)))
    return [pack_graph(row['table'],[e['slots'] for e in plan],
             torch.stack([cache[tuple(canonicalize(e['text'],e['table']))] for e in plan]))
            for row,plan in zip(rows,plans)]

def query_packets(primitive,rows,vocabulary):
    graphs=graph_packets(primitive,rows,vocabulary)
    queries=expand_queries(rows)
    return [pack_query(graphs[q['row']],rows[q['row']]['table'].index(rows[q['row']]['names'][q['pair'][0]]),
                       rows[q['row']]['table'].index(rows[q['row']]['names'][q['pair'][1]]))
            for q in queries],queries

@torch.no_grad()
def prepare(primitive,packets,zero=False,shuffled=False):
    if not packets:
        raise ValueError('empty query list')
    decoded=[unpack_query(p) for p in packets]
    vectors=[d['vectors'] for d in decoded]
    if shuffled:
        groups={}
        for i,d in enumerate(decoded):
            groups.setdefault(len(d['table']),[]).append(i)
        shifted=list(vectors)
        for n,indices in groups.items():
            shift=n*(n-1)
            for j,index in enumerate(indices):
                shifted[index]=vectors[indices[(j-shift)%len(indices)]]
        vectors=shifted
    flat=torch.cat(vectors)
    if zero:
        flat=torch.zeros_like(flat)
    probabilities=torch.softmax(primitive.receive(flat),dim=-1)
    result=torch.zeros((len(packets),3,4),dtype=torch.float32)
    lengths=[]
    offset=0
    for i,d in enumerate(decoded):
        path=find_route(d['edges'],*d['query'],len(d['table']))
        lengths.append(len(path))
        for step,(edge,reverse) in enumerate(path):
            p=probabilities[offset+edge]
            result[i,step]=p[INVERSE] if reverse else p
        offset+=len(d['edges'])
    return result,torch.tensor(lengths,dtype=torch.long)

class SharedOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,1))
        self.unknown_bias=nn.Parameter(torch.zeros(1))
    def combine(self,first,second):
        if first.ndim!=2 or first.shape[1]!=4 or first.shape!=second.shape:
            raise ValueError('expected paired relation probabilities')
        direction=self.channel(torch.stack((first,second),dim=-1)).squeeze(-1)
        return torch.cat((direction,self.unknown_bias.view(1,1).expand(len(first),1)),dim=-1)
    def forward(self,p,lengths):
        if (p.ndim!=3 or p.shape[1:]!=(3,4) or p.dtype!=torch.float32
                or lengths.shape!=(len(p),) or lengths.dtype!=torch.long
                or not torch.isfinite(p).all() or torch.any((p<0)|(p>1))
                or torch.any((lengths<1)|(lengths>3))):
            raise ValueError('invalid routed probability input')
        logits=self.combine(p[:,0],p[:,0])
        second=self.combine(p[:,0],p[:,1])
        logits=torch.where((lengths>=2).unsqueeze(-1),second,logits)
        if torch.any(lengths>=3):
            third=self.combine(torch.softmax(logits,dim=-1)[:,:4],p[:,2])
            logits=torch.where((lengths>=3).unsqueeze(-1),third,logits)
        return logits

@torch.no_grad()
def predict(model,p,lengths):
    return torch.cat([model(p[i:i+512],lengths[i:i+512]).argmax(-1) for i in range(0,len(p),512)]).tolist()

def train(p,lengths,zero,targets,seed,deadline):
    torch.manual_seed(seed)
    model=SharedOperator()
    torch.manual_seed(seed+1000)
    baseline=SharedOperator()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),lr=.003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    started=time.monotonic()
    for _ in range(3000):
        deadline()
        indices=balanced_indices(targets,generator,128)
        for receiver,opt,values in ((model,optimizer,p),(baseline,baseline_optimizer,zero)):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(receiver(values[indices],lengths[indices]),targets[indices])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite operator loss')
            loss.backward()
            if not torch.isfinite(nn.utils.clip_grad_norm_(receiver.parameters(),1.)):
                raise ValueError('nonfinite operator gradients')
            opt.step()
        baseline_loss=float(loss.detach())
    with torch.no_grad():
        main_loss=float(nn.functional.cross_entropy(model(p[indices],lengths[indices]),targets[indices]))
    return model.eval(),baseline.eval(),dict(seed=seed,steps=3000,batch=128,
            parameters=sum(v.numel() for v in model.parameters()),final_loss=main_loss,
            baseline_final_loss=baseline_loss,seconds=time.monotonic()-started)

def metrics(queries,predictions):
    from .semantic_composition_pilot import classification
    result=score_queries(queries,predictions)
    for name,indices in dict(distance2=[i for i,q in enumerate(queries) if q['distance']==2],
                             distance3=[i for i,q in enumerate(queries) if q['distance']==3],
                             positive_distance3=[i for i,q in enumerate(queries) if q['distance']==3 and q['target']!=4]).items():
        result[name]=({**classification([queries[i]['target'] for i in indices],
                                       [predictions[i] for i in indices],5),'records':len(indices)}
                      if indices else score_queries([],[])['all'])
    return result

def reversal(queries,predictions):
    lookup={(q['row'],tuple(q['pair'])):i for i,q in enumerate(queries)}
    eligible=[i for i,q in enumerate(queries) if q['target']!=4]
    return dict(eligible_directed_queries=len(eligible),both_correct=sum(
        predictions[i]==q['target'] and predictions[lookup[q['row'],tuple(q['pair'][::-1])]]==
        queries[lookup[q['row'],tuple(q['pair'][::-1])]]['target']
        for i,q in enumerate(queries) if q['target']!=4)/len(eligible))

def rule(p,lengths):
    labels=p.argmax(-1).tolist()
    return [values[0] if len(set(values[:length]))==1 else 4 for values,length in zip(labels,lengths.tolist())]

def run(study,output):
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=600:
            raise TimeoutError('shared operator budget exceeded')
    torch.set_num_threads(2)
    study,output=Path(study),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((study/'report.json').read_text(encoding='utf-8'))
    original=json.loads((root/'antlab/runs/binding-pilot-summary-20261008.json').read_text(encoding='utf-8'))
    historical_path=root/'antlab/runs/learned-composition-query-report-20261008.json'
    historical=json.loads(historical_path.read_text(encoding='utf-8'))
    if (binding.get('format')!='avl-binding-pilot-v1' or not binding.get('completed')
            or not binding.get('pilot_passed') or binding['dataset_sha256']!=original['dataset_sha256']):
        raise ValueError('invalid primitive reproduction')
    for name,sha in original['source_hashes'].items():
        if binding['source_hashes'][name]!=sha or digest(root/name)!=sha:
            raise ValueError('primitive source changed')
    vocabulary=binding['vocabulary']
    checkpoints={seed:binding['seeds'][str(seed)]['checkpoint_sha256'] for seed in (44,55,66)}
    primitives={seed:load_frozen(study,seed,vocabulary,sha) for seed,sha in checkpoints.items()}
    frozen={seed:fingerprint(m) for seed,m in primitives.items()}
    data=make_data()
    signatures={k:{semantic_signature(rows[q['row']],q['pair']) for q in old_expand(rows)}
                for k,rows in data.items() if len(rows[0]['names'])==3}
    if signatures['train'] & (signatures['joint']|signatures['new_joint']):
        raise ValueError('three-node semantic split leakage')
    if any(len(r['names'])!=3 for r in data['train']):
        raise ValueError('long-chain training leak')
    paths=('antlab/semantic_shared_operator_data.py','antlab/semantic_shared_operator_pilot.py',
           'antlab/tests/test_semantic_shared_operator_data.py','antlab/tests/test_semantic_shared_operator_pilot.py',
           'antlab/semantic_learned_query_data.py','antlab/semantic_learned_query_pilot.py',
           'antlab/semantic_composition_pilot.py','antlab/semantic_alignment_pilot.py',
           'docs/AVL_SHARED_OPERATOR_PROTOCOL.md','.github/workflows/semantic-shared-operator.yml')
    report=dict(format='avl-shared-relation-operator-v1',completed=False,pilot_passed=False,
        scope='designed routing/inverse channels plus frozen supervised decoder and learned shared channel operator',
        new_parameters=66,frozen_receiver_parameters=676,active_receiver_parameters=742,
        neural_rule_discovery_claim=False,new_primitive_concepts=False,torch=torch.__version__,
        dataset_sha256=canonical_hash(data),source_hashes={p:digest(root/p) for p in paths},
        binding_source_hashes=original['source_hashes'],binding_report_sha256=digest(study/'report.json'),
        checkpoint_hashes=checkpoints,frozen_state_sha256=frozen,
        reproduced_checkpoint_byte_match={s:checkpoints[s]==original['seeds'][str(s)]['checkpoint_sha256'] for s in checkpoints},
        historical_flat_report_sha256=digest(historical_path),
        historical_reference_only={s:historical['seeds'][str(s)]['evaluations']['joint']['neural']['all'] for s in checkpoints},
        semantic_signature_counts={k:dict(count=len(v),train_overlap=len(v & signatures['train'])) for k,v in signatures.items()},
        paired_samples={k:dict(scenes=len(select_pairs(rows)),sha256=canonical_hash(select_pairs(rows)))
                        for k,rows in data.items()},partitions={},seeds={})
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for seed,primitive in primitives.items():
        deadline()
        train_packets,train_queries=query_packets(primitive,data['train'],vocabulary)
        p_train,length_train=prepare(primitive,train_packets)
        zero_train,_=prepare(primitive,train_packets,zero=True)
        targets=torch.tensor([q['target'] for q in train_queries],dtype=torch.long)
        model,baseline,training=train(p_train,length_train,zero_train,targets,seed+200,deadline)
        filename=output/('operator-seed-'+str(seed)+'.pt')
        torch.save(dict(format=report['format'],sender_seed=seed,operator_seed=seed+200,
                        operator_state=model.state_dict(),no_vector_state=baseline.state_dict()),filename)
        saved=torch.load(filename,map_location='cpu',weights_only=True)
        restored=SharedOperator()
        restored.load_state_dict(saved['operator_state'],strict=True)
        if fingerprint(restored)!=fingerprint(model):
            raise ValueError('operator roundtrip mismatch')
        entry=dict(training=training,checkpoint_sha256=digest(filename),checkpoint_bytes=filename.stat().st_size,
                   evaluations={},gates={})
        other=primitives[55 if seed==44 else 66 if seed==55 else 44]
        for split,rows in data.items():
            deadline()
            packets,queries=(train_packets,train_queries) if split=='train' else query_packets(primitive,rows,vocabulary)
            p,lengths=(p_train,length_train) if split=='train' else prepare(primitive,packets)
            zero,_=(zero_train,length_train) if split=='train' else prepare(primitive,packets,zero=True)
            shuffled,_=prepare(primitive,packets,shuffled=True)
            wrong,_=prepare(other,packets)
            predictions=predict(model,p,lengths)
            evaluation=dict(neural=metrics(queries,predictions),
                rule=metrics(queries,rule(p,lengths)),no_vector=metrics(queries,predict(baseline,zero,lengths)),
                zero=metrics(queries,predict(model,zero,lengths)),shuffled=metrics(queries,predict(model,shuffled,lengths)),
                wrong_checkpoint=metrics(queries,predict(model,wrong,lengths)),
                query_reversal=reversal(queries,predictions))
            if seed==44:
                n=len(rows[0]['names'])
                copies=n*(n-1)
                transmitted=sum(map(len,packets))
                report['partitions'][split]=dict(scenes=len(rows),queries=len(queries),nodes=n,
                    support=evaluation['neural']['all']['support'],
                    positive_distance3_support=evaluation['neural']['positive_distance3']['support'],
                    transmitted_bytes=transmitted,
                    symbolic_gold_bytes=sum((6+table_size(r['table'])+2*(n-1)+(n-1)+10)*copies for r in rows),
                    text_reference_bytes=sum((len(r['text'])+table_size(r['table'])+10)*copies for r in rows),
                    graph_repeated_per_query=True,measured_query_bytes=sorted(set(map(len,packets))),
                    cost_scope='actual graph/name/topology/vector/query headers included; model identity/distribution and network framing excluded')
                if transmitted!=(181 if n==3 else 254)*len(queries):
                    raise ValueError('query accounting mismatch')
            subset=select_pairs(rows)
            sample_packets,sample_queries=query_packets(primitive,subset,vocabulary)
            sample_p,sample_lengths=prepare(primitive,sample_packets)
            sample_predictions=predict(model,sample_p,sample_lengths)
            for kind in ('equivalent','table','clause_order','replacement'):
                changed_rows=remade(subset,kind)
                changed_packets,changed_queries=query_packets(primitive,changed_rows,vocabulary)
                changed_p,changed_lengths=prepare(primitive,changed_packets)
                changed=predict(model,changed_p,changed_lengths)
                evaluation[kind]=dict(scenes=len(subset),queries=len(sample_queries),
                    metrics=metrics(changed_queries,changed),
                    both_correct=sum(a==q['target'] and b==cq['target'] for a,q,b,cq in
                                     zip(sample_predictions,sample_queries,changed,changed_queries))/len(sample_queries))
            entry['evaluations'][split]=evaluation
            if split!='train':
                neural=evaluation['neural']
                macro=neural['all']['macro_recall_observed']
                no_vector=evaluation['no_vector']['all']['macro_recall_observed']
                gates=dict(accuracy=neural['all']['accuracy']>=.95,macro=macro>=.95,
                    direct=neural['direct']['accuracy']>=.95,
                    positive_path=neural['positive_path']['accuracy']>=.95,
                    undetermined=neural['undetermined']['accuracy']>=.95,
                    invariant=all(evaluation[k]['both_correct']>=.95 for k in ('equivalent','table','clause_order')),
                    replacement=evaluation['replacement']['metrics']['all']['accuracy']>=.95,
                    query_reversal=evaluation['query_reversal']['both_correct']>=.95,
                    no_vector=no_vector<=.50,gap=macro-no_vector>=.40)
                if len(rows[0]['names'])==4:
                    gates['positive_distance3']=neural['positive_distance3']['accuracy']>=.95
                entry['gates'][split]=gates
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
