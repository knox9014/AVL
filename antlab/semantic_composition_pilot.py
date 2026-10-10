"""Experimental explicit graph envelope for two frozen learned AVL relations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import time
import torch
from .semantic_binding_data import canonicalize
from .semantic_binding_pilot import digest, canonical_hash, encode_texts
from .semantic_alignment_pilot import load_frozen
from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_composition_data import (validate_names, validate_edges, table_size,
                                       edge_rows, infer_chain, make_row, make_data)

HEADER=struct.Struct('<4sBB')

def pack_message(table,edges,vectors):
    validate_names(table)
    validate_edges(edges)
    if vectors.shape!=(2,16):
        raise ValueError('expected exactly two relation vectors')
    names=b''.join(struct.pack('<H',len(n))+n.encode('ascii') for n in table)
    topology=bytes(i for edge in edges for i in edge)
    return HEADER.pack(b'AVC1',3,2)+names+topology+pack_vectors(vectors)

def unpack_message(packet):
    if not isinstance(packet,bytes) or not 156<=len(packet)<=348:
        raise ValueError('invalid composition packet length')
    if HEADER.unpack_from(packet)!=(b'AVC1',3,2):
        raise ValueError('unsupported composition header')
    position=HEADER.size
    table=[]
    for _ in range(3):
        if position+2>len(packet):
            raise ValueError('truncated name length')
        length=struct.unpack_from('<H',packet,position)[0]
        position+=2
        if not 1<=length<=64 or position+length>len(packet):
            raise ValueError('invalid name length')
        try:
            table.append(packet[position:position+length].decode('ascii'))
        except UnicodeDecodeError as error:
            raise ValueError('non-ASCII name') from error
        position+=length
    validate_names(table)
    if position+4>len(packet):
        raise ValueError('truncated topology')
    edges=[list(packet[position:position+2]),list(packet[position+2:position+4])]
    validate_edges(edges)
    vectors=unpack_vectors(packet[position+4:])
    if vectors.shape!=(2,16):
        raise ValueError('invalid relation count')
    return dict(table=table,edges=edges,vectors=vectors)

def physical_meaning(decoded,names):
    validate_names(names)
    if set(names)!=set(decoded['table']) or len(decoded['labels'])!=2:
        raise ValueError('query names do not match graph')
    validate_edges(decoded['edges'])
    lookup={}
    for edge,label in zip(decoded['edges'],decoded['labels']):
        if type(label) is not int or label not in range(4):
            raise ValueError('invalid decoded relation')
        a,b=[decoded['table'][i] for i in edge]
        lookup[a,b]=label
        lookup[b,a]=label^1
    try:
        return lookup[names[0],names[1]],lookup[names[1],names[2]]
    except KeyError as error:
        raise ValueError('requested chain is absent') from error

def classification(targets,predictions,classes):
    if not len(targets) or len(targets)!=len(predictions):
        raise ValueError('metric length mismatch')
    if any(type(x) is not int or x not in range(classes) for x in [*targets,*predictions]):
        raise ValueError('invalid metric class')
    confusion=[[0]*classes for _ in range(classes)]
    for target,prediction in zip(targets,predictions):
        confusion[target][prediction]+=1
    support=[sum(row) for row in confusion]
    observed=[i for i,n in enumerate(support) if n]
    recall=[confusion[i][i]/n if n else None for i,n in enumerate(support)]
    return dict(accuracy=sum(confusion[i][i] for i in range(classes))/len(targets),
                macro_recall_observed=sum(recall[i] for i in observed)/len(observed),
                observed_classes=observed,support=support,recall=recall,confusion=confusion)

@torch.no_grad()
def encode_batch(model,rows,vocabulary):
    plans=[edge_rows(row['text'],row['table']) for row in rows]
    unique={}
    for plan in plans:
        for edge in plan:
            unique.setdefault(tuple(canonicalize(edge['text'],edge['table'])),edge)
    keys=list(unique)
    tokens,lengths=encode_texts(list(unique.values()),vocabulary)
    vectors=model.encode(tokens,lengths)
    cache=dict(zip(keys,vectors))
    return [pack_message(row['table'],[e['slots'] for e in plan],
            torch.stack([cache[tuple(canonicalize(e['text'],e['table']))] for e in plan]))
            for row,plan in zip(rows,plans)]

@torch.no_grad()
def receive_batch(model,packets):
    decoded=[unpack_message(packet) for packet in packets]
    vectors=torch.cat([d['vectors'] for d in decoded])
    labels=model.receive(vectors).argmax(-1).tolist()
    for i,d in enumerate(decoded):
        d['labels']=labels[2*i:2*i+2]
    return decoded

def measure(rows,decoded,expected=None):
    expected=[tuple(r['relations']) for r in rows] if expected is None else expected
    actual=[physical_meaning(d,r['names']) for r,d in zip(rows,decoded)]
    target_codes=[4*a+b for a,b in expected]
    actual_codes=[4*a+b for a,b in actual]
    return dict(exact=classification(target_codes,actual_codes,16),
                edge=classification([v for pair in expected for v in pair],
                                    [v for pair in actual for v in pair],4),
                derived_query=classification([infer_chain(*p) for p in expected],
                                             [infer_chain(*p) for p in actual],5))

def canonical_targets(row,decoded):
    lookup={}
    for i,relation in enumerate(row['relations']):
        a,b=row['names'][i:i+2]
        lookup[a,b]=relation
        lookup[b,a]=relation^1
    return [lookup[tuple(decoded['table'][i] for i in edge)] for edge in decoded['edges']]

def remade(rows,kind):
    result=[]
    for row in rows:
        relations=list(row['relations'])
        orientation,order,reverse=row['orientation'],row['order'],row['reverse']
        if kind=='equivalent':
            orientation^=3
        elif kind=='table':
            order=(order+1)%6
        elif kind=='clause_order':
            reverse^=1
        elif kind=='replacement':
            relations[0]^=1
        else:
            raise ValueError('unknown fixture intervention')
        result.append(make_row(row['names'],relations,orientation,order,row['template'],reverse))
    return result

def fingerprint(model):
    value=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value.update(name.encode('ascii'))
        value.update(str((str(tensor.dtype),list(tensor.shape))).encode('ascii'))
        value.update(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
    return value.hexdigest()

@torch.no_grad()
def run(study,output):
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=600:
            raise TimeoutError('composition audit budget exceeded')
    torch.set_num_threads(2)
    study,output=Path(study),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((study/'report.json').read_text(encoding='utf-8'))
    original=json.loads((root/'antlab/runs/binding-pilot-summary-20261008.json').read_text(encoding='utf-8'))
    if (binding.get('format')!='avl-binding-pilot-v1' or not binding.get('completed')
            or not binding.get('pilot_passed') or binding['dataset_sha256']!=original['dataset_sha256']):
        raise ValueError('invalid binding reproduction')
    for name,sha in original['source_hashes'].items():
        if binding['source_hashes'][name]!=sha or digest(root/name)!=sha:
            raise ValueError('primitive source configuration changed')
    vocabulary=binding['vocabulary']
    checkpoints={seed:binding['seeds'][str(seed)]['checkpoint_sha256'] for seed in (44,55,66)}
    models={seed:load_frozen(study,seed,vocabulary,sha) for seed,sha in checkpoints.items()}
    frozen={seed:fingerprint(model) for seed,model in models.items()}
    data=make_data()
    paths=('antlab/semantic_composition_data.py','antlab/semantic_composition_pilot.py',
           'antlab/tests/test_semantic_composition_data.py','antlab/tests/test_semantic_composition_pilot.py',
           'antlab/semantic_alignment_pilot.py','docs/AVL_COMPOSITION_PILOT_PROTOCOL.md',
           '.github/workflows/semantic-composition-pilot.yml')
    report=dict(format='avl-three-entity-composition-v1',completed=False,pilot_passed=False,
                scope='designed graph transport of two pretrained relation vectors; no new neural training',
                topology_and_segmentation_designed=True,learned_composition_claim=False,
                derived_query_rule='same directional relation is transitive; otherwise undetermined',
                torch=torch.__version__,dataset_sha256=canonical_hash(data),
                binding_report_sha256=digest(study/'report.json'),
                binding_source_hashes=original['source_hashes'],
                source_hashes={p:digest(root/p) for p in paths},
                checkpoint_hashes=checkpoints,frozen_state_sha256=frozen,
                reproduced_checkpoint_byte_match={seed:checkpoints[seed]==original['seeds'][str(seed)]['checkpoint_sha256']
                                                  for seed in checkpoints},
                model_parameters={seed:sum(p.numel() for p in m.parameters()) for seed,m in models.items()},
                additional_neural_parameters=0,partitions={},seeds={})
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    modal=Counter(row['target_tuple'] for row in data['development']).most_common(1)[0][0]
    for seed,model in models.items():
        entry=dict(evaluations={},gates={})
        other=models[(55 if seed==44 else 66 if seed==55 else 44)]
        for split,rows in data.items():
            deadline()
            packets=encode_batch(model,rows,vocabulary)
            decoded=receive_batch(model,packets)
            evaluation=dict(transmitted=measure(rows,decoded))
            if seed==44:
                table_total=sum(table_size(row['table']) for row in rows)
                total=sum(map(len,packets))
                symbolic_total=sum(HEADER.size+table_size(row['table'])+4+2 for row in rows)
                report['partitions'][split]=dict(records=len(rows),
                    tuple_support=[sum(row['target_tuple']==i for row in rows) for i in range(16)],
                    graph_header_bytes=6*len(rows),topology_bytes=4*len(rows),
                    vector_packet_bytes=140*len(rows),table_bytes=table_total,
                    transmitted_bytes=total,structured_oracle_bytes=symbolic_total,
                    text_reference_bytes=sum(len(row['text'].encode('ascii')) for row in rows)+table_total,
                    cost_scope='includes envelope/name table/topology/AVL1 headers; excludes model identity/distribution/network framing',
                    symbolic_oracle_exact_accuracy=1.0)
                if total!=150*len(rows)+table_total:
                    raise ValueError('actual packet accounting mismatch')
            for kind in ('zero','shuffled','swapped'):
                variants=[]
                for i,d in enumerate(decoded):
                    v=(torch.zeros_like(d['vectors']) if kind=='zero' else
                       decoded[(i-1)%len(decoded)]['vectors'] if kind=='shuffled' else d['vectors'].flip(0))
                    variants.append(pack_message(d['table'],d['edges'],v))
                changed=receive_batch(model,variants)
                if kind!='swapped':
                    evaluation[kind]=measure(rows,changed)
                else:
                    indices=[i for i,(row,d) in enumerate(zip(rows,decoded))
                             if len(set(canonical_targets(row,d)))==2]
                    if not indices:
                        raise ValueError('no eligible swap pairs')
                    expected=[]
                    for i in indices:
                        labels=canonical_targets(rows[i],decoded[i])[::-1]
                        expected.append(physical_meaning({**decoded[i],'labels':labels},rows[i]['names']))
                    changed_subset=[changed[i] for i in indices]
                    row_subset=[rows[i] for i in indices]
                    evaluation['vector_swap']=dict(eligible_records=len(indices),
                         changed=measure(row_subset,changed_subset,expected),
                         original_agreement=measure(row_subset,changed_subset)['exact']['accuracy'])
            evaluation['wrong_checkpoint']=measure(rows,receive_batch(other,packets))
            targets=[row['target_tuple'] for row in rows]
            evaluation['modal_no_message']=classification(targets,[modal]*len(rows),16)
            for kind in ('equivalent','table','clause_order','replacement'):
                changed_rows=remade(rows,kind)
                changed=receive_batch(model,encode_batch(model,changed_rows,vocabulary))
                metrics=measure(changed_rows,changed)
                evaluation[kind]=dict(metrics=metrics,
                    both_correct=sum(physical_meaning(d,r['names'])==tuple(r['relations']) and
                                     physical_meaning(c,r['names'])==tuple(cr['relations'])
                                     for r,d,cr,c in zip(rows,decoded,changed_rows,changed))/len(rows))
            entry['evaluations'][split]=evaluation
            if split!='development':
                exact=evaluation['transmitted']['exact']
                entry['gates'][split]=dict(accuracy=exact['accuracy']>=.95,
                    observed_tuple_macro=exact['macro_recall_observed']>=.95,
                    invariant=all(evaluation[k]['both_correct']>=.95 for k in ('equivalent','table','clause_order')),
                    replacement=evaluation['replacement']['metrics']['exact']['accuracy']>=.95,
                    zero=evaluation['zero']['exact']['accuracy']<=.30,
                    gap=exact['accuracy']-evaluation['zero']['exact']['accuracy']>=.60,
                    swap=evaluation['vector_swap']['changed']['exact']['accuracy']>=.95,
                    swap_original=evaluation['vector_swap']['original_agreement']<=.05)
            deadline()
        report['seeds'][str(seed)]=entry
    for seed,model in models.items():
        if fingerprint(model)!=frozen[seed] or digest(study/('seed-'+str(seed)+'.pt'))!=checkpoints[seed]:
            raise ValueError('frozen model changed')
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
