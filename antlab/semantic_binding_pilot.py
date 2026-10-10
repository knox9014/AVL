"""Preregistered, separately supervised two-subject AVL binding pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import torch
from torch import nn
from .semantic_codec import pack_vectors, unpack_vectors
from .semantic_binding_data import make_data, canonicalize, counterpart, table_bytes

TESTS = ('pair','joint','new_names','phrasing')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('ascii')).hexdigest()

class BindingNet(nn.Module):
    def __init__(self,vocabulary_size):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary_size+1,16,padding_idx=0)
        self.gru=nn.GRU(16,48,batch_first=True)
        self.projection=nn.Linear(48,16)
        self.receiver=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,4))

    def encode(self,tokens,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(tokens),lengths.cpu(),
                                                 batch_first=True,enforce_sorted=False)
        _,hidden=self.gru(packed)
        return torch.tanh(self.projection(hidden[-1]))

    def receive(self,vectors):
        if vectors.ndim!=2 or vectors.shape[1]!=16 or vectors.dtype!=torch.float32 or not torch.isfinite(vectors).all():
            raise ValueError('expected finite float32 width16')
        return self.receiver(vectors)

def encode_texts(rows,vocabulary):
    mapping={word:i+1 for i,word in enumerate(vocabulary)}
    sequences=[]
    for row in rows:
        words=canonicalize(row['text'],row['table'])
        if any(word not in mapping for word in words):
            raise ValueError('word absent from training vocabulary')
        sequences.append([mapping[word] for word in words])
    if not sequences:
        raise ValueError('empty source list')
    lengths=torch.tensor([len(seq) for seq in sequences],dtype=torch.long)
    tokens=torch.zeros((len(rows),int(lengths.max())),dtype=torch.long)
    for i,seq in enumerate(sequences):
        tokens[i,:len(seq)]=torch.tensor(seq)
    return tokens,lengths

def wire_vectors(vectors):
    if vectors.ndim!=2 or vectors.shape[1]!=16 or not len(vectors):
        raise ValueError('expected nonempty width16 vectors')
    return torch.cat([unpack_vectors(pack_vectors(v.unsqueeze(0))) for v in vectors])

def score(targets,predictions):
    if targets.shape!=predictions.shape or targets.ndim!=1 or not len(targets):
        raise ValueError('inconsistent metric inputs')
    if any(not 0<=int(v)<4 for v in torch.cat((targets,predictions))):
        raise ValueError('invalid class')
    support=[int((targets==c).sum()) for c in range(4)]
    if not all(support):
        raise ValueError('four-class evaluation requires all classes')
    confusion=[[int(((targets==c)&(predictions==p)).sum()) for p in range(4)] for c in range(4)]
    recall=[confusion[c][c]/support[c] for c in range(4)]
    return dict(accuracy=float((targets==predictions).double().mean()),
                macro_recall=sum(recall)/4,support=support,recall=recall,confusion=confusion)

def paired_score(targets,predictions,other_targets,other_predictions):
    if not targets.shape==predictions.shape==other_targets.shape==other_predictions.shape or not len(targets):
        raise ValueError('inconsistent paired inputs')
    return float(((targets==predictions)&(other_targets==other_predictions)).double().mean())

@torch.no_grad()
def infer(model,rows,vocabulary,receiver=None):
    tokens,lengths=encode_texts(rows,vocabulary)
    vectors=torch.cat([model.encode(tokens[i:i+256],lengths[i:i+256]) for i in range(0,len(rows),256)])
    received=wire_vectors(vectors)
    head=model if receiver is None else receiver
    return head.receive(received).argmax(-1),received

def targets_for(rows):
    return torch.tensor([row['target'] for row in rows],dtype=torch.long)

def train(rows,vocabulary,seed,deadline):
    torch.manual_seed(seed)
    model=BindingNet(len(vocabulary))
    tokens,lengths=encode_texts(rows,vocabulary)
    targets=targets_for(rows)
    torch.manual_seed(seed+1000)
    baseline=BindingNet(len(vocabulary)).receiver
    optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    base_optim=torch.optim.AdamW(baseline.parameters(),lr=.003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    started=time.monotonic()
    for step in range(2000):
        deadline()
        indices=torch.randint(len(rows),(128,),generator=generator)
        optim.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model.receive(model.encode(tokens[indices],lengths[indices])),targets[indices])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite sender training loss')
        loss.backward()
        if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(),1.)):
            raise ValueError('nonfinite sender gradients')
        optim.step()
        base_optim.zero_grad(set_to_none=True)
        base_loss=nn.functional.cross_entropy(baseline(torch.zeros((128,16))),targets[indices])
        if not torch.isfinite(base_loss):
            raise ValueError('nonfinite baseline loss')
        base_loss.backward()
        nn.utils.clip_grad_norm_(baseline.parameters(),1.)
        base_optim.step()
    model.eval()
    baseline.eval()
    return model,baseline,dict(steps=2000,batch=128,seconds=time.monotonic()-started,
                               final_loss=float(loss.detach()),baseline_loss=float(base_loss.detach()),
                               parameters=sum(p.numel() for p in model.parameters()),
                               receiver_parameters=sum(p.numel() for p in model.receiver.parameters()),
                               baseline_parameters=sum(p.numel() for p in baseline.parameters()))

def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=600:
            raise TimeoutError('declared600second binding budget exceeded')
    torch.set_num_threads(2)
    data=make_data()
    vocabulary=sorted({word for row in data['train'] for word in canonicalize(row['text'],row['table'])})
    training_forms={tuple(canonicalize(row['text'],row['table'])) for row in data['train']}
    root=Path(__file__).resolve().parents[1]
    paths=('antlab/semantic_binding_data.py','antlab/semantic_binding_pilot.py',
           'antlab/semantic_codec.py','docs/AVL_BINDING_PILOT_PROTOCOL.md',
           'antlab/tests/test_semantic_binding_data.py','antlab/tests/test_semantic_binding_pilot.py')
    report=dict(format='avl-binding-pilot-v1',completed=False,pilot_passed=False,
                supervised_end_to_end=True,arbitrary_name_comprehension_learned=False,
                scope='finite spatial relation binding with designed name-to-slot adapter',
                torch=torch.__version__,dataset_sha256=canonical_hash(data),
                source_hashes={path:digest(root/path) for path in paths},
                vocabulary=vocabulary,canonical_training_forms=len(training_forms),
                partitions={},seeds={},crossplay={})
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for split,rows in data.items():
        targets=targets_for(rows)
        support=[int((targets==c).sum()) for c in range(4)]
        if len(set(support))!=1 or not all(support):
            raise ValueError('evaluation imbalance')
        # Build actual literal-table bytes; table holds no relation labels.
        literal_packets=[b''.join(struct.pack('<H',len(name))+name.encode('ascii') for name in row['table']) for row in rows]
        if any(len(packet)!=table_bytes(row['table']) for row,packet in zip(rows,literal_packets)):
            raise ValueError('literal table accounting mismatch')
        table_total=sum(map(len,literal_packets))
        report['partitions'][split]=dict(records=len(rows),support=support,
              canonical_training_overlap=sum(tuple(canonicalize(r['text'],r['table'])) in training_forms for r in rows),
              vector_packet_bytes=76*len(rows),table_bytes=table_total,
              transmitted_bytes=76*len(rows)+table_total,
              text_reference_bytes=sum(len(r['text'].encode('ascii')) for r in rows)+table_total,
              structured_oracle_bytes=len(rows)+table_total,
              cost_scope='per-message headers and literals included; network/model distribution excluded')
    models={}
    joint_vectors={}
    for seed in (44,55,66):
        deadline()
        model,baseline,training=train(data['train'],vocabulary,seed,deadline)
        models[seed]=model
        filename='seed-'+str(seed)+'.pt'
        torch.save(dict(format=report['format'],seed=seed,vocabulary=vocabulary,
                        model_state=model.state_dict(),baseline_state=baseline.state_dict()),output/filename)
        entry=dict(training=training,checkpoint_sha256=digest(output/filename),evaluations={},gates={})
        for split in ('train','validation',*TESTS):
            rows=data[split]
            targets=targets_for(rows)
            prediction,vectors=infer(model,rows,vocabulary)
            if split=='joint':
                joint_vectors[seed]=vectors
            with torch.no_grad():
                zero=model.receive(torch.zeros_like(vectors)).argmax(-1)
                shuffled=model.receive(vectors.roll(1,0)).argmax(-1)
                no_message=baseline(torch.zeros_like(vectors)).argmax(-1)
            evaluation=dict(transmitted=score(targets,prediction),zero=score(targets,zero),
                            shuffled=score(targets,shuffled),no_message=score(targets,no_message))
            pairs={}
            for kind in ('equivalent','opposite','table'):
                changed=[counterpart(row,kind) for row in rows]
                other_prediction,_=infer(model,changed,vocabulary)
                other_targets=targets_for(changed)
                pairs[kind]=dict(both_correct=paired_score(targets,prediction,other_targets,other_prediction),
                                 changed_scene_accuracy=score(other_targets,other_prediction)['accuracy'])
                if kind=='opposite':
                    evaluation['valid_replacement']=dict(original_agreement=float((other_prediction==targets).double().mean()),
                                                         replacement=score(other_targets,other_prediction))
            evaluation['paired']=pairs
            entry['evaluations'][split]=evaluation
            if split in TESTS:
                entry['gates'][split]=dict(
                    accuracy=evaluation['transmitted']['accuracy']>=.95,
                    macro_recall=evaluation['transmitted']['macro_recall']>=.95,
                    paired=all(v['both_correct']>=.95 for v in pairs.values()),
                    replacement=evaluation['valid_replacement']['replacement']['accuracy']>=.95,
                    no_message=evaluation['no_message']['accuracy']<=.30,
                    gap=evaluation['transmitted']['accuracy']-evaluation['no_message']['accuracy']>=.60)
            deadline()
        report['seeds'][str(seed)]=entry
    joint_targets=targets_for(data['joint'])
    with torch.no_grad():
        for sender in models:
            report['crossplay'][str(sender)]={str(receiver):score(joint_targets,models[receiver].receive(joint_vectors[sender]).argmax(-1))
                                              for receiver in models}
    report.update(completed=True,pilot_passed=all(all(all(g.values()) for g in entry['gates'].values())
                                                 for entry in report['seeds'].values()),
                  total_seconds=time.monotonic()-started)
    deadline()
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.output)
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
