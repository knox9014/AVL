"""Controlled supervised vector grounding, learning curves and cross-play."""
import argparse
import json
from pathlib import Path
import struct
import time
import torch
from torch import nn
from .semantic_binding_data import make_data,canonicalize,table_bytes
from .semantic_binding_pilot import encode_texts,targets_for,score,digest,canonical_hash
from .semantic_grounded_codec import anchors,pack,unpack

ARMS={'free16':(16,0),'grounded16':(16,1),'grounded2':(2,1)}
SEEDS=(44,55,66)
FINAL=('pair','joint','new_names','phrasing')


class VectorNet(nn.Module):
    def __init__(self,vocabulary_size,width,contract):
        super().__init__()
        anchors(width)
        if contract not in (0,1) or (contract==0 and width!=16):
            raise ValueError('unsupported contract')
        self.width=width; self.contract=contract
        self.embedding=nn.Embedding(vocabulary_size+1,16,padding_idx=0)
        self.gru=nn.GRU(16,48,batch_first=True)
        self.projection=nn.Linear(48,width)
        self.receiver=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,4))

    def encode(self,tokens,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(tokens),lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.gru(packed)
        return torch.tanh(self.projection(hidden[-1]))

    def receive(self,vectors):
        if vectors.ndim!=2 or vectors.shape[1]!=self.width or vectors.dtype!=torch.float32 or not torch.isfinite(vectors).all():
            raise ValueError('expected finite float32 contract vectors')
        return self.receiver(vectors)


def objective(model,vectors,targets):
    loss=nn.functional.cross_entropy(model.receive(vectors),targets)
    if model.contract:
        points=torch.tensor(anchors(model.width),dtype=vectors.dtype,device=vectors.device)
        loss=loss+(vectors-points[targets]).square().sum(-1).mean()
    return loss


def wire(vectors,contract):
    if vectors.ndim!=2 or vectors.dtype!=torch.float32 or not len(vectors):
        raise ValueError('expected nonempty float32 vector matrix')
    # Actual independent one-message framing, not a batch-size cost shortcut.
    result=[unpack(pack([row],contract),contract,vectors.shape[1])[0] for row in vectors.detach().cpu().tolist()]
    return torch.tensor(result,dtype=torch.float32)


@torch.no_grad()
def encode(model,rows,vocabulary):
    tokens,lengths=encode_texts(rows,vocabulary)
    return torch.cat([model.encode(tokens[i:i+256],lengths[i:i+256]) for i in range(0,len(rows),256)])


def train(rows,validation,vocabulary,seed,width,contract,deadline):
    torch.manual_seed(seed)
    model=VectorNet(len(vocabulary),width,contract)
    tokens,lengths=encode_texts(rows,vocabulary); targets=targets_for(rows)
    optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    started=time.monotonic(); curve=[]
    for step in range(1,2001):
        deadline()
        indices=torch.randint(len(rows),(128,),generator=generator)
        optim.zero_grad(set_to_none=True)
        vectors=model.encode(tokens[indices],lengths[indices])
        loss=objective(model,vectors,targets[indices])
        if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
        loss.backward()
        if not torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(),1.)):
            raise ValueError('nonfinite training gradients')
        optim.step()
        if step in (250,500,1000,2000):
            model.eval()
            with torch.no_grad():
                z=wire(encode(model,validation,vocabulary),contract)
                metric=score(targets_for(validation),model.receive(z).argmax(-1))
            curve.append(dict(step=step,examples_seen=step*128,validation=metric,seconds=time.monotonic()-started))
            model.train()
    model.eval()
    return model,dict(steps=2000,batch=128,seconds=time.monotonic()-started,
        final_loss=float(loss.detach()),parameters=sum(p.numel() for p in model.parameters()),
        receiver_parameters=sum(p.numel() for p in model.receiver.parameters()),curve=curve,
        first_sampled_validation_95=next((p['step'] for p in curve if p['validation']['macro_recall']>=.95),None))


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>=900: raise TimeoutError('declared900second budget exceeded')
    torch.set_num_threads(2)
    data=make_data()
    vocabulary=sorted({word for row in data['train'] for word in canonicalize(row['text'],row['table'])})
    paths=['antlab/semantic_grounded_pilot.py','antlab/semantic_grounded_codec.py',
           'antlab/semantic_binding_data.py','antlab/semantic_binding_pilot.py',
           'antlab/tests/test_semantic_grounded_pilot.py','antlab/tests/test_semantic_grounded_codec.py',
           'docs/AVL_GROUNDED_VECTOR_PROTOCOL.md']
    root=Path(__file__).resolve().parents[1]
    report=dict(format='avl-grounded-vector-v1',completed=False,pilot_passed=False,
                scope='supervised four-relation shared anchors; not general language optimality',
                torch=torch.__version__,threads=2,dataset_sha256=canonical_hash(data),
                source_hashes={path:digest(root/path) for path in paths},vocabulary=vocabulary,
                canonical_training_forms=len({tuple(canonicalize(r['text'],r['table'])) for r in data['train']}),
                arms={},partitions={})
    (output/'data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for split,rows in data.items():
        literals=[b''.join(struct.pack('<H',len(name))+name.encode('ascii') for name in r['table']) for r in rows]
        if any(len(b)!=table_bytes(r['table']) for r,b in zip(rows,literals)):
            raise ValueError('literal accounting mismatch')
        table_total=sum(map(len,literals))
        report['partitions'][split]=dict(records=len(rows),table_bytes=table_total,
            text_reference_bytes=sum(len(r['text'].encode('ascii')) for r in rows)+table_total,
            symbolic_gold_bytes=len(rows)+table_total,
            transmitted_bytes={arm:sum(len(pack([[0.]*width],contract)) for _ in rows)+table_total for arm,(width,contract) in ARMS.items()})
    for arm,(width,contract) in ARMS.items():
        models={}; cached={}; entry=dict(width=width,contract=contract,seeds={},crossplay={},gates={})
        report['arms'][arm]=entry
        for seed in SEEDS:
            model,training=train(data['train'],data['validation'],vocabulary,seed,width,contract,deadline)
            models[seed]=model
            filename=arm+'-'+str(seed)+'.pt'
            torch.save(dict(format=report['format'],arm=arm,seed=seed,vocabulary=vocabulary,width=width,contract=contract,model_state=model.state_dict()),output/filename)
            record=dict(training=training,checkpoint_sha256=digest(output/filename),evaluations={})
            entry['seeds'][str(seed)]=record
            for split in ('train','validation',*FINAL):
                deadline()
                raw=encode(model,data[split],vocabulary)
                z=wire(raw,contract); targets=targets_for(data[split])
                cached[seed,split]=z
                with torch.no_grad():
                    generator=torch.Generator().manual_seed(9000+seed)
                    noise=(torch.rand(raw.shape,generator=generator)*2-1)*.05
                    noisy=wire(raw+noise,contract)
                    metric=dict(transmitted=score(targets,model.receive(z).argmax(-1)),
                        zero=score(targets,model.receive(torch.zeros_like(z)).argmax(-1)),
                        shuffled=score(targets,model.receive(z.roll(1,0)).argmax(-1)),
                        bounded_noise=score(targets,model.receive(noisy).argmax(-1)))
                    if contract:
                        metric['anchor_squared_error_sum_mean']=float((z-torch.tensor(anchors(width))[targets]).square().sum(-1).mean())
                record['evaluations'][split]=metric
            record['examples']=[dict(text=r['text'],table=r['table'],target=r['target'],vector=z.tolist())
                                for r,z in zip(data['joint'][:8],cached[seed,'joint'][:8])]
        with torch.no_grad():
            for split in FINAL:
                targets=targets_for(data[split])
                entry['crossplay'][split]={str(sender):{str(receiver):score(targets,models[receiver].receive(cached[sender,split]).argmax(-1)) for receiver in SEEDS} for sender in SEEDS}
        entry['gates']=dict(matching=all(r['evaluations'][split]['transmitted']['macro_recall']>=.95 for r in entry['seeds'].values() for split in FINAL),
            crossplay=all(entry['crossplay'][split][str(s)][str(r)]['macro_recall']>=.95 for split in FINAL for s in SEEDS for r in SEEDS if s!=r))
        values=[entry['crossplay'][split][str(s)][str(r)]['macro_recall'] for split in FINAL for s in SEEDS for r in SEEDS if s!=r]
        entry['off_diagonal_macro_mean']=sum(values)/len(values)
        entry['off_diagonal_macro_min']=min(values)
        # Persist completed arms even if a later arm hits the declared budget.
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report.update(completed=True,pilot_passed=all(all(report['arms'][a]['gates'].values()) for a in ('grounded16','grounded2')),total_seconds=time.monotonic()-started)
    deadline()
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2,allow_nan=False))

