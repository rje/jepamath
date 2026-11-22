import sys, os, glob, json, numpy as np, torch
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import webdataset as wds
from jepagen.shards import load_sample
from jepatrain.data import downsample
from jepatrain.reeval import load_run
from locate_surprise import MEAN, STD
from count_readout import oracle_tokens
from slot_probe import segments
torch.set_num_threads(40)
enc,_,_,step=load_run('/mnt/data/jepamath/trackb/cloud-v4b-128-cw50-100k','weights_100000.pt','cpu'); enc.eval()
g=enc.grid; N=g*g
rr,cc=np.divmod(np.arange(N),g)
adj=(np.abs(rr[:,None]-rr[None])<=1)&(np.abs(cc[:,None]-cc[None])<=1)&~np.eye(N,dtype=bool)
same_sum=np.zeros((len(enc.blocks),enc.blocks[0].heads)); diff_sum=np.zeros_like(same_sum); ns=0; nd=0
n=0
for s in wds.WebDataset(sorted(glob.glob('/mnt/data/jepamath/v4b_probe/train/possible/*.tar')),shardshuffle=False,empty_check=False):
    ep,frames,seg,state=load_sample(s); last=state['frames'][-1]
    in_box={o['id'] for o in last['objects'] if o['region']=='box'}
    sg=segments(oracle_tokens(seg[-1],g,in_box)[None])[0].numpy()
    fr=downsample(frames,128); x=((torch.from_numpy(fr[-4:])[None].permute(0,4,1,2,3).float()/255-MEAN)/STD)
    obj=np.nonzero(sg>0)[0]
    if len(obj)==0: continue
    same=adj[obj]&(sg[None,:]==sg[obj][:,None]); diff=adj[obj]&(sg[None,:]!=sg[obj][:,None])
    if same.sum()==0 or diff.sum()==0: continue
    with torch.no_grad():
        t=enc.embed(x).flatten(2).transpose(1,2)+enc.pos[:N]
        for li,b in enumerate(enc.blocks):
            B,_,D=t.shape
            q,k,_v=b.qkv(b.n1(t)).view(B,N,3,b.heads,D//b.heads).permute(2,0,3,1,4)
            a=torch.softmax(q@k.transpose(-1,-2)/(D//b.heads)**0.5,-1)[0].numpy()[:,obj]   # [h, nobj, N]
            same_sum[li]+= (a*same[None]).sum((1,2)); diff_sum[li]+=(a*diff[None]).sum((1,2))
            t=b(t)
    ns+=same.sum(); nd+=diff.sum(); n+=1
    if n==400: break
r=(same_sum/ns)/(diff_sum/nd)
print("clips",n,"; per-layer ratio of attention to same-object vs other adjacent tokens (per pair): max / median over heads")
for li in range(len(r)): print(f"  layer {li+1}: max {r[li].max():.2f}  median {np.median(r[li]):.2f}")
