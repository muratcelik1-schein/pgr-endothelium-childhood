# -*- coding: utf-8 -*-
"""Per gene age association and the three criteria across the endothelial series, written
so that the genome wide panel is drawn from a shipped table. The definitions copy
make_figures.py and verify_all.py exactly: log1p, standardised within study, Spearman with
age on the source axis; fold as (last band + 1e-4)/(first band + 1e-4); monotone with a
1e-9 tolerance; zero fraction on donors below 2 on the source axis. The production gate is
asserted before anything is written."""
import argparse, numpy as np
from scipy.stats import rankdata, spearmanr
ap=argparse.ArgumentParser()
ap.add_argument("--pb",required=True); ap.add_argument("--groups",required=True)
ap.add_argument("--genes",required=True); ap.add_argument("--out",required=True)
A=ap.parse_args()
B=[(0.75,2),(2,6),(6,12),(12,20),(20,99)]
z=np.load(A.pb); sums=z["sums"].astype(np.float64); nc=z["n_cells"]
genes=np.array([l.rstrip("\n") for l in open(A.genes)])
meta=[]
with open(A.groups) as fh:
    h=fh.readline().rstrip("\n").split("\t")
    for l in fh: meta.append(dict(zip(h,l.rstrip("\n").split("\t"))))
ct=np.array([m["celltype"] for m in meta]); ag_all=np.array([float(m["age_years"]) for m in meta])
st_all=np.array([m["study"] for m in meta])
k=(ct=="Endo")&(ag_all>0.75)&(nc>=10)
P=sums[k]; CP=P/np.maximum(P.sum(1,keepdims=True),1)*1e4
ag, st = ag_all[k], st_all[k]
keep=np.where((CP>0).mean(0)>=0.20)[0]; G, X = genes[keep], CP[:,keep]
assert len(G)==18794 and k.sum()==65, (len(G), k.sum())
Z=np.log1p(X)
for s in set(st):
    j=st==s; Z[j]=(Z[j]-Z[j].mean(0))/np.maximum(Z[j].std(0),1e-9)
ra=rankdata(ag); rz=rankdata(Z,axis=0)
ra=(ra-ra.mean())/ra.std(); rz=(rz-rz.mean(0))/np.maximum(rz.std(0),1e-9)
R=ra@rz/len(ra)
pos={}
for i,g in enumerate(G): pos.setdefault(g,i)
assert abs(R[pos["PGR"]]-spearmanr(ag,Z[:,pos["PGR"]])[0])<1e-9
Mb=np.array([X[(ag>=lo)&(ag<hi)].mean(0) for lo,hi in B])
fold=(Mb[-1]+1e-4)/(Mb[0]+1e-4)
mono=np.all(np.diff(Mb,axis=0)>=-1e-9,axis=0)
zero=(X[ag<2]==0).mean(0)*100
sel=(fold>=5)&(zero>=50)&mono
rank=np.empty(len(R),int); rank[np.argsort(-R)]=np.arange(1,len(R)+1)
ip=pos["PGR"]
print(f"PGR rho {R[ip]:.4f} rank {rank[ip]} fold {fold[ip]:.2f} zero {zero[ip]:.0f} passing {int(sel.sum())}")
assert abs(R[ip]-0.680)<0.0006 and rank[ip]==7 and abs(fold[ip]-16.97)<0.006 and int(sel.sum())==54, "URETIM KAPISI GECMEDI"
with open(A.out,"w") as fh:
    fh.write("gene\trho_age\trank\tfold_last_over_first\tzero_pct_band1\tmonotonic\tpasses_three_criteria\tband_means\n")
    for i in range(len(G)):
        fh.write(f"{G[i]}\t{R[i]:.4f}\t{rank[i]}\t{fold[i]:.4f}\t{zero[i]:.1f}\t{int(mono[i])}\t{int(sel[i])}\t{';'.join(f'{v:.4f}' for v in Mb[:,i])}\n")
print("written",A.out,len(G))
