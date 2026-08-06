"""Supplementary Figure S2: synthetic versus physical opening."""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from lhn import LHNParams, synthetic_liouville_obc, slow_skin_sector
from lhn.style import use_style,C,panel_label
use_style(); OUT=os.path.join(os.path.dirname(__file__),"..","figures")
p=LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=8,pbc=True)
Ms=np.array([8,12,20,30,40]); syn_slow=[]; syn_coh=[]
for M in Ms:
 L=synthetic_liouville_obc(p,int(M)); vals,vec=eig(L); mask=vals.real>-4
 cs=[]; cc=[]
 for k in range(len(vals)):
  v=vec[:,k].reshape((M,p.N)); w=np.sum(np.abs(v)**2,axis=1); w/=w.sum(); c=(w*np.arange(M)).sum()/(M-1)
  (cs if mask[k] else cc).append(c)
 syn_slow.append(np.mean(cs)); syn_coh.append(np.mean(cc))
Ns=np.array([6,8,10,12,14,16,18]); phys=[]
for N in Ns:
 m=slow_skin_sector(LHNParams(N=int(N),J=1,G_R=1,G_L=.35,gphi=8,pbc=False)); phys.append(m["slow_mask"].mean())
fig,axes=plt.subplots(1,3,figsize=(11.5,3.5))
ax=axes[0]; ax.plot(Ms,syn_slow,"o-",color=C["topo"],label="synthetic slow band"); ax.plot(Ms,syn_coh,"s-",color=C["gray"],label="synthetic coherences"); ax.axhline(.5,color="black",lw=.8,ls=":"); ax.set_xlabel("opened cells $M$"); ax.set_ylabel("mean cell centre"); ax.set_title("Fixed internal dimension $R=10$"); ax.legend(fontsize=7); panel_label(ax,"a")
ax=axes[1]; ax.plot(Ms,np.full_like(Ms,1/p.N,dtype=float),"o-",color=C["topo"],label=r"synthetic: $M/(MR)=1/R$"); ax.plot(Ns,phys,"s-",color=C["triv"],label=r"physical: $N/N^2$"); ax.plot(Ns,1/Ns,":",color="black",label=r"$1/N$"); ax.set_xlabel("linear size"); ax.set_ylabel("skin-sector fraction"); ax.set_title("Different thermodynamic counting"); ax.legend(fontsize=7); panel_label(ax,"b")
ax=axes[2]; ax.axis("off"); ax.text(.05,.82,"Synthetic opening",fontweight="bold"); ax.text(.05,.68,r"$M$ cells $\times$ fixed $R$ internal states",fontsize=9); ax.text(.05,.55,r"one winding band $\Rightarrow M$ skin modes",fontsize=9,color=C["topo"]); ax.text(.05,.36,"Physical opening",fontweight="bold"); ax.text(.05,.22,r"$N$ ket sites $\times N$ bra sites",fontsize=9); ax.text(.05,.09,r"slow sector $N$ of $N^2$ modes",fontsize=9,color=C["triv"]); ax.set_title("Boundary geometry changes the counting"); panel_label(ax,"c",dx=-.02)
fig.tight_layout(); path=os.path.join(OUT,"figS2_boundary_geometry.png"); fig.savefig(path); print("saved",path)
