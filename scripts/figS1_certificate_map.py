"""Supplementary Figure S1: lambda-gamma certificate map."""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
import numpy as np
import matplotlib.pyplot as plt
from lhn import LHNParams, exact_characteristic_determinant, effective_markov_symbol, winding_number, winding_certificate, pointwise_gap_reserve
from lhn.style import use_style,C,panel_label
use_style(); OUT=os.path.join(os.path.dirname(__file__),"..","figures")
N=10; base=dict(N=N,J=1.0,G_R=1.0,G_L=.35,pbc=True)
gs=np.geomspace(.3,30,35); lams=np.linspace(-2.6,-.05,50)
WG=np.zeros((len(gs),len(lams))); WP=np.zeros_like(WG); GLOB=np.zeros_like(WG,dtype=bool); POINT=np.zeros_like(WG,dtype=bool)
qs=np.linspace(0,2*np.pi,360,endpoint=False)
for iy,g in enumerate(gs):
 p=LHNParams(gphi=float(g),**base)
 for ix,lam in enumerate(lams):
  fn=lambda q,p=p,lam=lam: np.array([[exact_characteristic_determinant(q,p,lam)]])
  WG[iy,ix]=winding_number(fn,0,n_q=192)
  ef=lambda q,p=p,lam=lam: np.array([[effective_markov_symbol(q,p)-lam]])
  WP[iy,ix]=winding_number(ef,0,n_q=192)
  GLOB[iy,ix]=winding_certificate(p,lam,uniform_in_N=True)["certified"]
  POINT[iy,ix]=np.min(pointwise_gap_reserve(qs,p,lam,uniform_in_N=True))>0
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for ax,Z,title,letter in [(axes[0],WG,"exact finite-$N$ winding","a"),(axes[1],WP,"effective Markov winding","b")]:
 im=ax.pcolormesh(lams,gs,Z,cmap="RdBu_r",vmin=-1,vmax=1,shading="auto"); ax.set_yscale("log"); ax.set_xlabel(r"real reference point $\lambda$"); ax.set_ylabel(r"$\gamma_\phi$"); ax.set_title(title); panel_label(ax,letter)
fig.colorbar(im,ax=axes[:2],fraction=.025,pad=.02,label="winding")
ax=axes[2]; cat=GLOB.astype(int)+POINT.astype(int); im=ax.pcolormesh(lams,gs,cat,cmap="viridis",vmin=0,vmax=2,shading="auto"); ax.set_yscale("log"); ax.set_xlabel(r"real reference point $\lambda$"); ax.set_ylabel(r"$\gamma_\phi$"); ax.set_title("certificate hierarchy"); panel_label(ax,"c"); cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.03,ticks=[0,1,2]); cb.ax.set_yticklabels(["none","pointwise","global+pointwise"])
fig.tight_layout(); path=os.path.join(OUT,"figS1_certificate_map.png"); fig.savefig(path); print("saved",path)
