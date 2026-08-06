"""Figure 3: the N-mode, subextensive Liouvillian skin sector."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

from lhn import LHNParams, liouvillian, liouvillian_block_q, slow_skin_sector
from lhn.style import use_style, C, panel_label

use_style()
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
BASE = dict(J=1.0, G_R=1.0, G_L=0.35, gphi=8.0)

fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7))

# (a) Liouville-space geometry
ax = axes[0,0]
Ndraw = 9
for a in range(Ndraw):
    for b in range(Ndraw):
        col = C["topo"] if a == b else "0.78"
        size = 42 if a == b else 18
        ax.scatter(b, a, s=size, color=col, edgecolor="black" if a==b else "none", zorder=2)
ax.plot(np.arange(Ndraw), np.arange(Ndraw), color=C["topo"], lw=2.0,
        label="population diagonal ($N$ modes)")
ax.add_patch(plt.Rectangle((-0.45,-0.45), Ndraw-0.1, Ndraw-0.1,
                           fill=False, lw=2.0, color=C["triv"],
                           label="physical ket--bra boundary"))
ax.annotate("biased slow sector", xy=(7,7), xytext=(2.4,7.9),
            arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=8)
ax.set_aspect("equal"); ax.set_xlim(-0.7,Ndraw-0.3); ax.set_ylim(-0.7,Ndraw-0.3)
ax.set_xlabel("bra coordinate $b$"); ax.set_ylabel("ket coordinate $a$")
ax.set_title(r"Liouville space has $N^2$ modes")
ax.legend(loc="lower right", fontsize=7.0)
panel_label(ax,"a")

# metrics at N=16
N0=16
p0=LHNParams(N=N0,pbc=False,**BASE)
met=slow_skin_sector(p0)
vals=met["eigenvalues"]; slow=met["slow_mask"]

# (b) spectrum colored by diagonal weight
ax=axes[0,1]
pp=LHNParams(N=N0,pbc=True,**BASE)
qs=np.linspace(0,2*np.pi,240,endpoint=False)
pbc=np.concatenate([np.linalg.eigvals(liouvillian_block_q(q,pp)) for q in qs])
ax.plot(pbc.real,pbc.imag,".",ms=.65,color="0.65",alpha=.45,label="PBC bands")
sc=ax.scatter(vals.real,vals.imag,c=met["diagonal_weight"],s=19,cmap="viridis",
              edgecolor="black",linewidth=.18,label="physical OBC")
ax.axvline(-4,color="0.5",ls=":",lw=1)
ax.set_xlabel(r"Re $\lambda$"); ax.set_ylabel(r"Im $\lambda$")
ax.set_title(fr"Exactly $N={slow.sum()}$ population-associated OBC modes")
cb=fig.colorbar(sc,ax=ax,fraction=.047,pad=.03); cb.set_label("diagonal weight")
ax.legend(loc="upper left",fontsize=7.0)
panel_label(ax,"b")

# choose representative slow and coherence modes
slow_idx=np.where(slow & (np.abs(vals)>1e-7))[0]
ks=slow_idx[np.argmax(met["centre_of_mass"][slow_idx])]
coh_idx=np.where(~slow)[0]
kc=coh_idx[np.argmin(np.abs(vals[coh_idx].real + (BASE["gphi"]+BASE["G_R"]+BASE["G_L"]))) ]

def operator_weight(k):
    X=met["eigenvectors"][:,k].reshape((N0,N0),order="F")
    w=np.abs(X)**2; w/=w.sum()
    return w

# (c) slow heatmap
for ax,k,title,letter in [
    (axes[0,2],ks,fr"slow skin mode, $\lambda={vals[ks].real:.2f}{vals[ks].imag:+.2f}i$","c"),
    (axes[1,0],kc,fr"coherence mode, $\lambda={vals[kc].real:.2f}{vals[kc].imag:+.2f}i$","d"),
]:
    w=operator_weight(k)
    im=ax.imshow(np.log10(w+1e-12),origin="lower",cmap="magma",vmin=-12,vmax=0,aspect="equal")
    ax.set_xlabel("bra $b$"); ax.set_ylabel("ket $a$")
    ax.set_title(title)
    cb=fig.colorbar(im,ax=ax,fraction=.047,pad=.03); cb.set_label(r"$\log_{10}|X_{ab}|^2$")
    panel_label(ax,letter)

# finite-size metrics
Ns=np.array([6,8,10,12,14,16,18])
counts=[]; frac=[]; ipr_s=[]; ipr_c=[]; com_s=[]; com_c=[]
for N in Ns:
    m=slow_skin_sector(LHNParams(N=int(N),pbc=False,**BASE))
    s=m["slow_mask"]
    counts.append(s.sum()); frac.append(s.mean())
    ipr_s.append(m["ipr"][s].mean()); ipr_c.append(m["ipr"][~s].mean())
    com_s.append(m["centre_of_mass"][s].mean()); com_c.append(m["centre_of_mass"][~s].mean())
counts=np.array(counts); frac=np.array(frac); ipr_s=np.array(ipr_s); ipr_c=np.array(ipr_c)

# (e) count and fraction
ax=axes[1,1]
ax.plot(Ns,counts,"o-",color=C["topo"],mec="black",mew=.3,label=r"skin-sector count $n_{\rm skin}$")
ax.plot(Ns,Ns,"--",color="black",lw=1.2,label=r"$N$")
ax.set_xlabel("physical size $N$"); ax.set_ylabel("number of modes",color=C["topo"])
ax.tick_params(axis="y",colors=C["topo"])
ax2=ax.twinx(); ax2.plot(Ns,frac,"s-",color=C["triv"],mec="black",mew=.3,label=r"fraction $n_{\rm skin}/N^2$")
ax2.plot(Ns,1/Ns,":",color=C["gray"],lw=1.5,label=r"$1/N$")
ax2.set_ylabel("fraction of Liouvillian modes",color=C["triv"]); ax2.tick_params(axis="y",colors=C["triv"]); ax2.grid(False)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,loc="center left",fontsize=7.0)
ax.set_title("Extensive in Hilbert space, subextensive in Liouville space")
panel_label(ax,"e")

# (f) IPR scaling
ax=axes[1,2]
s_s=np.polyfit(np.log(Ns),np.log(ipr_s),1)[0]
s_c=np.polyfit(np.log(Ns),np.log(ipr_c),1)[0]
ax.loglog(Ns,ipr_s,"o-",color=C["topo"],mec="black",mew=.3,label=fr"slow sector $\sim N^{{{s_s:.2f}}}$")
ax.loglog(Ns,ipr_c,"s-",color=C["gray"],mec="black",mew=.3,label=fr"coherences $\sim N^{{{s_c:.2f}}}$")
ax.set_xlabel("physical size $N$"); ax.set_ylabel(r"mean two-dimensional IPR")
ax.set_title("Localized slow modes, extended coherence modes")
ax.legend(loc="best",fontsize=7.1)
panel_label(ax,"f")

fig.tight_layout()
path=os.path.join(OUT,"fig3_subextensive_skin.png")
fig.savefig(path)
print("counts:",dict(zip(Ns.tolist(),counts.tolist())))
print(f"IPR slopes: slow={s_s:.4f}, coherence={s_c:.4f}")
print("saved",path)
