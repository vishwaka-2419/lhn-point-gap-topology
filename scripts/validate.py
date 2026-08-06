"""Deterministic validation suite for the revised manuscript."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

from lhn import (
    LHNParams, liouvillian, liouvillian_block_q, pbc_spectrum_from_blocks,
    winding_number, bloch_classical, bloch_postselected,
    exact_characteristic_determinant, exact_schur_scalar,
    coherence_eigenvalues, effective_markov_symbol,
    classical_point_gap_margin, analytical_error_bound, pointwise_error_bound,
    pointwise_gap_reserve, winding_certificate,
    interval_validated_pointwise_certificate, dimensionless_coordinates,
    centered_dimensionless_uniform_reserve, slow_skin_sector,
    liouville_bloch_blocks, synthetic_liouville_obc,
)

results=[]
def check(name, condition, detail=""):
    condition=bool(condition); results.append(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))

print("\n--- 1. Liouvillian structure and Bloch decomposition ---")
p=LHNParams(N=8,J=1,G_R=1,G_L=.4,gphi=.3,pbc=True)
L=liouvillian(p)
tr=np.eye(p.N).reshape(-1,order="F").conj()
check("trace preservation",np.max(np.abs(tr@L))<1e-10,f"{np.max(np.abs(tr@L)):.2e}")
full=np.linalg.eigvals(L); blocks=pbc_spectrum_from_blocks(p,liouvillian_block_q)
cost=np.abs(full[:,None]-blocks[None,:]); r,c=linear_sum_assignment(cost)
check("PBC spectrum equals union of momentum blocks",cost[r,c].max()<1e-9,f"{cost[r,c].max():.2e}")
check("biased classical winding is nonzero",abs(winding_number(bloch_classical(1,.4),-.7))==1)
check("reciprocal no-jump Bloch Hamiltonian has zero winding",winding_number(bloch_postselected(p),-.85j)==0)

print("\n--- 2. Exact finite-size identities ---")
rng=np.random.default_rng(20260806)
md=ms=mc=0.0
for N in [4,5,7,10,14]:
    for _ in range(7):
        pa=LHNParams(N=N,J=float(rng.uniform(.1,1.2)),G_R=float(rng.uniform(.5,1.5)),
                     G_L=float(rng.uniform(.1,.8)),gphi=float(rng.uniform(1,18)),pbc=True)
        lam=float(rng.uniform(-.7,-.05)); q=float(rng.uniform(0,2*np.pi))
        M=liouvillian_block_q(q,pa)-lam*np.eye(N)
        d0=np.linalg.det(M); d1=exact_characteristic_determinant(q,pa,lam)
        md=max(md,abs(d0-d1)/max(1,abs(d0)))
        D=M[1:,1:]; s0=M[0,0]-(M[0:1,1:]@np.linalg.solve(D,M[1:,0:1]))[0,0]
        s1=exact_schur_scalar(q,pa,lam); ms=max(ms,abs(s0-s1)/max(1,abs(s0)))
        e0=np.linalg.eigvals(D); e1=coherence_eigenvalues(q,pa,lam)
        cc=np.abs(e0[:,None]-e1[None,:]); rr,qq=linear_sum_assignment(cc); mc=max(mc,float(cc[rr,qq].max()))
check("characteristic determinant",md<2e-11,f"max rel={md:.2e}")
check("scalar Schur complement",ms<2e-12,f"max rel={ms:.2e}")
check("coherence-block eigenvalues",mc<2e-11,f"max abs={mc:.2e}")

print("\n--- 3. Classical margin and error bounds ---")
max_margin=0.0; max_global_ratio=0.0; max_point_ratio=0.0
for g in [4.5,5,8,12,30]:
    pa=LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=g,pbc=True); lam=-.4
    qs=np.linspace(0,2*np.pi,24001,endpoint=False)
    dense=np.min(np.abs(np.array([effective_markov_symbol(q,pa)-lam for q in qs])))
    max_margin=max(max_margin,abs(dense-classical_point_gap_margin(pa,lam)))
    actual=np.array([abs(exact_schur_scalar(q,pa,lam)-(effective_markov_symbol(q,pa)-lam)) for q in qs])
    gb=analytical_error_bound(pa,lam)
    pb=pointwise_error_bound(qs,pa,lam)
    max_global_ratio=max(max_global_ratio,float(actual.max()/gb))
    mask=(pb>1e-13)&np.isfinite(pb)
    max_point_ratio=max(max_point_ratio,float(np.max(actual[mask]/pb[mask])))
    check_zero=np.max(actual[~mask]) if np.any(~mask) else 0.0
    if check_zero>1e-10: max_point_ratio=np.inf
check("closed-form ellipse margin",max_margin<3e-8,f"max abs={max_margin:.2e}")
check("actual error below global bound",max_global_ratio<1,f"max ratio={max_global_ratio:.3f}")
check("actual error below pointwise bound",max_point_ratio<1+1e-5,f"max ratio={max_point_ratio:.3f}")

print("\n--- 4. Sharpened all-N certificate ---")
p432=LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=4.32,pbc=True)
p433=p432.copy(gphi=4.33)
c432=interval_validated_pointwise_certificate(p432,-.4,uniform_in_N=True,initial_intervals=256,max_depth=18)
c433=interval_validated_pointwise_certificate(p433,-.4,uniform_in_N=True,initial_intervals=256,max_depth=18)
check("pointwise interval certificate is inconclusive below threshold",not c432["certified"],str(c432["unresolved_interval"]))
check("pointwise interval certificate holds at gamma_phi=4.33",c433["certified"],f"reserve lb={c433['minimum_interval_reserve']:.3e}")
old=winding_certificate(LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=8.1,pbc=True),-.4,uniform_in_N=True)
check("original global all-N certificate remains valid at 8.1",old["certified"],f"reserve={old['reserve']:.3e}")
for g in [4.33,5,8,30,1000]:
    pa=LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=g,pbc=True)
    wf=winding_number(lambda q,pa=pa: liouvillian_block_q(q,pa),-.4,n_q=768)
    wm=winding_number(lambda q,pa=pa: np.array([[effective_markov_symbol(q,pa)]]),-.4,n_q=768)
    check(f"full and Markov winding agree at gamma_phi={g:g}",wf==wm==-1,f"{wf},{wm}")

print("\n--- 5. Dimensionless theorem ---")
pa=LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=8,pbc=True)
k=pa.gphi+pa.G_R+pa.G_L; sig=pa.G_R+pa.G_L+4*pa.J**2/k; lam=-sig
co=dimensionless_coordinates(pa,lam)
dim=centered_dimensionless_uniform_reserve(co["j"],co["g"],co["delta"])
dim_direct=winding_certificate(pa,lam,uniform_in_N=True)["reserve"]/k
check("dimensionless centered reserve equals dimensional reserve/kappa",abs(dim-dim_direct)<1e-12,f"{dim:.6e}")
check("pointwise dimensionless criterion enlarges the centered certified region",
      np.min(pointwise_gap_reserve(np.linspace(0,2*np.pi,8192,endpoint=False),pa,lam,uniform_in_N=True))>0 and dim<0)

print("\n--- 6. Subextensive skin sector ---")
Ns=np.array([6,8,10,12,14,16,18]); counts=[]; iprs=[]; iprc=[]; separations=[]; min_com=[]; mean_com=[]
for N in Ns:
    m=slow_skin_sector(LHNParams(N=int(N),J=1,G_R=1,G_L=.35,gphi=8,pbc=False))
    s=m["slow_mask"]; counts.append(int(s.sum())); iprs.append(m["ipr"][s].mean()); iprc.append(m["ipr"][~s].mean())
    separations.append(m["diagonal_weight"][s].min()-m["diagonal_weight"][~s].max())
    min_com.append(m["centre_of_mass"][s].min()); mean_com.append(m["centre_of_mass"][s].mean())
check("exactly N population-associated modes over tested sizes",np.array_equal(counts,Ns),str(counts))
check("diagonal-weight classification is spectrally separated",min(separations)>.7,f"min separation={min(separations):.3f}")
ss=np.polyfit(np.log(Ns),np.log(iprs),1)[0]; sc=np.polyfit(np.log(Ns),np.log(iprc),1)[0]
check("slow-sector IPR remains O(1)",abs(ss)<.2,f"slope={ss:.3f}")
check("coherence-sector IPR scales approximately as N^-2",abs(sc+2)<.2,f"slope={sc:.3f}")
check("skin-sector fraction is exactly 1/N",np.allclose(np.array(counts)/Ns**2,1/Ns))
check("every classified slow mode is directionally edge biased",min(min_com)>.64,f"minimum COM={min(min_com):.3f}")
check("mean slow-sector accumulation strengthens with size",np.all(np.diff(mean_com)>0) and mean_com[-1]>.92,f"N=18 mean COM={mean_com[-1]:.3f}")

print("\n--- 7. Synthetic boundary check ---")
b0,bp,bm,res=liouville_bloch_blocks(LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=8,pbc=True))
check("Bloch family contains only nearest cell harmonics",res<1e-11,f"res={res:.2e}")
for M in [10,20,30]:
    Ls=synthetic_liouville_obc(LHNParams(N=10,J=1,G_R=1,G_L=.35,gphi=8,pbc=True),M)
    vals,vec=eig(Ls); s=vals.real>-4
    check(f"synthetic opening has M slow modes for M={M}",s.sum()==M,f"count={s.sum()}")
    if M>=20:
        com=[]
        for k0 in np.where(s)[0]:
            v=vec[:,k0].reshape(M,10); w=np.sum(np.abs(v)**2,axis=1); w/=w.sum(); com.append((w*np.arange(M)).sum()/(M-1))
        check(f"synthetic slow modes accumulate at one edge for M={M}",np.mean(com)>.9,f"mean COM={np.mean(com):.3f}")

print("\n--- summary ---")
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
