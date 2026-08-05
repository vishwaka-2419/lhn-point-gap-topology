# Analytical definitions and numerical conventions

## 1. Lindblad chain

In the single-excitation basis \(\{|j\rangle\}_{j=0}^{N-1}\), the model is

\[
\dot\rho=\mathcal L[\rho]
=-i[H,\rho]+
\sum_\mu\left(A_\mu\rho A_\mu^\dagger-
\frac12\{A_\mu^\dagger A_\mu,\rho\}\right),
\]

with reciprocal coherent hopping

\[
H=J\sum_j\bigl(|j+1\rangle\langle j|+|j\rangle\langle j+1|\bigr),
\]

biased incoherent jumps

\[
R_j=\sqrt{\Gamma_R}|j+1\rangle\langle j|,\qquad
L_j=\sqrt{\Gamma_L}|j\rangle\langle j+1|,
\]

and local dephasing

\[
D_j=\sqrt{\gamma_\phi}|j\rangle\langle j|.
\]

The numerical implementation uses column-stacking vectorisation,
\(\mathrm{vec}(A\rho B)=(B^T\otimes A)\mathrm{vec}(\rho)\).

## 2. Momentum-block decomposition

For periodic boundaries, write
\(E_{j+r,j}=|j+r\rangle\langle j|\) and Fourier transform the centre-of-mass
coordinate. The resulting \(N\times N\) block \(\mathcal L_q\), acting on the
relative coordinate \(r\), has

\[
[\mathcal L_q]_{r+1,r}=-iJ(1-e^{iq}),\qquad
[\mathcal L_q]_{r-1,r}=-iJ(1-e^{-iq}),
\]

\[
[\mathcal L_q]_{r,r}=-(\Gamma_R+\Gamma_L+\gamma_\phi),\quad r\ne0,
\]

and

\[
[\mathcal L_q]_{0,0}=-(\Gamma_R+\Gamma_L)
+\Gamma_Re^{-iq}+\Gamma_Le^{iq}.
\]

The union of the spectra of these blocks at the allowed discrete momenta agrees
with the full periodic Liouvillian spectrum to numerical precision.

For \(J\ne0\), the population coordinate \(r=0\) couples to coherences. The
population subspace is therefore not an exact invariant subspace at finite
coherent hopping. The scalar \(r=0\) entry is the classical biased-walk Bloch
symbol, but a closed classical generator is obtained only at \(J=0\) or after a
controlled elimination of coherences.

## 3. No-jump generator

For periodic boundaries,

\[
H_{\rm eff}=H-\frac{i}{2}\sum_\mu A_\mu^\dagger A_\mu
=H-\frac{i}{2}(\Gamma_R+\Gamma_L+\gamma_\phi)\mathbb 1.
\]

Thus

\[
H_{\rm eff}(k)=2J\cos k-
\frac{i}{2}(\Gamma_R+\Gamma_L+\gamma_\phi),
\]

which is a line segment traced in both directions. Reference points outside the
segment are point-gapped, but the winding is zero. Under open boundaries,
missing edge jump channels produce nonuniform imaginary onsite shifts; the open
and periodic spectra need not coincide. The robust conclusion is zero bulk
point-gap winding, not identical finite-chain spectra.

In this construction, the non-reciprocal momentum dependence
\(\Gamma_Re^{-iq}+\Gamma_Le^{iq}\) enters through recycling terms. The winding
is a property of the complete operator family and is not additively assigned to
one term.

## 4. Point-gap winding

For a parameterised family \(M(q)\) and a reference point \(\lambda\) excluded
from its spectrum,

\[
w(\lambda)=\frac{1}{2\pi i}\int_0^{2\pi}dq\,
\partial_q\ln\det[M(q)-\lambda].
\]

The code uses the same determinant-phase procedure for the Liouvillian blocks,
the classical Bloch generator, and the no-jump Hamiltonian. The finite-grid
point-gap margin is evaluated with

\[
\Delta_{\rm pg}(\lambda)=
\min_q\sigma_{\min}[M(q)-\lambda],
\]

rather than eigenvalue distance alone, because the matrices are non-normal.

When \(\Gamma_R=\Gamma_L\),
\(\mathcal L_q^T=\mathcal L_{-q}\). Consequently,
\(\det[\mathcal L_q-\lambda]\) is even under \(q\mapsto-q\), and the winding
vanishes for every open point gap.

## 5. Strong-dephasing reduction

Partition the block into the population coordinate and coherence coordinates.
At spectral parameter \(z\), the exact algebraic reduction is

\[
\mathcal L_{\rm eff}(q;z)=
\mathcal L_{00}+\mathcal L_{0r}(z-\mathcal L_{rr})^{-1}\mathcal L_{r0}.
\]

A frequency-independent Markov generator requires two approximations:

1. set \(z=0\), appropriate for slow modes near the stationary state;
2. expand to second order in \(J/\kappa\), where
   \(\kappa=\gamma_\phi+\Gamma_R+\Gamma_L\).

This gives

\[
\mathcal L_{\rm eff}(q;0)\simeq
-(\Gamma_R+\Gamma_L)+\Gamma_Re^{-iq}+\Gamma_Le^{iq}
-\frac{4J^2}{\kappa}(1-\cos q).
\]

The last term is a symmetric random-walk contribution with rate

\[
D=\frac{2J^2}{\gamma_\phi+\Gamma_R+\Gamma_L}.
\]

The effective rates are \(\Gamma_R+D\) and \(\Gamma_L+D\). Their difference is
unchanged at this order; their ratio is reduced. Persistence of a winding still
requires a specified point gap to remain open. Figure 2 follows the fixed
reference point \(\lambda_0=-0.4\); the full-Liouvillian singular-value margin
remains nonzero along the displayed path. Agreement with the local Markov
reduction is asserted only in the strong-dephasing range.


## 6. Exact finite-size determinant and scalar reduction

Let

\[
u(q)=-iJ(1-e^{iq}),\qquad v(q)=-iJ(1-e^{-iq}),
\]

and define \(\alpha=\kappa+\lambda\). After separating the population
coordinate, the coherence block is

\[
\mathsf D_\lambda(q)=-\alpha I+K(q),\qquad K^\dagger=-K.
\]

Its eigenvalues are

\[
\mu_m=-\alpha+4iJ\sin(q/2)\cos(m\pi/N),\qquad m=1,\ldots,N-1.
\]

For a real reference point with \(\alpha>0\), the coherence determinant is
nonzero and has zero winding. Define

\[
\Theta_0=1,\quad \Theta_1=-\alpha,\quad
\Theta_m=-\alpha\Theta_{m-1}-uv\Theta_{m-2}.
\]

With

\[
A_\lambda(q)=-(\Gamma_R+\Gamma_L)+\Gamma_Re^{-iq}
+\Gamma_Le^{iq}-\lambda,
\]

the exact finite-size determinant is

\[
\det(\mathcal L_q-\lambda I)=A_\lambda\Theta_{N-1}
-2uv\Theta_{N-2}-(-1)^N(u^N+v^N),
\]

and the exact scalar Schur complement is

\[
S_N=A_\lambda-
\frac{2uv\Theta_{N-2}+(-1)^N(u^N+v^N)}{\Theta_{N-1}}.
\]

Thus the full Liouvillian winding equals the winding of \(S_N\) whenever the
coherence block remains invertible.

## 7. Sufficient winding-equality certificate

The second-order classical symbol is

\[
W_{\rm eff}(q)=(\Gamma_R+D)e^{-iq}+(\Gamma_L+D)e^{iq}
-(\Gamma_R+\Gamma_L+2D).
\]

For \(N\ge4\), let \(c_N=\cos(\pi/N)\). If
\(\alpha>4|J|c_N\), then

\[
\sup_q\left|S_N(q,\lambda)-[W_{\rm eff}(q)-\lambda]\right|
\le \epsilon_N(\lambda),
\]

where

\[
\epsilon_N(\lambda)=
\frac{8J^2|\lambda|}{\kappa\alpha}
+
\frac{128J^4c_N^2}{\alpha^3[1-4|J|c_N/\alpha]}.
\]

For a real reference point \(\lambda=-x<0\), define

\[
\Sigma=\Gamma_R+\Gamma_L+\frac{4J^2}{\kappa},\qquad
\Delta=\Gamma_R-\Gamma_L.
\]

The exact classical point-gap margin is

\[
m_{\rm cl}^2=
\min_{0\le t\le2}[(x-\Sigma t)^2+\Delta^2(2t-t^2)].
\]

If \(m_{\rm cl}>\epsilon_N\), the straight-line homotopy between the exact
scalar and the Markov symbol cannot cross zero, and

\[
w_{\mathcal L}(\lambda)=w_{\rm eff}(\lambda).
\]

Replacing \(c_N\) by one gives a conservative sufficient condition uniform over
every finite \(N\ge4\). This criterion is sufficient rather than necessary.
Below the certified threshold, direct numerical continuation is reported
separately.

## 8. Boundary sensitivity and finite-size diagnostics

At \(J=0\), the open biased chain has the exact stationary distribution

\[
p_j\propto(\Gamma_R/\Gamma_L)^j.
\]

At finite \(J\), the profile remains strongly edge-biased but is not exactly the
same geometric distribution. The calculations also show pronounced changes
between periodic- and open-boundary spectra.

These observations do not by themselves establish an extensive Liouvillian skin
effect. The present finite-size eigenoperator diagnostics show only a modest
centre-of-mass shift, a decreasing fraction of strongly edge-localised modes over
the sampled sizes, and finite-size dissipative-gap fits that do not demonstrate
asymptotic saturation. The manuscript therefore uses the narrower term
“boundary sensitivity.”

## 9. Boundary-rate Fisher information

A weak return jump from site \(N-1\) to site 0 is parameterised by a nonnegative
rate \(\theta\). Because its dissipator is linear in \(\theta\),
\(\partial_\theta\mathcal L\) is obtained analytically. At \(\theta=0\), the
calculation is a one-sided derivative \(\theta\to0^+\).

The stationary-state derivative satisfies

\[
\mathcal L\,\partial_\theta\rho_{\rm ss}
=-(\partial_\theta\mathcal L)\rho_{\rm ss},
\qquad
\mathrm{Tr}\,\partial_\theta\rho_{\rm ss}=0.
\]

For an ideal site-occupation measurement,

\[
F_C=\sum_j\frac{(\partial_\theta p_j)^2}{p_j}.
\]

In the biased chain, exponentially small stationary probabilities coexist with a
response that is not exponentially suppressed, producing exponential growth of
\(F_C\) over the calculated size range. In the reciprocal chain,
\(p_j=1/N\), while \(\partial_\theta p_j=\mathcal O(1)\) over an
\(\mathcal O(N)\) fraction of the chain; therefore \(F_C\sim N^2\).

This Fisher information depends on the chosen rate parameter and transforms
under reparameterisation. It is not topologically quantised and is not
normalised by preparation time, relaxation time, observation time, detector
noise, or control resources.

## 10. Exceptional-point reference model

For a resonantly driven, relaxing two-level system,

\[
H=\frac{\Omega}{2}\sigma_x+\frac{\delta}{2}\sigma_z,
\qquad A=\sqrt{\gamma}\,\sigma_-,
\]

the relevant Bloch eigenvalues coalesce at \(\Omega_{\rm EP}=\gamma/4\). A drive
perturbation within the defective block produces square-root splitting, whereas a
detuning perturbation at the same point is linear in the tested regime. The
stationary-state quantum Fisher information for drive estimation is smooth
through the exceptional point for this model. This is not a general statement
about all probes, initial states, transient protocols, or output measurements.

## 11. Disorder ensemble

For each bond \(b\), the classical rates are

\[
\Gamma_{R,b}=\Gamma_R(1+Wu_{R,b}),\qquad
\Gamma_{L,b}=\Gamma_L(1+Wu_{L,b}),
\]

where the independent base variables are drawn uniformly from \([-1,1]\). A
fixed pair of base arrays is reused as \(W\) changes, defining a continuous
disorder path. The scripts restrict the displayed range so all rates remain
strictly positive and reject any non-positive rate. The real-space winding is
computed by twisting the wrap-around bond while keeping the reference point
fixed.

## 12. ESR-STM conventions

Experimental coherent couplings are generally reported as cyclic frequencies
\(f\), whereas Hamiltonian coefficients in the Lindblad equation are angular
rates \(2\pi f\). Decay rates \(1/T_1\) and \(1/T_2\) do not require this
conversion.

Hahn-echo and dynamical-decoupling times are sequence-dependent refocused
quantities and cannot be inserted uncritically as the transverse rate of a
time-independent Markovian generator. The approximately 40 ns Rabi-envelope
decay reported for Ti is treated only as an indicative driven-coherence scale.
The quoted Ti--Ti exchange parameter applies to a specific close dimer geometry;
separate dipolar estimates are order-of-magnitude values for the stated
orientation and are not lower bounds.

The model requires directional dissipative hopping. Reciprocal exchange and
dipolar interactions do not supply it. In addition, a translationally invariant
biased ring has nonzero cycle affinity and cannot satisfy equilibrium detailed
balance with a single-valued static potential. The ESR-STM discussion therefore
identifies an implementation requirement rather than an existing realisation.
