# Analytical derivation and numerical conventions

## 1. Model

In the one-excitation basis, the Lindblad generator uses

\[
H=J\sum_j(|j+1\rangle\langle j|+|j\rangle\langle j+1|),
\]

\[
R_j=\sqrt{\Gamma_R}|j+1\rangle\langle j|,\quad
L_j=\sqrt{\Gamma_L}|j\rangle\langle j+1|,\quad
D_j=\sqrt{\gamma_\phi}|j\rangle\langle j|.
\]

Column-stacking is used:
`vec(A rho B) = (B.T kron A) vec(rho)`.

## 2. Momentum blocks

For periodic boundaries, Fourier transformation of
`E_{j+r,j}=|j+r><j|` in `j` gives

\[
[\mathcal L_q]_{r+1,r}=-iJ(1-e^{iq}),\qquad
[\mathcal L_q]_{r-1,r}=-iJ(1-e^{-iq}),
\]

\[
[\mathcal L_q]_{r,r}=-(\Gamma_R+\Gamma_L+\gamma_\phi),\quad r\ne0,
\]

\[
[\mathcal L_q]_{0,0}=-(\Gamma_R+\Gamma_L)
+\Gamma_Re^{-iq}+\Gamma_Le^{iq}.
\]

The union of the block spectra at the allowed discrete momenta equals the full
periodic Liouvillian spectrum.

## 3. Exact determinant and Schur scalar

Define

\[
u=-iJ(1-e^{iq}),\quad v=-iJ(1-e^{-iq}),\quad
\kappa=\gamma_\phi+\Gamma_R+\Gamma_L,\quad
\alpha=\kappa+\lambda,
\]

and

\[
A_\lambda=-(\Gamma_R+\Gamma_L)+\Gamma_Re^{-iq}+\Gamma_Le^{iq}-\lambda.
\]

The coherence-block eigenvalues are

\[
-\alpha+4iJ\sin(q/2)\cos(m\pi/N),\quad m=1,\ldots,N-1.
\]

With

\[
\Theta_0=1,\quad\Theta_1=-\alpha,\quad
\Theta_m=-\alpha\Theta_{m-1}-uv\Theta_{m-2},
\]

\[
\det(\mathcal L_q-\lambda I)=A_\lambda\Theta_{N-1}
-2uv\Theta_{N-2}-(-1)^N(u^N+v^N),
\]

\[
S_N=A_\lambda-
\frac{2uv\Theta_{N-2}+(-1)^N(u^N+v^N)}{\Theta_{N-1}}.
\]

When `alpha > 0`, the coherence determinant has zero winding, so the full
Liouvillian winding is the winding of `S_N`.

## 4. Effective Markov symbol

The coherence linewidth is

\[
\kappa=\gamma_\phi+\Gamma_R+\Gamma_L,
\]

not `gamma_phi` alone. Second-order zero-frequency elimination gives a symmetric
rate

\[
D=2J^2/\kappa
\]

and

\[
W_{\rm eff}(q)=(\Gamma_R+D)e^{-iq}+(\Gamma_L+D)e^{iq}
-(\Gamma_R+\Gamma_L+2D).
\]

## 5. Global certificate

For `N >= 4`, `c_N=cos(pi/N)`, and `alpha > 4|J|c_N`,

\[
\sup_q|S_N-(W_{\rm eff}-\lambda)|\le
\frac{8J^2|\lambda|}{\kappa\alpha}
+\frac{128J^4c_N^2}{\alpha^3(1-4|J|c_N/\alpha)}.
\]

For `lambda=-x<0`, define

\[
\Sigma=\Gamma_R+\Gamma_L+4J^2/\kappa,\qquad
\Delta=\Gamma_R-\Gamma_L.
\]

The exact classical margin is

\[
m_{\rm cl}^2=\min_{0\le t\le2}
[(x-\Sigma t)^2+\Delta^2(2t-t^2)].
\]

If the margin exceeds the error bound, the exact and effective windings are
equal. Setting `c_N=1` gives an all-finite-size sufficient condition.

## 6. Pointwise certificate

Retaining `s_q=|sin(q/2)|` yields

\[
\epsilon_{\rm pw}(q)=
\frac{8J^2|\lambda|s_q^2}{\kappa\alpha}
+\frac{128J^4c_N^2s_q^4}
{\alpha^3(1-4|J|c_Ns_q/\alpha)}.
\]

The sharper sufficient condition is

\[
\min_q\{|W_{\rm eff}(q)-\lambda|-\epsilon_{\rm pw}(q)\}>0.
\]

`interval_validated_pointwise_certificate` proves this over the full Brillouin
zone by interval subdivision. It does not infer a proof from a finite momentum
grid.

## 7. Dimensionless form

For `lambda=-x`, use

\[
j=|J|/\kappa,\quad g=\gamma_\phi/\kappa,\quad
\xi=x/\kappa,\quad s=1-g+4j^2,\quad
\delta=(\Gamma_R-\Gamma_L)/\Sigma.
\]

Then

\[
|W_{\rm eff}-\lambda|/\kappa=
\sqrt{[\xi-s(1-\cos q)]^2+\delta^2s^2\sin^2q},
\]

and the normalized pointwise error is implemented in
`centered_dimensionless_pointwise_reserve` for the centred reference
`lambda=-Sigma`.

## 8. Subextensive skin count

At `J=0`, physical OBC gives

\[
\mathcal L_{\rm OBC}=W_{\rm OBC}\oplus\mathcal C,
\]

where `W_OBC` has dimension `N` and the coherence sector has dimension
`N(N-1)`. With

\[
S_{jj}=(\Gamma_R/\Gamma_L)^{j/2},
\]

`S^{-1} W_OBC S` is symmetric. All `N` population right eigenvectors therefore
carry the directional envelope `S`, while the full Liouville space has `N^2`
modes. Thus `n_skin=N` and the fraction is `1/N`.

At finite `J` and strong dephasing, `slow_skin_sector` identifies the
population-associated continuation by diagonal weight and reports full
ket--bra IPR diagnostics. The exact finite-`J` count is a tested numerical
result in the stated parameter range, not a general theorem.
