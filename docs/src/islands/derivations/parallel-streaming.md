# Derivation — the parallel (island) streaming coefficients

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7)
of the island-induced parallel-streaming channel of the master orbit-averaged
drift-kinetic equation.
**Clears:** the `Operators.ParallelStreaming` coefficients `a_xi`, `a_x`
(`[VERIFY: I19 Eq. (32) streaming term, with the L23 §2.6 amendments]`,
QUESTIONS Q5), building on the already-cleared `ω̂_D`
(`magnetic_drift_frequency`) whose normalization it must match.
**Status:** ✅ **signed off 2026-07-11** — implemented as
`Configure.streaming_coefficients` and wired into `Operators.ParallelStreaming`;
the `{Ω, g}` advection structure (§3) is verified in
`test/runtests_islands_configure.jl`.

## 1. The channel in the master equation

The Level-0 master equation is I19 Eq. (32) (docs/01 §2), the orbit-averaged DKE
for `Ḡ₀(p̂, ξ, y; v̂, σ)`:

```math
-m\Big[\underbrace{\tfrac{p̂}{\hat L_q}\Theta(y_c-y)}_{\text{streaming}}
        + \hat\rho_{\theta i}\,\hat\omega_D
        - \tfrac{\hat\rho_{\theta i}}{2}\big\langle\tfrac1{\hat v_\parallel}\partial_x\hat\Phi\big\rangle_\theta\Big]
   \partial_\xi \bar G_0
+ m\Big[\underbrace{\tfrac{\hat w^2}{4\hat L_q}\sin\xi\,\Theta(y_c-y)}_{\text{streaming}}
        - \tfrac{\hat\rho_{\theta i}}{2}\big\langle\tfrac1{\hat v_\parallel}\partial_\xi\hat\Phi\big\rangle_\theta\Big]
   \partial_{p̂} \bar G_0
= \big\langle\tfrac1{\hat v_\parallel}\hat C_{ii}(\bar G_0)\big\rangle_\theta .
```

\noindent
The **island-streaming** channel is the two braced terms — the equilibrium
magnetic shear (`p̂/L̂_q`) driving the `∂_ξ` transit and the island radial field
(`ŵ²/4L̂_q · sinξ`, from `A_∥ = −(ψ̃/R)cosξ` → `B̃_r ∝ sinξ`) driving the `∂_{p̂}`
advection. `Θ(y_c-y)` restricts it to **passing** particles (`y < y_c = 1`):
trapped particles bounce and carry no net parallel transit, so they do not stream
along the island (I19 Eq. 32; L23 §2.6). The other braced terms are the drift
(`ω̂_D`, cleared) and the `E×B` channels (gated).

To leading order in the ``\Delta = w/r`` ordering the canonical momentum equals
the radial coordinate, ``p̂ = x - I\hat v_\parallel/\omega_c + \dots \to x`` (the
orbit-width correction is `O(ρ̂_θi)`), so `∂_{p̂} → ∂_x` and `p̂ → x` in the
operator, whose solve coordinate is `x` (docs/03 §2, Decision D1).

## 2. Normalization — matched to the cleared ``\hat\omega_D``

The operator residual sums `a_xi ∂_ξ g + c_D ∂_ξ g + …` and `a_x ∂_x g + …`, so
every coefficient must share one normalization. `c_D` is **already cleared** as
`c_D = ω̂_D` (`magnetic_drift_frequency`), which pins the normalization: divide
the whole master equation by `−m ρ̂_θi`. The drift term then reads
`(−m ρ̂_θi ω̂_D)/(−m ρ̂_θi) = ω̂_D = c_D` ✅ (unchanged), and the streaming terms
become

```math
a_\xi = \frac{-m\,(x/\hat L_q)\,\Theta}{-m\,\hat\rho_{\theta i}}
      = \frac{\hat L_q^{-1}\,x}{\hat\rho_{\theta i}}\,\Theta(y_c-y),
\qquad
a_x = \frac{+m\,(\hat w^2/4\hat L_q)\sin\xi\,\Theta}{-m\,\hat\rho_{\theta i}}
    = -\frac{\hat L_q^{-1}\,\hat w^2\,\sin\xi}{4\,\hat\rho_{\theta i}}\,\Theta(y_c-y) .
```

\noindent
`m` cancels; `ρ̂_θi` (normalized ion poloidal gyroradius, `~ ŵ` by the O2
ordering) survives as a genuine parameter because fixing `c_D = ω̂_D` referred the
whole equation to the drift scale. `ŵ = w_ψ` is the island half-width in `Ω`
(same `w` as `Ω = 2x²/ŵ² − cosξ` and the `ĥ` amplitude, `electron-closure.md §3`).

```math
\boxed{\;
a_\xi = \frac{\hat L_q^{-1}}{\hat\rho_{\theta i}}\,x\,\Theta(y_c-y),
\qquad
a_x = -\frac{\hat L_q^{-1}\hat w^2}{4\,\hat\rho_{\theta i}}\,\sin\xi\,\Theta(y_c-y)
\;}
```

## 3. The consistency check — streaming *is* advection along `Ω`

Parallel streaming must advect `g` along the island flux surfaces `Ω = const`;
that advection is the Poisson bracket `\{\Omega, g\} = \partial_x\Omega\,\partial_\xi g - \partial_\xi\Omega\,\partial_x g`. With `Ω = 2x²/ŵ² − cosξ`,
`∂_xΩ = 4x/ŵ²` and `∂_ξΩ = sinξ`, so

```math
\{\Omega,g\} = \frac{4x}{\hat w^2}\,\partial_\xi g - \sin\xi\,\partial_x g .
```

The derived coefficients factor **exactly** into this bracket:

```math
a_\xi\,\partial_\xi g + a_x\,\partial_x g
 = \frac{\hat L_q^{-1}\hat w^2}{4\,\hat\rho_{\theta i}}\,\Theta(y_c-y)
   \Big[\frac{4x}{\hat w^2}\partial_\xi g - \sin\xi\,\partial_x g\Big]
 = \frac{\hat L_q^{-1}\hat w^2}{4\,\hat\rho_{\theta i}}\,\Theta(y_c-y)\,\{\Omega,g\} .
```

\noindent
So the island-streaming operator is `(L̂_q⁻¹ ŵ²/4ρ̂_θi)Θ · {Ω, ·}` — pure
flux-surface advection, vanishing on `Ω`-contours (`{Ω,Ω}=0`) exactly as parallel
streaming must. This is a coefficient-free structural check that pins the relative
sign and magnitude of `a_ξ` and `a_x` with no freedom — the derivation's own
proof.

## 4. Cross-check table

| Source / check | Statement | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (32) streaming braces (docs/01 §2) | `(x/L̂_q)Θ ∂_ξ`, `(ŵ²/4L̂_q)sinξ Θ ∂_{p̂}` | ✅ structure (§1) |
| cleared `c_D = ω̂_D` normalization | divide by `−m ρ̂_θi` ⇒ `c_D` unchanged | ✅ (§2) |
| Poisson-bracket advection `{Ω, g}` | streaming `∝ {Ω, ·}`, `=0` on `Ω`-contours | ✅ **exact** (§3) |
| passing/trapped split `Θ(y_c−y)` | trapped carry no net transit | ✅ (§1) |

**Triage:** no discrepancy. The `{Ω, g}` factorization (§3) is a hard internal
check that leaves no coefficient freedom; the only supplied scale is the physical
parameter `ρ̂_θi`.

## 5. What sign-off authorizes

On sign-off (recorded in docs/01 §2): `Configure` may build, on the phase-space
grid, `a_ξ = (inv_Lq/ρ̂_θi)·x·Θ(y_c−y)` and
`a_x = −(inv_Lq·ŵ²/4ρ̂_θi)·sinξ·Θ(y_c−y)` (with `ŵ = w_psi`, `ρ̂_θi` a new
`Level0Physics` field, `Θ` from `y < 1`) and wire them into
`Operators.ParallelStreaming`, replacing the gated `a_xi`/`a_x`. `c_D` is
unchanged (the normalization was chosen to keep it `= ω̂_D`). This un-gates the
`:streaming` family (QUESTIONS Q5). The `E×B`, gradient drive, collision
magnitude, pitch measure, and far field remain gated.
