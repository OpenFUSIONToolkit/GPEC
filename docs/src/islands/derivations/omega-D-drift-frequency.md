# Derivation — the orbit-averaged magnetic drift frequency ``\hat\omega_D`` and the `:original`/`:improved` toggle

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7).
**Clears (on sign-off):** the ``\hat\omega_D`` expression and the
`:original`/`:improved` ``\hat L_B^{-1}`` toggle (`[CHECKED: I19 Eq. 32 def.;
D21 Eqs. 15, B1; D21 Eq. A2, p. 16]`, QUESTIONS Q3). This toggle is the single
highest-impact physics item — it is the ``\hat\omega_D`` term that produces the
``\sim\!\times 6`` threshold-width differential between the two drift models
(the reproducible T2 form of the sources' ``8.73 \to 1.46\,\rho_{bi}`` story;
Decision D9, docs/05).

**Status:** awaiting human sign-off. Until signed off, `MagneticDrift.c_D` stays
a supplied, gated coefficient.

## 1. What ``\hat\omega_D`` is

In the orbit-averaged (4D) Level-0 kinetic equation (docs/01 §2, I19 Eq. 32 —
verified first-hand, print p. 6), the ``\partial\bar G_0/\partial\xi``
coefficient collects three transport channels,

```math
-m\Big[\underbrace{\tfrac{\hat p}{\hat L_q}\,\Theta(y_c-y)}_{\text{streaming}}
      + \underbrace{\hat\rho_{\theta i}\,\hat\omega_D}_{\text{magnetic drift}}
      - \underbrace{\tfrac{\hat\rho_{\theta i}}{2}\big\langle \tfrac{1}{\hat v_\parallel}\tfrac{\partial\hat\Phi}{\partial x}\big\rangle_\theta}_{E\times B}
 \Big]\frac{\partial\bar G_0}{\partial\xi} ,
```

so ``\hat\rho_{\theta i}\,\hat\omega_D`` is the **orbit-averaged magnetic-drift
advection of the distribution in the helical angle ``\xi``**. This derivation
computes it from the guiding-centre magnetic drift (I19 Eq. 8,
``\mathbf v_b = -v_\parallel\,\mathbf b\times\nabla(v_\parallel/\omega_{cj})``)
and the conserved canonical momentum, in the large-aspect-ratio circular
geometry (orderings O1–O5).

Normalizations (docs/01 §5, I19 p. 6, verified first-hand): ``x=(\psi-\psi_s)/\psi_s``,
``y=\lambda B_{\max}``, ``\hat v=v/v_{thi}``, ``b=B/B_{\max}=(1-\varepsilon\cos\theta)/(1+\varepsilon)``,
``\hat L_q^{-1}=(\psi_s/q_s)\,dq/d\psi|_s``, ``\hat L_B^{-1}=(\psi_s/B)\,\partial B/\partial\psi``,
``\hat\rho_{\theta i}=\rho_{\theta i}/r_s``, ``\sigma=\mathrm{sgn}(v_\parallel)``,
``v_\parallel=\sigma v\sqrt{1-\lambda B}=\sigma v\sqrt{1-yb}``. The orbit average
``\langle\cdot\rangle_\theta`` is ``\tfrac{1}{2\pi}\oint d\theta`` (passing) or
``\tfrac{1}{2\pi}\sum_\sigma\int_{-\theta_b}^{\theta_b} d\theta`` (trapped), at
fixed ``p_\phi`` (I19 Eq. 31).

## 2. The drift-orbit radial width (from ``p_\phi`` conservation)

The toroidal canonical momentum ``p_\phi = (\psi-\psi_s) - I v_\parallel/\omega_{cj}``
(I19 Eq. 2) is conserved along orbits, so a particle's flux-surface label
oscillates about its orbit centre by the **drift-orbit width**

```math
x - \hat p = \frac{\psi-\psi_s-p_\phi}{\psi_s}
           = \frac{I\,v_\parallel}{\omega_{cj}\,\psi_s} .
```

Evaluate it in the normalized variables. With ``I=RB_\phi``,
``\omega_{cj}=e_jB/m_j``, ``v_\parallel=\sigma v\sqrt{1-yb}``,
``RB_\phi/B \simeq R_0/b`` (large aspect ratio, ``B=B_{\max}b``,
``RB_\phi=I\simeq R_0 B_{\phi0}``), and ``\psi_s\simeq r_s R_0 B_\theta``
(``d\psi/dr=RB_\theta``), together with the poloidal gyroradius
``\rho_{\theta i}=v_{thi}m_i/(e_iB_\theta)``:

```math
\boxed{\;
x_D(\theta;y,\hat v,\sigma) \equiv x-\hat p
   = \hat\rho_{\theta i}\,\frac{\sigma\hat v}{1+\varepsilon}\,\frac{\sqrt{1-yb}}{b}
\;}
```

(the ``1+\varepsilon`` from ``B_{\max}=B_{\phi0}(1+\varepsilon)``). This is the
finite ion orbit width — the physical heart of the DK-NTM: guiding centres do
not sit on a flux surface but on a surface shifted by ``x_D``, and this shift is
``\sigma``-, ``y``-, and ``\hat v``-dependent (docs/01 §2.2, the "drift island").

## 3. Term 1 — the shear-coupled piece (``1/\hat L_q``)

The equilibrium term ``\langle 1-q/q_s\rangle_\theta`` of the orbit-averaged
equation (I19 Eq. 31) carries the shear. Expanding ``q`` about ``\psi_s``,

```math
1 - \frac{q(\psi)}{q_s} = -\frac{q_s'}{q_s}(\psi-\psi_s) + \mathcal O((\psi-\psi_s)^2)
                       = -\frac{x}{\hat L_q},
```

and orbit-averaging at fixed ``\hat p`` using ``x = \hat p + x_D(\theta)`` (§2):

```math
\big\langle 1-q/q_s\big\rangle_\theta
 = -\frac{\hat p}{\hat L_q}\,\Theta(y_c-y)\;-\;\frac{1}{\hat L_q}\big\langle x_D\big\rangle_\theta .
```

The first piece is parallel streaming (nonzero only for passing particles,
``\Theta(y_c-y)``). The second is a **magnetic-drift** contribution — the shear
acting on the finite orbit width — and with ``x_D`` from §2 it is

```math
-\frac{1}{\hat L_q}\big\langle x_D\big\rangle_\theta
 = -\,\hat\rho_{\theta i}\,\frac{\sigma\hat v}{1+\varepsilon}\,
   \frac{1}{\hat L_q}\Big\langle \frac{\sqrt{1-yb}}{b}\Big\rangle_\theta .
```

## 4. Term 2 — the grad-``B`` piece (``1/\hat L_B``)

The explicit magnetic-drift term of I19 Eq. 31 is
``-\langle I\,\partial(v_\parallel/\omega_{cj})/\partial\psi\rangle_\theta``.
With ``v_\parallel/\omega_{cj} = (\sigma v m_j/e_j)\sqrt{1-\lambda B}/B`` and
``\partial/\partial\psi`` acting on ``B`` (fixed ``\lambda,v``),

```math
\frac{\partial}{\partial\psi}\!\left(\frac{\sqrt{1-\lambda B}}{B}\right)
 = -\,\frac{\partial B}{\partial\psi}\,\frac{2-\lambda B}{2B^2\sqrt{1-\lambda B}} ,
```

since ``\dfrac{d}{dB}\!\dfrac{\sqrt{1-\lambda B}}{B}
 = -\dfrac{2-\lambda B}{2B^2\sqrt{1-\lambda B}}``. Multiplying by ``-I`` and
normalizing (``\lambda B = yb``, ``\hat L_B^{-1}=(\psi_s/B)\,\partial B/\partial\psi``)
gives the second magnetic-drift contribution,

```math
-\Big\langle I\,\frac{\partial}{\partial\psi}\frac{v_\parallel}{\omega_{cj}}\Big\rangle_\theta
 = -\,\frac{\hat\rho_{\theta i}}{2}\,\frac{\sigma\hat v}{1+\varepsilon}
   \Big\langle \frac{1}{\hat L_B}\,\frac{2-yb}{b\sqrt{1-yb}}\Big\rangle_\theta .
```

## 5. Result

Collect the two magnetic-drift contributions. The ``\partial\bar G_0/\partial\xi``
coefficient of §1 carries the equilibrium term with an **overall minus sign**,
``-m\langle 1-q/q_s\rangle_\theta`` (this is what makes the streaming piece
``+\hat p/\hat L_q``, §3); so the §3 second piece enters ``\hat\rho_{\theta i}
\hat\omega_D`` as ``-\big(-\tfrac1{\hat L_q}\langle x_D\rangle\big)
= +\tfrac1{\hat L_q}\langle x_D\rangle``, while the §4 grad-``B`` term (already
written with its sign) adds directly:

```math
\hat\rho_{\theta i}\hat\omega_D
 = +\frac{1}{\hat L_q}\big\langle x_D\big\rangle_\theta
 \;-\;\frac{\hat\rho_{\theta i}}{2}\frac{\sigma\hat v}{1+\varepsilon}
   \Big\langle\frac{1}{\hat L_B}\frac{2-yb}{b\sqrt{1-yb}}\Big\rangle_\theta .
```

Substituting ``\langle x_D\rangle`` from §2 and dividing by
``\hat\rho_{\theta i}``:

```math
\boxed{\;
\hat\omega_D = \frac{\sigma\hat v}{1+\varepsilon}
 \left[\;\frac{1}{\hat L_q}\Big\langle\frac{\sqrt{1-yb}}{b}\Big\rangle_\theta
      \;-\;\frac{1}{2}\Big\langle\frac{1}{\hat L_B}\,\frac{2-yb}{b\sqrt{1-yb}}\Big\rangle_\theta\;\right]
\;}
```

matching I19 Eq. (32) (first-hand) and D21 Eq. (B1) term-for-term (§7). The two
terms are the shear-coupled orbit-width precession (``1/\hat L_q``) and the
grad-``B`` drift (``1/\hat L_B``).

## 6. The `:original` / `:improved` toggle — treatment of ``\partial B/\partial\psi``

The toggle is entirely in ``\hat L_B^{-1}=(\psi_s/B)\,\partial B/\partial\psi``.
In the model circular equilibrium ``R=R_0(1+\varepsilon\cos\theta)``, the radial
field gradient is (D21 App. A, verified first-hand, print p. 16)

```math
\frac{\partial B}{\partial\psi} = \frac{I'}{R} - \frac{I}{R^2}\frac{\partial R}{\partial\psi},
\qquad I' \sim R^2 p'/I \ \Rightarrow\ \frac{I'}{R}\sim \frac{\beta}{r^2}\ (\text{low }\beta,\ \text{drop}),
```

so, keeping the geometric term with ``\partial R/\partial\psi = \cos\theta/(R_0B_\theta)``,

```math
\frac{\partial B}{\partial\psi}
 = -\,\frac{B_\phi}{R_0^2 B_\theta}\cos\theta + \mathcal O(\varepsilon^2/r^2)
\qquad\text{(D21 Eq. A2).}
```

The field gradient is **``\propto\cos\theta``** — odd about the outboard
midplane. This is the crux:

- **`:original`** (I19 / DK-NTM): treat ``\hat L_B^{-1}`` as a *finite constant*
  (the ``\cos\theta`` structure not resolved). The ``1/\hat L_B`` bracket in
  ``\hat\omega_D`` survives orbit-averaging with an ``\mathcal O(1)`` value.
- **`:improved`** (D21 / RDK-NTM): keep the ``\cos\theta``. In the ``1/\hat L_B``
  bracket the remaining factor ``(2-yb)/(b\sqrt{1-yb})`` is even in ``\theta``
  and ``\mathcal O(1)`` over a passing orbit, so
  ``\langle\cos\theta\times(\text{even})\rangle_\theta = \mathcal O(\varepsilon)``:
  the grad-``B`` term is ``\varepsilon``-small after orbit averaging. The
  **documented proxy is ``\hat L_B^{-1}=0``** (D21 footnote; its Fig. 8 compares
  the proxy against the full ``\cos\theta`` form directly).

**Physical consequence (T2 gate, docs/05 D9).** Dropping the ``1/\hat L_B`` term
removes a drift contribution that, in `:original`, partially cancels the shear
term over the trapped/barely-passing population; its removal changes the
drift-island structure and lowers the threshold half-width by a factor of order
several. Islands measures this **within the code** as the `:original → :improved`
``w_c`` ratio (the reproducible form of the sources' ``8.73 \to 1.46\,\rho_{bi}``
absolute pair, which is T4/audit-gated). The absolute numbers are *not* the gate.

## 7. Cross-check table

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (32) (first-hand, print p. 6) | ``\hat\omega_D=\tfrac{\sigma\hat v}{1+\varepsilon}[\tfrac1{\hat L_q}\langle\tfrac{\sqrt{1-yb}}{b}\rangle-\tfrac12\langle\tfrac1{\hat L_B}\tfrac{2-yb}{b\sqrt{1-yb}}\rangle]`` | ✅ exact |
| D21 Eq. (B1) (first-hand, print p. 16) | ``\hat\omega_D=-\tfrac{\hat w}{\hat L_q}\langle\hat V_\parallel\rangle+\langle\tfrac{B^2}{B_\phi^2}\tfrac{\hat w}{\hat L_B}[\hat V_\parallel+\tfrac{\lambda\hat V^2}{2\hat V_\parallel}]B\rangle`` | ✅ (D21 ``w``-normalization; ``\hat V_\parallel+\tfrac{\lambda\hat V^2}{2\hat V_\parallel}=\hat V\tfrac{2-\lambda B}{2\sqrt{1-\lambda B}}`` = the same ``(2-yb)/\sqrt{1-yb}`` structure) |
| D21 Eq. (A2) (first-hand, print p. 16) | ``\partial B/\partial\psi=-(B_\phi/R_0^2B_\theta)\cos\theta+\mathcal O(\varepsilon^2)`` | ✅ (basis of the `:improved` proxy, §6) |
| Diss19 §2 / docs/01 §2.1 | same two-term ``\hat\omega_D``; `:original`/`:improved` toggle | ✅ |

**Triage:** all first-hand sources agree; no discrepancy (unlike the ψ̃ typo).
The one *modelling choice* is the `:improved` ``\hat L_B^{-1}=0`` **proxy** —
D21 documents it as a proxy for the ``\cos\theta`` form (accurate to
``\mathcal O(\varepsilon)``), not an identity. Islands should therefore carry
**both**: `:original` (finite ``\hat L_B^{-1}``) and `:improved`
(``\hat L_B^{-1}=0``), exactly as the `MagneticDrift.variant` field already
provides — the toggle *is* the deliverable.

## 8. What sign-off authorizes

On human sign-off (recorded in docs/01 §2.1), the coefficient array
`MagneticDrift.c_D` may be constructed from the boxed ``\hat\omega_D`` of §5:
``c_D = -m\,\hat\rho_{\theta i}\,\hat\omega_D`` evaluated on the ``(y,E,\sigma)``
grid, with the ``\hat L_B^{-1}`` treatment selected by
`MagneticDrift.variant` (`:original` → finite ``\hat L_B^{-1}``; `:improved`
→ ``\hat L_B^{-1}=0``). The orbit averages ``\langle\cdot\rangle_\theta`` are
evaluated with the existing quadrature machinery; the ``\hat L_q``, ``\hat L_B``,
``\varepsilon`` inputs come from the parameter vector (`Frames.Level0Parameters`
+ the equilibrium). Nothing here authorizes the collision, closure, or E×B
coefficients.
