# Derivation — the flattened-electron closure ``h(\Omega)`` and its amplitude

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7).
**Clears (on sign-off):** the flattened-electron profile function ``h(\Omega)``
and its amplitude ``w_\psi/2\sqrt2`` (`[CHECKED: I19 Eqs. 14–22; WCHH96 §; L23
§2.4]`, QUESTIONS Q3), and the closure ODE that ties to the already-green A7
identity.
**Deferred (flagged, §6):** the Hirshman–Sigmar flow coefficient ``k\simeq
-1.173`` and the passing-fraction constant in ``f_p\simeq 1-1.46\sqrt\varepsilon``
— specific neoclassical constants, each its own short derivation.

**Status:** ✅ **signed off 2026-07-11** (clearance recorded in docs/01 §2.4) for
the ``h(\Omega)`` form and amplitude — implemented as `Coefficients.h_amplitude`
(``C=w_\psi/2\sqrt2``), feeding `Fields.h_profile`'s prefactor. The flow constants
``k`` and ``f_p`` (§6) remain **deferred / NaN-gated**.

## 1. Setup

At Level 0 electrons have ``\rho_{\theta e}\ll w`` (O7), so their drift islands
coincide with the magnetic island and their response is the analytic
flattened-electron (WCHH96) closure (docs/01 §2.4). The electron distribution is
(I19 Eq. 17, first-hand)

```math
f_e = \Big(1-\frac{e\Phi}{T_e}\Big)F_{Mes} + h(\Omega)\,F'_{Mes}
     - \frac{Iv_\parallel}{\omega_{ce}}F'_{Mes}\frac{\partial h}{\partial\psi}
     + \bar h_e ,
```

where ``h(\Omega)`` is the perturbed profile function — **constant on island
flux surfaces ``\Omega``** (the flattening), exactly flat inside the separatrix
(``\Omega<1``) and ``\to x`` far outside. This derivation determines ``h`` and
its amplitude. Convention (module CLAUDE.md): ``\Omega=2x^2/w^2-\cos\xi`` with
``w=w_\psi``, and the flux-surface average
``\langle f\rangle_\Omega=\oint f\,(\Omega+\cos\xi)^{-1/2}d\xi\,/\oint(\Omega+
\cos\xi)^{-1/2}d\xi`` (I19 Eq. 21).

## 2. The closure constraint determines ``h(\Omega)``

Flattening on ``\Omega`` surfaces with quasineutrality requires the
flux-surface-averaged radial curvature of ``h`` to vanish (I19; the unit target
``\langle\partial^2 h/\partial x^2\rangle_\Omega=0``, docs/01 §6, ladder A7).
Compute it for ``h=h(\Omega)``. With ``\partial\Omega/\partial x=4x/w^2`` and
``x^2=\tfrac{w^2}{2}(\Omega+\cos\xi)``,

```math
\frac{\partial^2 h}{\partial x^2}
 = h''(\Omega)\Big(\frac{4x}{w^2}\Big)^2 + h'(\Omega)\frac{4}{w^2}
 = \frac{8}{w^2}h''(\Omega)(\Omega+\cos\xi) + \frac{4}{w^2}h'(\Omega).
```

Flux-averaging and setting to zero,

```math
2\,h''(\Omega)\,\langle\Omega+\cos\xi\rangle_\Omega + h'(\Omega) = 0. \tag{$\star$}
```

Now relate ``\langle\Omega+\cos\xi\rangle_\Omega`` to the geometry function
``Q(\Omega)=\tfrac1{2\pi}\oint\sqrt{\Omega+\cos\xi}\,d\xi`` (I19 Eq. 18), whose
derivative is ``Q'(\Omega)=\tfrac1{4\pi}\oint(\Omega+\cos\xi)^{-1/2}d\xi``:

```math
\langle\Omega+\cos\xi\rangle_\Omega
 = \frac{\oint(\Omega+\cos\xi)^{1/2}d\xi}{\oint(\Omega+\cos\xi)^{-1/2}d\xi}
 = \frac{2\pi Q}{4\pi Q'} = \frac{Q}{2Q'} .
```

Substituting into ``(\star)``: ``2h''\,\tfrac{Q}{2Q'}+h'=0``, i.e.
``\dfrac{h''}{h'}=-\dfrac{Q'}{Q}``, which integrates to

```math
\boxed{\; h'(\Omega)=\frac{C}{Q(\Omega)},\qquad
 h(\Omega)=\Theta(\Omega-1)\,C\!\int_1^\Omega\frac{d\Omega'}{Q(\Omega')} \;}
```

flat inside the separatrix (the ``\Theta(\Omega-1)``, since there is no
flattening gradient on the closed field lines within). This is I19 Eq. 18 up to
the amplitude ``C``. **The A7 identity ``\langle\partial^2h/\partial x^2\rangle_
\Omega=0`` is exactly ``(\star)`` with ``h'=C/Q``** — the already-green A7 gate
*is* this closure constraint, verified for any ``C`` (`Fields.flat_average_d2h_dx2`).

## 3. The amplitude ``C=w_\psi/2\sqrt2`` from far-field matching

Far from the island (``\Omega\to\infty``) the flattened profile must match the
unperturbed radial coordinate, ``h\to x``. Large-``\Omega`` asymptotics:
``Q(\Omega)=\tfrac1{2\pi}\oint\sqrt{\Omega+\cos\xi}\,d\xi\to\sqrt\Omega``, so

```math
h(\Omega)\to C\int^\Omega\frac{d\Omega'}{\sqrt{\Omega'}} = 2C\sqrt\Omega .
```

Meanwhile ``x=\tfrac{w}{\sqrt2}\sqrt{\Omega+\cos\xi}\to\tfrac{w}{\sqrt2}\sqrt\Omega``
(from ``x^2=\tfrac{w^2}{2}(\Omega+\cos\xi)``). Matching ``h\to x``:

```math
2C\sqrt\Omega = \frac{w}{\sqrt2}\sqrt\Omega
\quad\Longrightarrow\quad
\boxed{\; C = \frac{w}{2\sqrt2} = \frac{w_\psi}{2\sqrt2} \;}
```

exactly the I19 Eq. 18 amplitude. So both the *form* and the *amplitude* of
``h(\Omega)`` are fixed — the former by the flattening constraint, the latter by
far-field matching.

## 4. Cross-check table (the derived parts)

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (18) (first-hand, print p. 4) | ``h=\Theta(\Omega-1)\tfrac{w_\psi}{2\sqrt2}\int_1^\Omega d\Omega'/Q``, ``Q=\tfrac1{2\pi}\oint\sqrt{\Omega+\cos\xi}d\xi`` | ✅ form (§2) + amplitude ``w_\psi/2\sqrt2`` (§3) |
| I19 §6 / L23 Eq. 4.1.1 | ``\langle\partial^2h/\partial x^2\rangle_\Omega=0`` | ✅ = the closure constraint ``(\star)`` (§2); already the green A7 gate |
| I19 Eq. (17) | ``f_e`` structure with ``h(\Omega)``, ``-Iv_\parallel/\omega_{ce}F'_{Mes}\partial h/\partial\psi`` | ✅ structure |

**Triage:** the ``h(\Omega)`` form and amplitude agree first-hand; no discrepancy.

## 5. The flux-surface-averaged electron flow (structure)

The electron parallel flow that carries the bootstrap current is (I19 Eq. 22,
first-hand, print p. 5)

```math
\frac{\langle\langle Bu_{\parallel e}\rangle_\theta\rangle_\Omega}{B_0 v_{the}}
 = -\frac{f_t}{1+f_t}\frac{Iv_{the}}{\omega_{ce}}\frac{n'}{n}
   \Big(1+\eta_e+\tfrac12 k f_c\eta_e\Big)\Big\langle\frac{\partial h}{\partial\psi}\Big\rangle_\Omega
 + \frac{f_c}{1+f_t}
   \frac{\langle\langle Bu_{\parallel i}\rangle_\theta\rangle_\Omega}{B_0 v_{thi}} ,
```

with ``f_t`` the trapped fraction, ``f_c=1-f_t`` passing, ``\eta_e=L_n/L_{Te}``,
and ``k`` the Hirshman–Sigmar coefficient. The **structure** is derived here (it
follows from the parallel momentum balance with the pitch-angle operator §2 of
the collision derivation, and the ``h``-gradient drive of §2–3); the two
**numerical constants** are §6. Note the flow depends on the *numerically
computed ion flow* ``u_{\parallel i}`` — the closure is coupled, not one-way
(the `electrons = :flattened` vs `:kinetic` toggle, E4, is a separate study).

## 6. Deferred constants (flagged, not asserted)

- **``k\simeq -1.173`` (Hirshman–Sigmar).** A specific parallel-viscosity /
  flow coefficient obtained by solving the Spitzer-problem moment hierarchy with
  the pitch-angle operator; a self-contained constant, not derived here (policy
  rule 4). L23 reproduces ``-1.1730`` as a unit test.
- **``f_p\simeq 1-1.46\sqrt\varepsilon`` (passing fraction).** The trapped
  fraction ``f_t=1.46\sqrt\varepsilon`` at low ``\varepsilon`` follows from the
  pitch-angle integral ``f_t=1-\tfrac34\langle B^2\rangle\int_0^{1/B_{\max}}
  \lambda\,d\lambda/\langle\sqrt{1-\lambda B}\rangle_\theta`` in the
  large-aspect-ratio limit; the specific ``1.46`` is its own short reduction and
  is left deferred rather than asserted.

## 7. What sign-off authorizes

On sign-off (recorded in docs/01 §2.4): the ``h(\Omega)`` amplitude
``C=w_\psi/2\sqrt2`` clears `Fields.h_profile`'s `prefactor` (and hence
`ElectronClosure.h_prefactor`), with the `Q(\Omega)`/`h(\Omega)` machinery and
the A7 identity already implemented. The flow-relation *structure* (§5) is
cleared; its constants ``k`` and ``f_p`` (`ElectronClosure.k_HS`, `.f_p`) stay
NaN-gated pending their own derivations (§6). The quasineutrality closure
coefficient (``C_\phi``) is a separate derivation.
