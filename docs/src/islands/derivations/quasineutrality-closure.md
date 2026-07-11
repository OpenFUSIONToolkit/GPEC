# Derivation — the Level-0 quasineutrality closure

**Provenance:** `[DERIVED: 2026-07-11]` — independent re-derivation (Decision D7).
**Clears (on sign-off):** the Level-0 quasineutrality relation and its closure
coefficient ``1/(2\hat L_{n0})`` (`[CHECKED: I19 Eq. A.11; L23 Eq. 2.4.14;
Picard form Diss19 Eq. 2.45]`, QUESTIONS Q3), plus the arbitrary-``\tau``
generalization docs/01 §3 asks for.

**Status:** awaiting human sign-off. Until signed off, `Operators.Quasineutrality`'s
closure coefficient `α` stays a supplied, gated argument.

## 1. Setup

At Level 0 the only field equation is quasineutrality, ``n_i[\Phi;g_i] =
n_e[\Phi;\text{closure}]`` (docs/01 §3; Ampère is a diagnostic, O3). The two
species' densities are the velocity moments of their responses (I19 Eqs. 17, 23,
first-hand). This derivation takes those responses and solves ``n_i=n_e`` for
``\Phi``, deriving the closure coefficient. Normalizations (docs/01 §5):
``x=(\psi-\psi_s)/\psi_s``, ``\hat L_{n0}^{-1}=(\psi_s/n_0)\,dn_0/d\psi``,
``\hat\Phi=e_i\Phi/T_i``, ``\tau=T_e/T_i``, ``\hat h=h/\psi_s`` (so ``\hat h\to x``
far away, electron-closure derivation §3).

## 2. The species densities

**Ions** (I19 Eq. 23): ``f_i=(1-\tfrac{e_i\Phi}{T_i})F_{Mis}+(\psi-\psi_s)F'_{Mis}
+g_i``. The velocity moment (``e_i=+e``, ``Z_i=1``):

```math
n_i = n_0\Big(1-\frac{e\Phi}{T_i}\Big) + n_0'\,(\psi-\psi_s) + \delta\bar n_i,
\qquad \delta\bar n_i=\int g_i\,d^3v ,
```

using ``\int F_{Mis}d^3v=n_0`` and ``\int F'_{Mis}d^3v=n_0'=dn_0/d\psi``.

**Electrons** (flattened closure, electron-closure derivation; I19 Eq. 17):
``f_e=(1-\tfrac{e_e\Phi}{T_e})F_{Mes}+h(\Omega)F'_{Mes}-\tfrac{Iv_\parallel}{\omega_{ce}}
F'_{Mes}\partial_\psi h+\bar h_e``. The ``v_\parallel``-odd term vanishes in the
density moment; with ``e_e=-e`` and the leading closure (``\bar h_e`` higher
order),

```math
n_e = n_0\Big(1+\frac{e\Phi}{T_e}\Big) + n_0'\,h(\Omega) .
```

The ``n_0' h(\Omega)`` term **is** the electron density perturbation carried by
the flattening — the flattened electrons pile up/deplete along ``\Omega`` surfaces
exactly as ``h`` prescribes.

## 3. Quasineutrality → the closure

Impose ``n_i=n_e``. The equilibrium ``n_0`` cancels:

```math
-\,n_0 e\Phi\Big(\frac{1}{T_i}+\frac{1}{T_e}\Big)
 = n_0'\big[h(\Omega)-(\psi-\psi_s)\big] - \delta\bar n_i .
```

The ``(1/T_i+1/T_e)`` is the **sum of the ion and electron adiabatic responses**
— both species shield the potential — and is the origin of the closure
denominator. Solving for ``\hat\Phi=e\Phi/T_i`` and using
``T_i(1/T_i+1/T_e)=1+T_i/T_e=(\tau+1)/\tau``:

```math
\boxed{\;
\hat\Phi = \frac{\tau}{\tau+1}\,
   \Big[\, \frac{\delta\bar n_i}{n_0} + \hat L_{n0}^{-1}\big(x-\hat h(\Omega)\big) \,\Big]
\;}
```

using ``n_0'(\psi-\psi_s)/n_0=\hat L_{n0}^{-1}x`` and
``n_0'h/n_0=\hat L_{n0}^{-1}\hat h``.

**At ``\tau=1``** (the sources' ``T_e=T_i``), ``\tau/(\tau+1)=1/2``:

```math
\hat\Phi = \frac{1}{2}\Big[\frac{\delta\bar n_i}{n_0}+\hat L_{n0}^{-1}(x-\hat h)\Big]
 = \frac{1}{2\hat L_{n0}}\Big[\underbrace{\hat L_{n0}\,\frac{\delta\bar n_i}{n_0}}_{\equiv\,\delta n_i/n_0}
   + x - \hat h\Big],
```

**exactly I19 Eq. A.11**, ``\hat\Phi=[\delta n_i/n_0+x-\hat h]/(2\hat L_{n0})``,
provided I19's normalized ion perturbation is
``\delta n_i/n_0\equiv\hat L_{n0}\,\delta\bar n_i/n_0`` (its convention scales the
kinetic density perturbation by the gradient length so the whole bracket shares
the ``1/(2\hat L_{n0})``). This is a **normalization convention, not a
discrepancy** — the physically-invariant statement is the boxed general-``\tau``
form of §3, whose ``x-\hat h`` piece and ``\tau=1`` coefficient match I19 exactly.

## 4. Kinetic-electron (Picard) form

When electrons are solved kinetically (E4 toggle, not the flattened closure), the
same quasineutrality reads ``\delta\hat\Phi=(\delta\hat n_i-\delta\hat n_e)/2``
(Diss19 Eq. 2.45) — the ``\hat h`` term is replaced by the kinetic electron
density perturbation ``\delta\hat n_e``. In Islands both are one residual block
inside the global Newton system (`Operators.Quasineutrality`); the sources'
nested Picard loop is what Newton–Krylov replaces (docs/01 §3).

## 5. Cross-check table

| Source | Form | Agrees with `[DERIVED]`? |
|---|---|---|
| I19 Eq. (A.11) (first-hand, print p. 11) | ``e_i\Phi/T_i=[\delta n_i/n_0+x-\hat h(\Omega)]/(2\hat L_{n0})`` | ✅ (``\tau=1`` limit of §3; ``\delta n_i`` normalization convention noted) |
| L23 Eq. (2.4.14) | same closed form | ✅ |
| Diss19 Eq. (2.45) | Picard form ``\delta\hat\Phi=(\delta\hat n_i-\delta\hat n_e)/2`` | ✅ (§4) |
| docs/01 §3 (τ general) | "keep ``\tau=T_e/T_i`` general and flag departures" | ✅ — the boxed ``\tau/(\tau+1)`` form delivers it |

**Triage:** no discrepancy. The one subtlety (the ``\delta n_i`` normalization)
is a definitional convention, resolved in §3; the derivation additionally
supplies the arbitrary-``\tau`` generalization the sources omit
(they assume ``T_e=T_i``).

## 6. What sign-off authorizes

On sign-off (recorded in docs/01 §3): the closure coefficient
``\tau/(\tau+1)`` (``\to 1/2`` at ``\tau=1``) and the ``\hat L_{n0}^{-1}(x-\hat h)``
structure may populate `Operators.Quasineutrality`'s residual — i.e. the field
residual ``R_\Phi = M[g]-\alpha\hat\Phi`` gets ``\alpha`` and the drive built
from this relation, with ``\tau`` from the parameter vector
(`Frames.Level0Parameters`). The moment machinery (`velocity_moment!`) and the
``\hat h``/``Q`` functions are already implemented; nothing here authorizes the
``\Delta`` prefactors (a separate derivation).
