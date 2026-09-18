"""Symbolic verification of the r_s-referenced Connor et al. 2015 Eq. 59 toroidal critical-Δ factor.

Checks, with SymPy (run: uv run --with sympy python3 verify_toroidal_delta_crit_symbolic.py):
  1. Λ ≡ ψ'χ'' − χ'ψ'' = ψ'² (ι/2π)'            (Connor's two definitions agree)
  2. the GPEC expressions for ψ_t', (ι/2π)', Λ, α in terms of (chi1, v1, q, q1) are the
     chain-rule images of the V-derivative definitions
  3. the code formula k_ref·v1·(α²Λ²/(⟨B²⟩ v1²⟨|∇ψ_N|²⟩))^{1/4} equals
     V_s (α²Λ²/(⟨B²⟩⟨|∇V|²⟩))^{1/4} · r_s(dV/dr)/V_s   (V_s cancels)
  4. circular large-aspect-ratio limit: Eq. 59's geometric factor → ½√(n s r/R), and the
     r_s-referenced factor → √(n s r/R), for an arbitrary q(r)
  5. prefactor chain: Eq. 61 · r_s ≡ the rfitzp code formula −√2π^{3/2}D_R/W_d with Connor Eq. 65
  6. scaling: the r_s-referenced factor is invariant under B → λB and lengths → μ·lengths; the
     Fortran STRIDE form (one power of ψ_t' in Λ) scales as (λ/μ)^{-1/2}
"""
import sympy as sp

ok = True
def check(name, expr):
    global ok
    res = sp.simplify(expr)
    passed = res == 0
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"   residual = {res}"))

# ---------------------------------------------------------------- 1. Λ identity
V = sp.symbols('V', positive=True)
psi = sp.Function('psi')(V)      # toroidal flux ψ(V)
chi = sp.Function('chi')(V)      # poloidal flux χ(V)
iota2pi = sp.diff(chi, V) / sp.diff(psi, V)
Lam_def = sp.diff(psi, V) * sp.diff(chi, V, 2) - sp.diff(chi, V) * sp.diff(psi, V, 2)
check("1. psi'chi'' - chi'psi''  ==  psi'^2 (iota/2pi)'", Lam_def - sp.diff(psi, V)**2 * sp.diff(iota2pi, V))

# ---------------------------------------------------------------- 2. GPEC chain rule
x = sp.symbols('psi_N', positive=True)          # normalized poloidal flux
chi1, n = sp.symbols('chi1 n', positive=True)   # chi1 = dχ/dψ_N = 2π psio ; toroidal mode number
psit = sp.Function('psi_t')(x)                  # toroidal flux ψ_t(ψ_N)
Vf = sp.Function('V')(x)                        # volume V(ψ_N)
v1 = sp.diff(Vf, x)                             # dV/dψ_N
q = sp.diff(psit, x) / chi1                     # q = dψ_t/dχ
q1 = sp.diff(q, x)                              # dq/dψ_N
# V-derivatives via d/dV = (1/v1) d/dψ_N
psit_V = sp.diff(psit, x) / v1
iota2pi_V = sp.diff(1 / q, x) / v1
Lam_V = psit_V**2 * iota2pi_V
alpha_V = 2 * sp.pi * n / (chi1 / v1)           # α = 2πn/χ'
# code expressions (LayerInputs.jl toroidal_dgeo)
psit1_code = q * chi1 / v1
Lam_code = psit1_code**2 * (-q1 / (q**2 * v1))
alpha_code = 2 * sp.pi * n * v1 / chi1
check("2a. psi_t' code == chain rule", psit1_code - psit_V)
check("2b. Lambda code == psi_t'^2 (iota/2pi)'", Lam_code - Lam_V)
check("2c. alpha code == 2 pi n / chi'", alpha_code - alpha_V)

# ---------------------------------------------------------------- 3. reference conversion, V_s cancels
Vs, rs, dadpsi, B2, G, v1s, a2, L2 = sp.symbols('V_s r_s da_dpsi B2 G v1 alpha2 Lambda2', positive=True)
eq59_geo = Vs * (a2 * L2 / (B2 * (v1s**2 * G)))**sp.Rational(1, 4)   # ⟨|∇V|²⟩ = v1²⟨|∇ψ_N|²⟩
dVdr = v1s / dadpsi                                                    # dV/dr = (dV/dψ_N)/(da/dψ_N)
conversion = rs * dVdr / Vs                                            # Y=(V−V_s)/V_s → x̂=(r−r_s)/r_s
k_ref = rs / dadpsi
dgeo_code = k_ref * v1s * (a2 * L2 / (B2 * v1s**2 * G))**sp.Rational(1, 4)
check("3. code dgeo == Eq.59 factor x r_s (dV/dr)/V_s  (V_s cancels)", dgeo_code - eq59_geo * conversion)

# ---------------------------------------------------------------- 4. circular LAR limit, arbitrary q(r)
r, R, B = sp.symbols('r R B', positive=True)
qr = sp.Function('q')(r)
V_r = 2 * sp.pi**2 * R * r**2                    # volume
psit_r = sp.pi * r**2 * B                        # toroidal flux (uniform B)
dchi_dr = 2 * sp.pi * r * B / qr                 # dχ/dr = dψ_t/dr / q  (q = dψ_t/dχ)
dVdr_r = sp.diff(V_r, r)
d_dV = lambda f: sp.diff(f, r) / dVdr_r
psit_V_r = d_dV(psit_r)
chi_V_r = dchi_dr / dVdr_r
Lam_r = psit_V_r**2 * d_dV(chi_V_r / psit_V_r)   # ψ'^2 (χ'/ψ')' = ψ'^2 (ι/2π)'
alpha_r = 2 * sp.pi * n / chi_V_r
B2_r = B**2                                      # ⟨B²⟩ at leading order
gradV2_r = dVdr_r**2                             # ⟨|∇V|²⟩ = (dV/dr)²
s_r = r * sp.diff(qr, r) / qr                    # r-based shear
eq59_r = V_r * (alpha_r**2 * Lam_r**2 / (B2_r * gradV2_r))**sp.Rational(1, 4)
lar_half = sp.Rational(1, 2) * sp.sqrt(n * s_r * r / R)
# q' may be negative: compare squares of both sides (both sides positive for s>0) and the sign
check("4a. Eq.59 geometric factor (Y ref) -> 1/2 sqrt(n s r/R)", sp.powsimp(eq59_r**4 - lar_half**4, force=True))
dgeo_r = eq59_r * r * dVdr_r / V_r
check("4b. r_s-referenced factor -> sqrt(n s r/R)", sp.powsimp(dgeo_r**4 - (n * s_r * r / R)**2, force=True))
check("4c. conversion r_s (dV/dr)/V_s == 2 on a circle", r * dVdr_r / V_r - 2)

# ---------------------------------------------------------------- 5. prefactor chain: Eq.61 · r_s == rfitzp
chipar, chiperp, DR, s = sp.symbols('chi_par chi_perp D_R s', positive=True)
eq61 = sp.pi**sp.Rational(3, 2) / 2 * (chipar / chiperp)**sp.Rational(1, 4) * sp.sqrt(n * s / (R * r)) * (-DR)
Wd = sp.sqrt(8) * (chiperp / chipar)**sp.Rational(1, 4) / sp.sqrt(r / R * s * n)   # Connor Eq. 65, W_d/r_s
rfitzp = -sp.sqrt(2) * sp.pi**sp.Rational(3, 2) * DR / Wd                            # LayerParameters.jl :rfitzp
check("5. Eq.61 * r_s == rfitzp code formula", eq61 * r - rfitzp)
# and the toroidal branch with dgeo -> sqrt(n s r/R) equals rfitzp
toroidal_lar = sp.Rational(1, 2) * (-DR) * sp.pi**sp.Rational(3, 2) * (chipar / chiperp)**sp.Rational(1, 4) * sp.sqrt(n * s * r / R)
check("5b. toroidal branch at LAR == rfitzp", toroidal_lar - rfitzp)

# ---------------------------------------------------------------- 6. scaling
# q is a shape function of r/R, so it is invariant under a uniform length scaling.
lam, mu = sp.symbols('lambda mu', positive=True)
qshape = sp.Function('q')(r / R)
dgeo_shape = dgeo_r.subs(qr, qshape).doit()
fortran_shape = (dgeo_r / sp.sqrt(psit_V_r)).subs(qr, qshape).doit()
scale = {B: lam * B, r: mu * r, R: mu * R}
ratio_code = sp.simplify((dgeo_shape.subs(scale, simultaneous=True) / dgeo_shape)**4)
ratio_fortran = sp.simplify((fortran_shape.subs(scale, simultaneous=True) / fortran_shape)**2)
check("6a. r_s-referenced factor invariant under B->lambda B, lengths->mu lengths", ratio_code - 1)
check("6b. Fortran form (psi_t'^1 in Lambda) scales as (lambda/mu)^(-1/2)", ratio_fortran - mu / lam)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
