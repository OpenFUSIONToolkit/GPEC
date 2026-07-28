#!/usr/bin/env python3
# cylindrical_newcomb_deltaprime.py -- independent large-aspect-ratio (cylindrical) tearing Delta-prime
# reference for the analytic TJ / LAR circular equilibrium, used to benchmark GPEC's toroidal Delta-prime
# in the single-rational-surface cases.
#
# Model (reduced MHD, straight periodic cylinder, circular, large aspect):
#   analytic TJ q-profile   q(x) = x^2 / f1(x),   f1(x) = [1 - (1-x^2)^nu] / (nu*qc),   x = r/a, nu = qa/qc
#   Newcomb outer equation  d/dx( x psi' ) - (m^2/x) psi - [ m*q*H'(x) / (m - n*q) ] psi = 0
#                           with H(x) = 2/q - x*q'/q^2  (current-gradient term; singular at q=m/n).
# Delta-prime is the jump in the logarithmic derivative across the rational surface x_s (q(x_s)=m/n):
#   a*Delta' = [psi'/psi]_{x_s+delta} - [psi'/psi]_{x_s-delta}   (the log-divergent parts cancel; extrapolate delta->0)
# We report the dimensionless r_s*Delta' (standard tearing index).
#
# Self-test: with the current term OFF the solution is vacuum x^{+/-m}, giving r_s*Delta' = -2m exactly.
# Boundary conditions: regular at axis (psi ~ x^m); NO WALL at the edge (vacuum psi ~ x^{-m}, i.e. psi'/psi = -m at x=1).
#
# Pure numpy (no scipy). Usage: python3 cylindrical_newcomb_deltaprime.py   (runs self-test + the 5 TJ cases)

import numpy as np

def _qparts(x, qc, nu):
    # analytic q(x) and q'(x) for q(x)=x^2/f1, f1=[1-(1-x^2)^nu]/(nu*qc); cheap (2 power ops)
    if x < 1e-8:
        return qc, 0.0
    u = 1.0 - x*x
    un = u**nu
    g = 1.0 - un
    q = nu*qc*x*x/g
    gp = 2.0*nu*x*u**(nu - 1.0)          # dg/dx
    qp = nu*qc*(2.0*x/g - x*x*gp/(g*g))  # dq/dx
    return q, qp

def q_of_x(x, qc, nu):
    return _qparts(float(x), qc, nu)[0]

def Hfun(x, qc, nu):
    q, qp = _qparts(x, qc, nu)
    return 2.0/q - x*qp/(q*q)

def Hprime(x, qc, nu, h=1e-6):
    xm = max(x - h, 1e-9)
    return (Hfun(x + h, qc, nu) - Hfun(xm, qc, nu)) / (x + h - xm)

def rhs(x, y, m, n, qc, nu, current=True):
    y1, y2 = y
    dy1 = y2 / x
    cur = 0.0
    if current:
        q, _ = _qparts(x, qc, nu)
        cur = m*q*Hprime(x, qc, nu) / (m - n*q)
    dy2 = (m*m/x + cur) * y1
    return np.array([dy1, dy2])

def rk4(f, x0, x1, y0, N):
    h = (x1 - x0) / N
    y = np.array(y0, float); x = x0
    for _ in range(N):
        k1 = f(x, y); k2 = f(x + 0.5*h, y + 0.5*h*k1)
        k3 = f(x + 0.5*h, y + 0.5*h*k2); k4 = f(x + h, y + h*k3)
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4); x += h
    return y

def find_rs(m, n, qc, nu):
    tgt = m / n
    a, b = 1e-4, 1.0 - 1e-9
    fa = q_of_x(a, qc, nu) - tgt; fb = q_of_x(b, qc, nu) - tgt
    if fa*fb > 0:
        return None
    for _ in range(200):
        mid = 0.5*(a + b); fm = q_of_x(mid, qc, nu) - tgt
        if abs(fm) < 1e-12 or (b - a) < 1e-13:
            return mid
        if fa*fm <= 0: b, fb = mid, fm
        else: a, fa = mid, fm
    return 0.5*(a + b)

def deltaprime(m, n, qc, nu, delta=1e-3, N=400000, current=True):
    rs = find_rs(m, n, qc, nu)
    if rs is None:
        return None, None
    f = lambda x, y: rhs(x, y, m, n, qc, nu, current)
    x0 = 1e-4
    yin = rk4(f, x0, rs - delta, [x0**m, m*x0**m], N)          # axis -> just inside surface
    ld_in = (yin[1]/(rs - delta)) / yin[0]
    yout = rk4(f, 1.0, rs + delta, [1.0, -float(m)], N)        # no-wall edge -> just outside surface
    ld_out = (yout[1]/(rs + delta)) / yout[0]
    aDp = ld_out - ld_in                                       # a*Delta' (a=1)
    return rs*aDp, rs                                          # return r_s*Delta' (dimensionless)

def deltaprime_extrap(m, n, qc, nu, deltas=(1e-2, 3e-3, 1e-3, 3e-4), N=60000):
    # Delta'(delta) carries a residual ~ G*ln(delta) because the log-singular part of psi'/psi has a
    # side-dependent coefficient. Fit Delta'(delta) = Delta'_true + G*ln(delta) and return the intercept.
    xs, ys = [], []
    for d in deltas:
        v, rs = deltaprime(m, n, qc, nu, delta=d, N=N, current=True)
        if v is None:
            return None, None, None
        xs.append(np.log(d)); ys.append(float(np.real(v)))
    A = np.vstack([np.ones_like(xs), xs]).T
    (b0, G), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    resid = float(np.sqrt(np.mean((A @ np.array([b0, G]) - np.array(ys))**2)))
    return b0, G, resid   # b0 = r_s*Delta'_true (intercept), G = log slope, resid = fit residual

CASES = [("q1",0.85,1.5,1),("q2",1.5,2.8,2),("q3",2.2,3.8,3),("q4",3.2,4.8,4),("q5",4.2,5.8,5)]

if __name__ == "__main__":
    print("="*74)
    print("  Cylindrical Newcomb Delta-prime (analytic TJ q-profile, no wall)")
    print("="*74)
    print("\n[self-test] current term OFF -> r_s*Delta' should equal -2m exactly:")
    for lab, qc, qa, m in CASES:
        nu = qa/qc
        val, rs = deltaprime(m, 1, qc, nu, delta=1e-3, N=200000, current=False)
        print(f"  {lab} (m={m}): r_s*Delta'={val:+.4f}   expected -2m={-2*m:+d}   r_s={rs:.4f}   "
              f"{'OK' if abs(val+2*m)<0.05 else 'CHECK'}")
    print("\n[physical] current term ON, delta-convergence (r_s*Delta'):")
    hdr = "  case   r_s      " + "".join(f"d={d:<9}" for d in ("1e-2","3e-3","1e-3"))
    print(hdr)
    for lab, qc, qa, m in CASES:
        nu = qa/qc
        vals = []
        for d in (1e-2, 3e-3, 1e-3):
            v, rs = deltaprime(m, 1, qc, nu, delta=d, N=300000, current=True)
            vals.append(v)
        print(f"  {lab:4s}  {rs:.4f}  " + "".join(f"{v:<+11.4f}" for v in vals))
