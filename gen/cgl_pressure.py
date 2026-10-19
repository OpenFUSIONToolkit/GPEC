"""FortSym native proof of the CGL pressure-work oracle (no numerical fallback)."""
import fortsym.sympy as fs
Dr, Di, Kr, Ki = fs.symbols("Dr Di Kr Ki")
pressure = 0
energy = 0
for D, K in ((Dr, Kr), (Di, Ki)):
    b = -(D + K)
    pp, pa = -D + b, -3*D - 2*b
    pressure -= ((pa-pp)*b + pa*D)/2
    energy += (fs.Rational(5,3)*D**2 + fs.Rational(1,3)*(D+3*K)**2)/2
assert (pressure-energy).is_zero is True
print("K5 PROVED: complex CGL pressure work")
# Curvature reconstruction from the consumer's own perpendicular convention.
D, K = fs.symbols("D K")
b = -(D+K)
assert (-(b+D)-K).is_zero is True
print("K6 PROVED: kappa.xi_perp = -(dbob + divx)")
