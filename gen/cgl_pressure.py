"""FortSym native proof that the double-adiabatic pressure work equals the
quadratic form in calculate_fcgl (src/KineticForces/Torque.jl).

Run with a native FortSym build (verified at ae635c80):
    FORTSYM_LIBRARY=$FORTSYM_BUILD/lib/libfortsym.so PYTHONPATH=$FORTSYM/python python3 gen/cgl_pressure.py
"""
import fortsym.sympy as fs

Dr, Di, Kr, Ki = fs.symbols("Dr Di Kr Ki")
pressure = 0
energy = 0
for D, K in ((Dr, Kr), (Di, Ki)):
    b = -(D + K)
    pp, pa = -D + b, -3*D - 2*b
    pressure -= ((pa - pp)*b + pa*D)/2
    energy += (fs.Rational(5, 3)*D**2 + fs.Rational(1, 3)*(D + 3*K)**2)/2
assert (pressure - energy).is_zero is True
print("PROVED: complex CGL pressure work = (5|D|^2/3 + |D+3K|^2/3)/2")
