"""Native proof of the zero-curvature pressure-work oracle."""
import fortsym.sympy as f
Dr,Di=f.symbols("Dr Di")
work=0
for D in (Dr,Di):
    b=-D
    pp=-D+b
    pa=-3*D-2*b
    work-=((pa-pp)*b+pa*D)/2
assert (work-Dr**2-Di**2).is_zero is True
print("PROVED: zero-curvature CGL pressure work = |D|**2")
