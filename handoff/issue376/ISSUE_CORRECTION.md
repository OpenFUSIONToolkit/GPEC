### Correction to the analysis above — §6's mechanism is real but is *not* the dominant cause

Follow-up work has put a number on the causal claim I made above, and it does not hold up as
stated. Correcting it here rather than leaving it standing.

**What still holds.** The node-error floor is real and I have now traced it to its source. The
per-surface geometry (`nu`, `offset`, `rcoords` in `splines/rzphi`) sits on a ψ-direction floor
that is flat in the grid spacing — near-axis `nu` residuals 2.41e-5 → 2.11e-5 → 2.12e-5 for
mpsi 256/512/1024 — while `jac` on the same grid converges cleanly (2.22e-7 → 4.29e-8 → 3.26e-9).
In `Fourfit.jl:84-90` the clean matrices (A/B/D/F/K) are built only from **θ-derivatives** of that
data, and the dirty ones (C/E/H, via g11/g12/g31) from **ψ-derivatives** of exactly the three
floored quantities — so their error is ε/Δψ and grows as the grid refines. The 1D profiles are
cleared: q, mu0p, 2piF and dVdpsi all converge, so `q1`/`p1`/`jtheta` are not involved.

**What does not hold.** I claimed this floor was *the* reason the step count tracks knot count.
I tested that by repairing the amplification directly — replacing only the ψ-derivative channel
with derivatives of a fit over a coarser ψ subgrid, leaving values, θ-derivatives and the
coordinate mapping untouched. The repair works exactly as designed: mid-plasma C/E/H residuals
drop 20–100× (C: 1.21e-5 → 5.30e-7) and **every** knot-to-knot autocorrelation flips positive
(C: −0.401 → +0.915; G: −0.176 → +0.942), near-axis included.

The step count barely moves:

| grid | accepted steps, stock | with repair | change |
|---|---|---|---|
| mpsi=512  | 3977 | 3551 | −11% |
| mpsi=1024 | 7638 | 6081 | −20% |

So knot-scale roughness in C/E/H is a **contributing** cause worth 11–20%, not the dominant one.
The share does grow with mpsi, as an ε/Δψ effect should, but ~80% of the growth remains unexplained.

One pointer for whoever picks this up: even after the repair, near-axis |f″|/|f| for K is 2.1e7 —
a curvature scale of ~2e-4 in ψ, comparable to the local knot spacing there (median Δψ = 1.7e-4 at
mpsi=1024). Some of those steps may be legitimate resolution of real near-axis structure rather
than noise-chasing, which is consistent with the uniform-grid probe in §3 cutting near-axis steps
1891 → 559 at the cost of leaving et[1] visibly under-resolved (0.993).

Also corrected: two candidate origins of ε are ruled out. Beyond the `etol` sweep already reported,
`DirectEquilibrium.jl:294` hard-codes `abstol=1e-8` in the field-line trace while only `reltol` is
configurable — which looked like the smoking gun, since `u0` starts at zeros and its components
*become* ν/offset/r². Tightening it to 1e-14 is null (3977 → 4071 at mpsi=512; 7638 → 7474 at
mpsi=1024). Whatever sets ε, it is not the field-line integration's error control.

The recommendations above are unaffected — they rest on the tolerance and integrator measurements,
not on this attribution — except that "fix the C/E/H floor" should now be read as worth ~20% at
mpsi=1024 and growing, rather than as the single root-cause fix.
