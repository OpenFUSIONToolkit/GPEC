# Reference.jl
#
# Hard-coded benchmark parameter sets for cross-validation of the GGJ
# inner-layer solvers. Values are taken from published papers.

"""
    glasser_wang_2020_eq55(; taua=1.0, taur=1e6, v1=1.0) -> GGJParameters

D-shaped aspect-ratio-2, q = 2 surface from Glasser & Wang, Phys. Plasmas
**27**, 012506 (2020), Eq. 55. This is the primary benchmark case for
validating the inps Wasow basis convergence (their Figs. 1–4). This is useful
only for benchmarking the galerkin solver and comparing to published results.

The five coefficients below are transcribed verbatim from Eq. 55; the paper's
companion operating point is the scaled growth rate `Q = 1.234e-1` (their Fig. 1).
Note Eq. 55 does not tabulate an inner-region matching `Δ(Q)` — its `Δ_±` (Eq. 54)
is a convergence-error norm — so a quantitative `Δ(Q)` cross-check needs an
independent inner-layer reference, not this paper alone.

Timescale parameters (taua, taur, v1) are set to canonical normalization;
callers should override them for physical cases.
"""
function glasser_wang_2020_eq55(; taua::Float64=1.0, taur::Float64=1e6, v1::Float64=1.0)
    return GGJParameters(;
        E=-3.369e-2, F=2.409e-3, G=8.950e1, H=1.292e-2, K=2.332e2,
        taua=taua, taur=taur, v1=v1
    )
end

"""
    q4_surface_benchmark() -> GGJParameters

Physical q = 4 rational-surface benchmark point (S = τ_R/τ_A ≈ 4.58×10⁶,
D_I ≈ −0.31166, α = √(−D_I) ≈ 0.5583). Primary validation point for the
rotated-ray backend on the imaginary-Q axis (pinned at Q = 100i, 500i in
the test suite).
"""
function q4_surface_benchmark()
    return GGJParameters(;
        E=-0.13733, F=0.022202, G=7.60633, H=0.053468, K=14.66987,
        M=30.26883, taua=2.11226e-7, taur=0.968219, v1=1.55009
    )
end
