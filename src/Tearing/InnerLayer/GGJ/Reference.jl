# Reference.jl
#
# Hard-coded benchmark parameter sets for cross-validation of the GGJ
# inner-layer solvers. Values are taken from published papers and the
# corresponding Fortran RMATCH test cases.

"""
    glasser_wang_2020_eq55() -> GGJParameters

D-shaped aspect-ratio-2, q = 2 surface from Glasser & Wang, Phys. Plasmas
**27**, 012506 (2020), Eq. 55. This is the primary benchmark case for
validating the inps Wasow basis convergence (their Figs. 1–4). This is useful
only for benchmarking the galerkin solver and comparing to published results.

Timescale parameters (taua, taur, v1) are set to canonical normalization;
callers should override them for physical cases.
"""
function glasser_wang_2020_eq55(; taua::Float64=1.0, taur::Float64=1e6, v1::Float64=1.0)
    return GGJParameters(
        E=-3.369e-2, F=2.409e-3, G=8.950e-1, H=1.292e-1, K=2.332e-1;
        taua=taua, taur=taur, v1=v1,
    )
end
