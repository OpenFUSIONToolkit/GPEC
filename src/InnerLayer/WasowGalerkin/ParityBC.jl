# ParityBC.jl
#
# Parity boundary-condition descriptor for the half-domain inner-layer FEM.
#
# A second-order inner-layer system in `ncomp` physical components, solved on
# the half domain x ∈ [0, xmax], admits a parity decomposition at x = 0: each
# physical component is either even (value-free, derivative pinned) or odd
# (derivative-free, value pinned) under x → −x. Each parity choice yields one
# homogeneous solve; the collection of solves produces the matching-data vector.
#
# `ParityBC` lists, per parity solve and per component, whether the boundary
# condition at x = 0 is Dirichlet (component value = 0) or Neumann (component
# derivative = 0). This generalizes the hardcoded (W, N, Θ) even/odd "pokes" of
# the original GGJ deltac.f port.

"""
    BCKind

Boundary-condition kind imposed on a single physical component at x = 0:

  - `DIRICHLET` — pin the component value to zero (`u(0) = 0`).
  - `NEUMANN`   — pin the component derivative to zero (`u'(0) = 0`).
"""
@enum BCKind DIRICHLET NEUMANN

"""
    ParityBC

Descriptor of the parity boundary conditions for the half-domain inner-layer
problem. Field `parities` is a vector with one entry per parity solve; each
entry is a length-`ncomp` vector of [`BCKind`](@ref) selectors, one per physical
component.

For the GGJ single-fluid system (`ncomp = 3`, components `(W, N, Θ)`) the two
parities are:

  - odd  : `(NEUMANN, DIRICHLET, DIRICHLET)` — `W'(0)=0, N(0)=0, Θ(0)=0`
  - even : `(DIRICHLET, NEUMANN, NEUMANN)`   — `W(0)=0, N'(0)=0, Θ'(0)=0`

A model whose physics breaks `x → −x` parity (e.g. a rotating two-fluid layer)
uses the full-domain path instead and supplies a single combined solve.
"""
struct ParityBC
    parities::Vector{Vector{BCKind}}
end

"""
    nparities(bc::ParityBC) -> Int

Number of parity solves described by `bc` (2 for the GGJ even/odd pair).
"""
nparities(bc::ParityBC) = length(bc.parities)
