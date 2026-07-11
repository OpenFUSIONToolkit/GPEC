"""
    Islands

Steady-state, multi-species drift-kinetic solver for the resonant island/layer
region in tokamaks. Generalizes the Modified Rutherford Equation the way SLAYER
generalized linear layer theory: it returns the growth moment `Δ_cos(w, ω; p)`
and torque moment `Δ_sin(w, ω; p)` for arbitrary parameters, and in its
small-amplitude limit reduces to the linear layer response.

Design docs live in `docs/src/islands/` (module conventions in
`src/Islands/CLAUDE.md`, design docs in `docs/src/islands/design/`). Physics
equations transcribed from the York drift-kinetic lineage carry `[VERIFY]` /
`[CHECKED]` tags until human-cleared — see the module CLAUDE.md.

Status: skeleton. The operator stack, phase-space grids, field/moment assembly,
solvers, and verification harness (design doc `03-architecture.md §1`) land as
milestone M1 proceeds.
"""
module Islands

# Submodule layout follows docs/src/islands/design/03-architecture.md §1.
# M1 landed the discretization + operator stack + MMS/AD harness; M2 lands the
# L0 solve machinery (species, frames, solve, fields, moments) as gated
# structure. Remaining design dirs (geometry/, closures/, io/) arrive with the
# milestones that need them (L2/L4/M2-io).
include("phasespace/PhaseSpace.jl")   # grids (x, ξ, λ→y, E, σ), layer-clustered maps
include("species/Species.jl")         # Species, backgrounds, roles (D3)
include("frames/Frames.jl")           # THE frequency/frame conversion module (gated forms)
include("operators/Operators.jl")     # the AbstractTerm stack + residual assembly
include("fields/Fields.jl")           # quasineutrality closure structure, h(Ω)/Q(Ω)
include("moments/Moments.jl")         # J̄_∥, Δ_cos/Δ_sin projections, ⟨·⟩_Ω diagnostics
include("coefficients/Coefficients.jl") # human-cleared L0 physics coefficient builders (M2b, D7)
include("solvers/Solvers.jl")         # Newton–Krylov, preconditioner, continuation
include("verify/Verify.jl")           # MMS + AD-vs-FD JVP harness, y_c monitor

import .PhaseSpace
import .SpeciesLists
import .Frames
import .Operators
import .Fields
import .Moments
import .Coefficients
import .Solvers
import .Verify

end # module Islands
