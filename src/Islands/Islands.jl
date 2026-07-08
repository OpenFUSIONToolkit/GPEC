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
# M1 lands the discretization + operator-stack skeleton + verification harness;
# the remaining submodules (species, frames, fields, moments, solvers) land as
# later milestones proceed.
include("phasespace/PhaseSpace.jl")   # grids (x, ξ, λ→y, E, σ), layer-clustered maps
include("operators/Operators.jl")     # the AbstractTerm stack + residual assembly
include("verify/Verify.jl")           # MMS + AD-vs-FD JVP harness (ladder A1, A2)
#   include("species/Species.jl")     # Species, backgrounds, roles                 (M2+)
#   include("frames/Frames.jl")       # THE frequency/frame conversion module       (M2+)
#   include("fields/Fields.jl")       # Φ̃ quasineutrality (A_∥ Ampère at L3)        (M2+/L3)
#   include("moments/Moments.jl")     # Δ_cos, Δ_sin, profiles, channel decomps      (M2)
#   include("solvers/Solvers.jl")     # Newton–Krylov, continuation, trace pass      (M2)

import .PhaseSpace
import .Operators
import .Verify

end # module Islands
