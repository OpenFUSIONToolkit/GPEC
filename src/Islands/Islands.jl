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

# Submodules land here as M1 proceeds, following the layout in
# docs/src/islands/design/03-architecture.md §1:
#   include("phasespace/PhaseSpace.jl")   # grids (x, ξ, λ, E, σ), layer-clustered maps
#   include("species/Species.jl")         # Species, backgrounds, roles
#   include("frames/Frames.jl")           # THE frequency/frame conversion module
#   include("operators/Operators.jl")     # the AbstractTerm stack
#   include("fields/Fields.jl")           # Φ̃ quasineutrality (A_∥ Ampère at L3)
#   include("moments/Moments.jl")         # Δ_cos, Δ_sin, profiles, channel decompositions
#   include("solvers/Solvers.jl")         # Newton–Krylov, continuation, trace pass
#   include("verify/Verify.jl")           # MMS + analytic-limit hooks, named configs

end # module Islands
