"""
Radial grid refinement utilities for the two-pass auto ψ grid.

Pass 1 forms the equilibrium on a coarse grid; `refined_psi_grid` then derives a knot
density from the measured nodal data (profiles, 2D geometry channels, and optional kinetic
profiles) and equidistributes it, pinning mandatory knots (e.g. rational surfaces) via
`merge_mandatory_nodes`. Pass 2 re-forms the equilibrium on the refined grid through the
`override_psi_nodes` keyword of `setup_equilibrium`.
"""

"""
    _validate_psi_nodes(psi_nodes, psilow, psihigh) -> Vector{Float64}

Check that an externally supplied ψ node vector is strictly increasing and spans exactly
[`psilow`, `psihigh`]. Errors loudly on violation (e.g. a psihigh separatrix re-clamp
between grid construction and equilibrium formation).
"""
function _validate_psi_nodes(psi_nodes::Vector{Float64}, psilow::Real, psihigh::Real)
    length(psi_nodes) >= 2 || error("override_psi_nodes must contain at least 2 nodes")
    all(diff(psi_nodes) .> 0) || error("override_psi_nodes must be strictly increasing")
    isapprox(psi_nodes[1], psilow; atol=1e-12) ||
        error("override_psi_nodes[1]=$(psi_nodes[1]) must equal psilow=$psilow")
    isapprox(psi_nodes[end], psihigh; atol=1e-12) ||
        error("override_psi_nodes[end]=$(psi_nodes[end]) must equal psihigh=$psihigh")
    return psi_nodes
end
