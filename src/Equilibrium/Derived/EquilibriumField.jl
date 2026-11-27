"""
    equilibrium_field(equil, R, Z) -> (; B_R, B_phi, B_Z, psi_norm)

The equilibrium magnetic field in T at machine coordinates `R`, `Z` in m (arrays of one shape), in the
right-handed (R, φ, Z) frame of the source file, and the normalized poloidal flux there. It comes from the
file's flux map ψ(R, Z) and F profile; outside the last closed flux surface F takes its edge value.

The equilibrium must come from a g-file or IMAS, and every point must lie inside its flux grid.
"""
function equilibrium_field(equil::PlasmaEquilibrium, R::AbstractArray{<:Real}, Z::AbstractArray{<:Real})
    ingest = equil.ingest
    ingest isa DirectIngest || error("equilibrium_field needs the flux map outside the plasma, which only g-file " *
          "and IMAS equilibria carry; eq_type = \"$(equil.config.eq_type)\" has none")
    size(R) == size(Z) || throw(DimensionMismatch("R has size $(size(R)) and Z has size $(size(Z))"))
    for (r, z) in zip(R, Z)
        (ingest.rmin <= r <= ingest.rmax && ingest.zmin <= z <= ingest.zmax) ||
            error(
                @sprintf("(R, Z) = (%.3f, %.3f) m lies outside the equilibrium's flux grid R ∈ [%.3f, %.3f], Z ∈ [%.3f, %.3f] m; ",
                    r, z, ingest.rmin, ingest.rmax, ingest.zmin, ingest.zmax) *
                "re-export the equilibrium on a grid that covers it"
            )
    end

    psi_in = cubic_interp((ingest.psi_xs, ingest.psi_ys), ingest.psi_rz; extrap=ExtendExtrap())
    F_in = cubic_interp(ingest.sq_xs, ingest.sq_fs[:, 1]; extrap=ExtendExtrap())
    # The solver rescales ψ to equal psio exactly at the magnetic axis; do the same.
    scale = ingest.psio / psi_in((equil.ro, equil.zo))
    psi_norm_at(r, z) = 1 - scale * psi_in((r, z)) / ingest.psio
    # Flux surfaces are star-shaped about the axis, so a point is inside the last closed flux surface when the
    # segment to it from the axis stays at ψ_N < 1; below an X-point it crosses ψ_N ≥ 1.
    h = min(ingest.psi_xs[2] - ingest.psi_xs[1], ingest.psi_ys[2] - ingest.psi_ys[1])
    inside(r, z) = all(t -> psi_norm_at(equil.ro + t * (r - equil.ro), equil.zo + t * (z - equil.zo)) < 1,
        range(0, 1; length=max(2, ceil(Int, hypot(r - equil.ro, z - equil.zo) / h) + 1)))

    B_R, B_phi, B_Z, psi_norm = (similar(R, Float64) for _ in 1:4)
    for i in eachindex(R, Z)
        r, z = Float64(R[i]), Float64(Z[i])
        psi_norm[i] = psi_norm_at(r, z)
        # The processed ψ peaks at the axis whatever the current's direction; ip_sign restores it.
        B_R[i] = -ingest.ip_sign * scale * psi_in((r, z); deriv=DerivOp(0, 1)) / r
        B_Z[i] = ingest.ip_sign * scale * psi_in((r, z); deriv=DerivOp(1, 0)) / r
        F = psi_norm[i] < 1 && inside(r, z) ? F_in(psi_norm[i]) : ingest.sq_fs[end, 1]
        B_phi[i] = ingest.bt_sign * F / r
    end
    return (; B_R, B_phi, B_Z, psi_norm)
end
