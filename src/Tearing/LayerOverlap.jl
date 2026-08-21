# LayerOverlap.jl
#
# Resistive-layer overlap scan: how far out in ψ the equilibrium domain needs
# to extend before adjacent rational surfaces' resistive layers run into each
# other. Once two neighbouring layers overlap, neither surface has a
# well-separated inner region, so the matched-asymptotic treatment stops being
# meaningful there and extending `psihigh` past that point buys nothing.
# Criterion: Fitzpatrick, Nucl. Fusion 2025 (doi 10.1088/1741-4326/ae4fdd) Sect. 5.9 —
# retain surfaces with psi < 1 - eps_c, where overlap is judged with the
# diffusive-resistive layer width of its Eq. (100), the regime strong shear forces
# near the separatrix.
#
# Lives in `Tearing` rather than `InnerLayer/SLAYER` for the same reason
# `build_ggj_inputs` does: it needs `ForceFreeStates._find_rational_surfaces`,
# and `InnerLayer` loads before `ForceFreeStates`.
#
# Surfaces beyond the equilibrium's own ψ grid are located on the separatrix edge
# law q ≈ -A·ln(1-ψ), via the shared `Equilibrium.edge_q_law` helper that also gates
# the grid-refinement edge density model -- one statement of the model, one fit, one
# limited-plasma guard. The log form is the only one that survives the
# extrapolation: a cubic in ψ (on q or on ι = 1/q) saturates instead of diverging. Measured on the DIII-D-like deck, extrapolating
# from a grid ending at ψ=0.97 to the true q=8 surface at ψ=0.99976, the edge law
# lands within 5% while a cubic ι undershoots by 18% — enough to miss the surface
# entirely and report no overlap where there is one.
#
# The extrapolated surfaces inform the *choice* of domain only; the final
# equilibrium is always re-formed and solved on the accepted domain.

using ..Utilities: KineticProfiles
using ..ForceFreeStates: _find_rational_surfaces, SingType
using ..InnerLayer.SLAYER: SLAYERParameters, build_slayer_inputs, slayer_layer_thickness,
    surface_minor_radius, surface_da_dpsi, radial_label
using ..Utilities.NeoclassicalResistivity: NeoResistivityModel, SauterNeoModel
using ..Equilibrium: edge_q_law, edge_q_law_psi, edge_q_law_dqdpsi, InverseIngest
using FastInterpolations: cubic_interp, ExtendExtrap, DerivOp
using Roots: find_zero, Brent
using Printf: @sprintf

"""
    LayerOverlapScan

Per-surface resistive layer widths and the `psihigh` they imply, from
[`resistive_layer_overlap`](@ref).

All widths are in **normalized flux**, converted from the metre-valued
`LayerWidths` scales by `Δψ = w / |da/dψ|` so they can be compared against
surface *separations* in ψ. Comparing the metre values directly against ψ is a
dimensional error.

# Fields

  - `m`, `n` -- resonant mode numbers per surface, ordered by increasing ψ
  - `psi` -- surface location in normalized flux
  - `rs` -- minor radius at the surface in meters
  - `delta_s_m` -- `|δ_s|`, the Riccati resistive layer width, in meters
  - `width_delta_s` -- the same width in normalized flux, `|δ_s|/|da/dψ|`
  - `width_visco` -- the visco-resistive comparison scale as a Δψ width
  - `width_dr` -- the diffusive-resistive width of Fitzpatrick (2025) Eq. (100) as a Δψ
    width. This is the criterion the recommendation uses: the paper's Sect. 5.6 shows that
    strong shear near the separatrix forces every layer there into the DR regime
  - `extrapolated` -- true when the surface was located outside the
    equilibrium's ψ grid via the edge q-law
  - `psihigh_delta_s`, `psihigh_visco` -- domain implied by each width channel,
    `nothing` when that channel never overlaps within the scanned range
  - `psihigh` -- the recommendation: the smaller of the two, or `nothing`
  - `first_overlap` -- index of the first surface that overlaps its inner
    neighbour, `nothing` when none do
  - `notes` -- surfaces that were located but could not be scored, and why
"""
struct LayerOverlapScan
    m::Vector{Int}
    n::Vector{Int}
    psi::Vector{Float64}
    rs::Vector{Float64}
    delta_s_m::Vector{Float64}
    width_delta_s::Vector{Float64}
    width_visco::Vector{Float64}
    width_dr::Vector{Float64}
    extrapolated::Vector{Bool}
    psihigh_delta_s::Union{Float64,Nothing}
    psihigh_visco::Union{Float64,Nothing}
    psihigh_dr::Union{Float64,Nothing}
    psihigh::Union{Float64,Nothing}
    first_overlap::Union{Int,Nothing}
    notes::Vector{String}
end

# Inverse equilibria (CHEASE) are solved on a PRESCRIBED boundary: `InverseIngest.sq_xs` spans
# [0, 1] and `sq_fs[:, 3]` is the code's own q there, finite at ψ = 1 (6.90 on the shipped
# fixture). So beyond `psihigh` there is nothing to model -- the real q is already known, and the
# scan reads it instead of extrapolating. That also means the search runs all the way to ψ = 1
# rather than stopping short of a separatrix that a fixed-boundary equilibrium does not have.
#
# The direct/EFIT path cannot do this: a g-file also carries q on [0, 1], but it is the
# reconstruction's q, which goes to a finite qa where a diverted plasma's q diverges. GPEC
# recomputes q by field-line tracing out to psihigh, and past that nothing has been traced.
_inverse_q_spline(ingest::InverseIngest) =
    cubic_interp(collect(Float64, ingest.sq_xs), collect(Float64, ingest.sq_fs[:, 3]);
        extrap=ExtendExtrap())

# Locate q = q_target on the real q profile, between `psi_lo` and `psi_hi`. Returns nothing when
# the target is outside the range this equilibrium actually reaches.
function _real_q_surface(qspl, q_target::Real, psi_lo::Float64, psi_hi::Float64)
    q_lo, q_hi = Float64(qspl(psi_lo)), Float64(qspl(psi_hi))
    (q_lo - q_target) * (q_hi - q_target) <= 0 || return nothing
    return find_zero(p -> Float64(qspl(p)) - q_target, (psi_lo, psi_hi), Brent())
end

"""
    resistive_layer_overlap(equil, profiles::KineticProfiles; n_tor, kwargs...)
        -> LayerOverlapScan

Locate the q = m/`n_tor` rational surfaces, compute each one's resistive layer
width, and report the outermost `psihigh` at which adjacent layers are still
separated.

Surfaces inside the equilibrium grid come from
`ForceFreeStates._find_rational_surfaces` (Brent root-finding segmented between
q-extrema, so reverse shear is handled). Surfaces beyond the grid come from the
separatrix edge law `q ≈ -A·ln(1-ψ)` fitted over the outer knots when
`extrapolate=true`, which is what lets the scan recommend a domain *larger* than
the one it was handed. See the file header for why a cubic extrapolation is not
usable here.

Layer widths come from [`slayer_layer_thickness`](@ref) via
[`build_slayer_inputs`](@ref), so the scan uses the same plasma inputs and the
same resistivity closure as the SLAYER analysis itself.

# Arguments

  - `equil` -- a formed `PlasmaEquilibrium`
  - `profiles` -- `KineticProfiles` spanning the scanned ψ range

# Keyword arguments

  - `n_tor` -- toroidal mode number; surfaces are q = m/`n_tor`

  - `m_max` -- runaway backstop on the poloidal mode number (default `2000`). This is **not**
    the physics limit: the outward search terminates when the edge law places a surface at or
    beyond `psi_cap`, i.e. when it is indistinguishable from the separatrix. On a diverted
    equilibrium rational surfaces accumulate without bound toward ψ = 1, so some cap is needed,
    but it should never be what stops the scan — if it is, the scan says so in `notes`, because
    "no overlap found" would otherwise be reported for a non-physical reason
  - `psihigh_safe` -- treat this as the outer edge of trusted equilibrium data
    (default `equil.profiles.xs[end]`)
  - `psi_cap` -- never place a surface beyond this ψ on the **direct** path (default
    `1 - 1e-9`). Purely numerical: it is not derived from the contour clamp or from any topology
    test, and it stops the outward search from crowding indefinitely onto ψ = 1 where surfaces
    become indistinguishable. Loosening it is cheap (1e-6 → 1e-9 adds ~5 surfaces on the
    DIII-D-like deck at no measurable cost and does not change the answer) and guards against
    missing an overlap that lies further out than the cap. Inverse equilibria ignore it: they
    have a prescribed boundary and the search runs to ψ = 1 on the real q.

    Note the accuracy limit here is **not** `psi_cap` but `surface_da_dpsi`, which clamps its
    finite-difference stencil to `1 - 1e-4`; surfaces closer than that to the boundary get
    geometry evaluated at `1 - 1e-4` rather than at their own ψ (da/dψ = 0.348729 identically
    for every ψ beyond it on the DIII-D-like deck). Their widths are therefore approximate
  - `extrapolate` -- search past `psihigh_safe` on the edge q-law (default `true`)
  - `theta` -- poloidal angle for the minor-radius chord (default `0.0`)
  - `mu_i`, `zeff`, `chi_perp`, `chi_tor`, `resistivity_model`, `lnLambda_form`
    -- passed through to `build_slayer_inputs`

The overlap criterion is that surface `k` overlaps its inner neighbour when
`ψ_k − w_k/2 < ψ_{k-1} + w_{k-1}/2`. When that happens **both** surfaces are
contaminated, so the recommendation cuts at the inner edge of surface `k-1`,
leaving `k-2` as the outermost trustworthy surface.
"""
function resistive_layer_overlap(equil, profiles::KineticProfiles;
    n_tor::Integer,
    m_max::Integer=2000,
    max_layer_solves::Integer=64,
    psihigh_safe::Real=Float64(equil.profiles.xs[end]),
    psi_cap::Real=1.0 - 1e-9,
    extrapolate::Bool=true,
    theta::Real=0.0,
    rs_method::Symbol=:midplane,
    mu_i::Real=2.0,
    zeff::Real=1.0,
    chi_perp=1.0,
    chi_tor=1.0,
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    lnLambda_form::Symbol=:nrl)

    n_tor > 0 || throw(ArgumentError("resistive_layer_overlap: n_tor must be positive, got $n_tor"))
    # Widths come back in metres of whichever radial label `rs_method` selects, so the
    # conversion into normalised flux must use that same label's Jacobian.
    _, _dr_dpsi = radial_label(equil; rs_method=rs_method, theta=theta)
    psihigh_safe = Float64(psihigh_safe)
    psi_cap = Float64(psi_cap)
    notes = String[]

    # In-grid surfaces for this n, ordered outward.
    found = [(m=s.m, psi=s.psifac, extrap=false)
             for s in _find_rational_surfaces(equil, Int(n_tor), Int(n_tor))
             if s.psifac <= psihigh_safe && s.m <= m_max]
    sort!(found; by=s -> s.psi)

    # Surfaces past the grid. An inverse equilibrium already knows its real q out to ψ = 1, so
    # it is read rather than modelled; only the direct path needs the edge law.
    edge_fit = nothing
    inv_q = (getfield(equil, :ingest) isa InverseIngest) ? _inverse_q_spline(equil.ingest) : nothing
    if extrapolate && inv_q !== nothing
        psi_top = 1.0                      # a prescribed boundary, not a separatrix
        m_start = isempty(found) ? 1 : maximum(s.m for s in found) + 1
        reached_end = false
        for m in m_start:Int(m_max)
            psi_m = _real_q_surface(inv_q, m / n_tor, psihigh_safe, psi_top)
            if psi_m === nothing
                reached_end = true         # q never reaches m/n inside this equilibrium
                break
            end
            psi_m > psihigh_safe || continue
            push!(found, (m=m, psi=psi_m, extrap=true))
        end
        !reached_end && m_start <= m_max &&
            push!(notes,
                "outward search stopped at the m_max = $m_max backstop rather than at the boundary " *
                "q; a \"no overlap\" result here is not conclusive -- raise m_max")
        push!(notes, "inverse equilibrium: surfaces beyond ψ = $(@sprintf("%.6f", psihigh_safe)) " *
                     "read from the real q profile out to ψ = 1 (no edge-law extrapolation)")
    elseif extrapolate && psihigh_safe < psi_cap
        edge_fit = edge_q_law(equil; psi_max=psihigh_safe)
        if edge_fit === nothing
            push!(
                notes,
                "edge q-law rejected: the diverging model q ~ -A*ln(1-ψ) does not describe " *
                "this edge (limited plasma with finite edge q, q not rising, or too few outer " *
                "knots), so no surfaces are placed beyond ψ = $(@sprintf("%.6f", psihigh_safe))"
            )
        else
            m_start = isempty(found) ? 1 : maximum(s.m for s in found) + 1
            # Terminates on psi_cap -- surfaces indistinguishable from the separatrix -- not on
            # m_max. Reaching m_max means the scan was cut short for a non-physical reason and
            # any "no overlap" verdict from it is unsafe, so record that.
            reached_cap = false
            for m in m_start:Int(m_max)
                psi_m = edge_q_law_psi(edge_fit, m / n_tor)
                psi_m > psihigh_safe || continue   # already inside the grid
                if psi_m >= psi_cap
                    reached_cap = true             # separatrix reached: the physical end of the scan
                    break
                end
                push!(found, (m=m, psi=psi_m, extrap=true))
            end
            if !reached_cap && m_start <= m_max
                push!(notes,
                    "outward search stopped at the m_max = $m_max backstop rather than at the " *
                    "separatrix; a \"no overlap\" result here is not conclusive -- raise m_max")
            end
        end
    end

    isempty(found) && return LayerOverlapScan(Int[], Int[], Float64[], Float64[], Float64[], Float64[],
        Float64[], Bool[], nothing, nothing, nothing, nothing,
        push!(notes, "no q = m/$n_tor surfaces found"))

    ms, ns, psis, rss = Int[], Int[], Float64[], Float64[]
    dels_m, w_dels, w_visc, w_dr, extraps = Float64[], Float64[], Float64[], Float64[], Bool[]

    # Overlap is decided by the first crossing walking outward, so surfaces far beyond it cannot
    # change the answer -- but each one costs a Riccati layer solve, and this scan runs whenever
    # kinetic profiles are readable, flag or no flag. Bound the work rather than only the m index,
    # and say so when the bound bites, since a "no overlap" verdict from a truncated list is not
    # conclusive.
    if length(found) > max_layer_solves
        push!(notes,
            "surface list truncated from $(length(found)) to the innermost $max_layer_solves for " *
            "the layer solve (max_layer_solves); a \"no overlap\" verdict here is not conclusive")
        found = found[1:max_layer_solves]
    end

    for s in found
        q_val = s.m / n_tor
        # dq/dψ from the equilibrium spline in-grid, from the edge law outside it.
        q1_val = if !s.extrap
            Float64(equil.profiles.q_deriv(s.psi))
        elseif inv_q !== nothing
            Float64(inv_q(s.psi; deriv=DerivOp(1)))   # real profile, not a model
        else
            edge_q_law_dqdpsi(edge_fit, s.psi)
        end
        da_dpsi = _dr_dpsi(s.psi)
        if !isfinite(da_dpsi) || da_dpsi == 0.0
            push!(notes, "m=$(s.m): da/dψ = $da_dpsi at ψ=$(@sprintf("%.6f", s.psi)); cannot convert width to flux units")
            continue
        end

        sing = SingType(; m=[s.m], n=[Int(n_tor)], psifac=s.psi, rho=sqrt(s.psi), q=q_val, q1=q1_val)
        # Built one surface at a time so a single degenerate surface (e.g. ω_*e == ω_*i,
        # which makes iota_e singular) is recorded and skipped rather than aborting the scan.
        params = try
            build_slayer_inputs(equil, [sing], profiles; rs_method=rs_method,
                mu_i=mu_i, zeff=zeff, chi_perp=chi_perp, chi_tor=chi_tor,
                dr_val=0.0, dc_type=:none, theta=theta,
                resistivity_model=resistivity_model, lnLambda_form=lnLambda_form)
        catch err
            err isa ArgumentError || rethrow()
            push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): $(err.msg)")
            continue
        end

        lw = slayer_layer_thickness(params[1])
        if !isfinite(lw.delta_s_m)
            push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): del_s Riccati did not converge")
            continue
        end
        # Both widths must be finite. A NaN delta_visco would compare false against every
        # neighbour in _first_overlap_limit, so the viscous criterion would silently report "no
        # overlap" and `recommended` would quietly lose its more conservative half -- biasing the
        # domain outward, which is the unsafe direction.
        if !isfinite(lw.delta_dr)
            push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): Eq. (100) width is not finite; " *
                         "surface excluded so the DR criterion cannot fail open")
            continue
        end
        if !isfinite(lw.delta_visco)
            push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): viscous width is not finite; " *
                         "surface excluded so the viscous criterion cannot fail open")
            continue
        end

        # Metres → normalized flux. This conversion is the whole reason the scan
        # can compare a layer width against a surface separation.
        push!(ms, s.m)
        push!(ns, Int(n_tor))
        push!(psis, s.psi)
        push!(rss, params[1].rs)
        push!(dels_m, lw.delta_s_m)
        push!(w_dels, lw.delta_s_m / abs(da_dpsi))
        push!(w_visc, lw.delta_visco / abs(da_dpsi))
        push!(w_dr, lw.delta_dr / abs(da_dpsi))
        push!(extraps, s.extrap)
    end

    ph_dels, k_dels = _first_overlap_limit(psis, w_dels)
    ph_visc, k_visc = _first_overlap_limit(psis, w_visc)
    ph_dr, k_dr = _first_overlap_limit(psis, w_dr)
    # The DR width is the criterion: near the separatrix the shear is strong enough that
    # Fitzpatrick (2025) Sect. 5.6 puts every layer in that regime. The delta_s ODE and the
    # viscous scale stay reported so the three can be compared, but they do not set the domain.
    recommended = ph_dr

    return LayerOverlapScan(ms, ns, psis, rss, dels_m, w_dels, w_visc, w_dr, extraps,
        ph_dels, ph_visc, ph_dr, recommended, k_dr, notes)
end

# Walk outward and return (recommended psihigh, index of first overlapping surface).
# Overlap at k means surfaces k and k-1 are both contaminated, so the domain is cut
# at the inner edge of k-1.
function _first_overlap_limit(psi::Vector{Float64}, w::Vector{Float64})
    for k in 2:length(psi)
        if psi[k] - w[k] / 2 < psi[k-1] + w[k-1] / 2
            return (psi[k-1] - w[k-1] / 2, k)
        end
    end
    return (nothing, nothing)
end

function Base.show(io::IO, ::MIME"text/plain", s::LayerOverlapScan)
    println(io, "LayerOverlapScan: $(length(s.psi)) surface(s), n = ",
        isempty(s.n) ? "-" : string(s.n[1]))
    if !isempty(s.psi)
        println(io, rpad("q", 8), rpad("psi", 13), rpad("r_s [m]", 11),
            rpad("dpsi(del_s)", 14), rpad("dpsi(visco)", 14), rpad("dpsi(DR)", 14), "source")
        for i in eachindex(s.psi)
            println(io,
                rpad("$(s.m[i])/$(s.n[i])", 8),
                rpad(@sprintf("%.7f", s.psi[i]), 13),
                rpad(@sprintf("%.5f", s.rs[i]), 11),
                rpad(@sprintf("%.4e", s.width_delta_s[i]), 14),
                rpad(@sprintf("%.4e", s.width_visco[i]), 14),
                rpad(@sprintf("%.4e", s.width_dr[i]), 14),
                s.extrapolated[i] ? "extrapolated" : "in-grid")
        end
    end
    _fmt(x) = x === nothing ? "none (no overlap in range)" : @sprintf("%.7f", x)
    println(io, "  psihigh from |del_s| : ", _fmt(s.psihigh_delta_s))
    println(io, "  psihigh from visco   : ", _fmt(s.psihigh_visco))
    println(io, "  psihigh from Eq.(100): ", _fmt(s.psihigh_dr))
    println(io, "  recommended psihigh  : ", _fmt(s.psihigh))
    for note in s.notes
        println(io, "  note: ", note)
    end
    return nothing
end
