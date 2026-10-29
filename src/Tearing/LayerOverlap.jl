# LayerOverlap.jl
#
# Resistive-layer overlap scan: where adjacent rational surfaces' resistive layers run into each
# other. Past that point neither surface keeps a well-separated inner region, so the
# matched-asymptotic treatment is not defined there. Criterion: Fitzpatrick, Nucl. Fusion 2025
# (doi 10.1088/1741-4326/ae4fdd) Sect. 5.9, judged with the diffusive-resistive width of its
# Eq. (100). How the result is used is documented in the integration-domain truncation section of
# docs/src/stability.md.
#
# Lives in `Tearing` rather than `InnerLayer/SLAYER` because it needs
# `ForceFreeStates._find_rational_surfaces`, and `InnerLayer` loads before `ForceFreeStates`.
#
# Surfaces beyond the equilibrium's own ψ grid are located on `Equilibrium.separatrix_q_law`: the
# logarithmic divergence of q at the X-point(s) on the last closed flux surface, with its
# location and slope taken from the 2D flux and its constant from the traced q on the outermost
# grid surface. The scan only recommends a domain; nothing here re-forms the equilibrium.
using ..Utilities: KineticProfiles
using ..ForceFreeStates: _find_rational_surfaces, SingType
using ..InnerLayer.SLAYER: SLAYERParameters, build_slayer_inputs, slayer_algebraic_widths, riccati_del_s,
    surface_minor_radius, surface_da_dpsi, radial_label
using ..Utilities.NeoclassicalResistivity: NeoResistivityModel, SauterNeoModel
using ..Equilibrium: separatrix_q_law, separatrix_q, separatrix_q_psi, separatrix_q_dqdpsi, InverseIngest
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
  - `psihigh_delta_s`, `psihigh_visco`, `psihigh_dr` -- domain implied by each width
    channel, `nothing` when that channel never overlaps within the scanned range
  - `psihigh` -- the recommendation, equal to `psihigh_dr`; the other two channels are reported
    for comparison and do not set the domain
  - `first_overlap` -- index of the first surface that overlaps its inner
    neighbour, `nothing` when none do
  - `caveats` -- reasons the verdict itself is qualified: an overlap that could not be
    established or excluded, or one decided by a surface placed on the separatrix law
  - `notes` -- per-surface and bookkeeping detail that does not qualify the verdict
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
    caveats::Vector{String}
    notes::Vector{String}
end

# Keyword constructor, so an empty scan names only the fields it sets.
LayerOverlapScan(; m=Int[], n=Int[], psi=Float64[], rs=Float64[], delta_s_m=Float64[],
    width_delta_s=Float64[], width_visco=Float64[], width_dr=Float64[], extrapolated=Bool[],
    psihigh_delta_s=nothing, psihigh_visco=nothing, psihigh_dr=nothing, psihigh=nothing,
    first_overlap=nothing, caveats=String[], notes=String[]) =
    LayerOverlapScan(m, n, psi, rs, delta_s_m, width_delta_s, width_visco, width_dr, extrapolated,
        psihigh_delta_s, psihigh_visco, psihigh_dr, psihigh, first_overlap, caveats, notes)

# An inverse equilibrium is solved on a prescribed boundary and carries its own q out to ψ = 1,
# so the scan reads the real q there instead of extrapolating, and searches to ψ = 1 rather than
# stopping short of a separatrix it does not have. The direct/EFIT path cannot: GPEC recomputes q
# by field-line tracing only out to psihigh.
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
q-extrema, so reverse shear is handled). Surfaces beyond the grid come from
[`separatrix_q_law`](@ref) when `extrapolate=true`, which is what lets the scan
recommend a domain *larger* than the one it was handed. That law diverges at the
X-point flux, which a reconstruction may place slightly outside its own boundary,
so rationals can lie at ψ > 1: inside the real separatrix but beyond the
equilibrium GPEC holds. Only the first of them is placed, to judge whether the
outermost surface inside is clean, and it is scored only if the kinetic profiles
extend that far. The recommendation never exceeds ψ = 1, and when no overlap can
be established the domain keeps its requested limit.

Layer widths come from [`slayer_algebraic_widths`](@ref) via
[`build_slayer_inputs`](@ref), so the scan uses the same plasma inputs and the
same resistivity closure as the SLAYER analysis itself. The criterion channel needs no
Riccati solve; `delta_s` is solved per surface as a reported comparison only.

# Arguments

  - `equil` -- a formed `PlasmaEquilibrium`
  - `profiles` -- `KineticProfiles` spanning the scanned ψ range

# Keyword arguments

  - `n_tor` -- toroidal mode number; surfaces are q = m/`n_tor`

  - `m_max` -- runaway backstop on the poloidal mode number (default `2000`), **not** the
    physics limit: the outward search should terminate within `psi_gap` of the separatrix
    flux, and if `m_max` is what stops it instead, the scan says so in `caveats`
    because "no overlap found" would then rest on a non-physical cutoff
  - `max_layer_solves` -- bound on how many surfaces get a Riccati layer solve (default `64`,
    innermost kept). The first overlap decides the answer, so surfaces far beyond it only cost
    time; when the bound bites, `caveats` records that a "no overlap" verdict is inconclusive
  - `psihigh_safe` -- treat this as the outer edge of trusted equilibrium data
    (default `equil.profiles.xs[end]`)
  - `psi_gap` -- never place a surface closer than this to the separatrix flux on the
    **direct** path (default `1e-9`, purely numerical: it stops the outward search from
    crowding indefinitely onto the separatrix where surfaces become indistinguishable).
    Inverse equilibria ignore it: they have a prescribed boundary and the search runs to
    ψ = 1 on the real q. Geometry needs no clamp of its own -- `radial_label` derivatives are
    analytic and valid under extrapolation
  - `extrapolate` -- search past `psihigh_safe` on the separatrix q-law (default `true`)
  - `theta` -- poloidal angle for the minor-radius chord (default `0.0`)
  - `rs_method` -- radial label the widths and their ψ-conversion Jacobian are taken in
    (default `:midplane`; see `radial_label`). The caller wiring this scan into the domain
    truncation passes `:flux`, the label Fitzpatrick Eq. (100) is anchored to
  - `bt`, `mu_i`, `zeff`, `chi_perp_e`, `chi_perp_i`, `chi_tor`, `tau_E`, `P_perp_model`, `resistivity_model`, `lnLambda_form`
    -- passed through to `build_slayer_inputs`. The defaults are placeholders: a caller
    scanning a real run must pass that run's own values, as the domain-truncation wiring does

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
    psi_gap::Real=1e-9,
    extrapolate::Bool=true,
    theta::Real=0.0,
    rs_method::Symbol=:midplane,
    bt=nothing,
    mu_i::Real=2.0,
    zeff=1.0,
    chi_perp_e=1.0,
    chi_perp_i=nothing,
    chi_tor=1.0,
    tau_E::Union{Real,Nothing}=nothing,
    P_perp_model::Symbol=:D_perp,
    resistivity_model::NeoResistivityModel=SauterNeoModel(),
    lnLambda_form::Symbol=:nrl)

    n_tor > 0 || throw(ArgumentError("resistive_layer_overlap: n_tor must be positive, got $n_tor"))
    psihigh_safe = Float64(psihigh_safe)
    caveats, notes = String[], String[]

    # In-grid surfaces for this n, ordered outward.
    found = [(m=s.m, psi=s.psifac, extrap=false)
             for s in _find_rational_surfaces(equil, Int(n_tor), Int(n_tor))
             if s.psifac <= psihigh_safe && s.m <= m_max]
    sort!(found; by=s -> s.psi)

    # Surfaces past the grid. An inverse equilibrium already knows its real q out to ψ = 1, so
    # it is read rather than modelled; only the direct path needs the edge law.
    edge_fit = nothing
    inv_q = (equil.ingest isa InverseIngest) ? _inverse_q_spline(equil.ingest) : nothing
    if extrapolate && inv_q !== nothing
        psi_top = 1.0                      # a prescribed boundary, not a separatrix
        m_start = isempty(found) ? 1 : maximum(s.m for s in found) + 1
        n_before = length(found)
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
            push!(caveats,
                "outward search stopped at the m_max = $m_max backstop rather than at the boundary " *
                "q; a \"no overlap\" result here is not conclusive -- raise m_max")
        length(found) > n_before &&
            push!(notes, "inverse equilibrium: surfaces beyond ψ = $(@sprintf("%.6f", psihigh_safe)) " *
                         "read from the real q profile out to ψ = 1 (no edge-law extrapolation)")
    elseif extrapolate
        edge_fit = separatrix_q_law(equil; psi_max=psihigh_safe)
        if edge_fit === nothing
            push!(
                notes,
                "no separatrix q-law: no X-point was found from the outermost flux surface, " *
                "or the law it gives does not reproduce the traced shear " *
                "there, so no surfaces are placed beyond ψ = $(@sprintf("%.6f", psihigh_safe))"
            )
        else
            # Every q = m/n above the anchor lies beyond the grid, so the search starts at the
            # first such m whatever the in-grid search returned: no surface can fall between
            # the two locators.
            m_start = floor(Int, n_tor * abs(edge_fit.q_anchor)) + 1
            m_grid = isempty(found) ? 0 : maximum(s.m for s in found)
            n_before = length(found)
            # Terminates within psi_gap of the separatrix flux -- surfaces indistinguishable from
            # it -- not on m_max. Reaching m_max means the scan was cut short for a non-physical
            # reason and any "no overlap" verdict from it is unsafe, so record that.
            reached_separatrix = false
            for m in m_start:Int(m_max)
                psi_m = separatrix_q_psi(edge_fit, sign(edge_fit.q_anchor) * m / n_tor; psi_gap=psi_gap)
                if psi_m === nothing
                    reached_separatrix = true
                    break
                end
                push!(found, (m=m, psi=psi_m, extrap=true))
                # One rational past the equilibrium's boundary is enough: it decides whether the
                # outermost surface inside is clean. Pairs further out lie wholly outside the
                # equilibrium and cannot constrain a domain that ends at or before ψ = 1.
                if psi_m > 1.0
                    reached_separatrix = true
                    break
                end
            end
            if !reached_separatrix && m_start <= m_max
                push!(caveats,
                    "outward search stopped at the m_max = $m_max backstop rather than at the " *
                    "separatrix; a \"no overlap\" result here is not conclusive -- raise m_max")
            end
            0 < m_grid < m_start - 1 &&
                push!(caveats,
                    "q = $(m_grid + 1)/$n_tor through $(m_start - 1)/$n_tor lie inside the grid but were " *
                    "not located there; an overlap between them may be missed")
            length(found) > n_before && last(found).psi > 1.0 &&
                push!(notes,
                    "q = $(last(found).m)/$n_tor lies beyond the equilibrium's own boundary (ψ = 1), inside its " *
                    "X-point flux ψ = $(@sprintf("%.6f", edge_fit.psi_x[1])); it is placed only to judge whether the " *
                    "outermost surface inside is clean, and needs kinetic profiles past ψ = 1 to be scored")
        end
    end

    if isempty(found)
        push!(notes, "no q = m/$n_tor surfaces found")
        return LayerOverlapScan(; caveats=caveats, notes=notes)
    end

    # Beyond the grid the equilibrium's q spline is a cubic extrapolation, so the radial label
    # takes q there from whichever source placed the surfaces.
    q_beyond = inv_q !== nothing ? (ψ -> Float64(inv_q(ψ))) :
               edge_fit !== nothing ? (ψ -> separatrix_q(edge_fit, ψ)) : nothing
    # Widths come back in metres of whichever radial label `rs_method` selects, so the
    # conversion into normalised flux must use that same label's Jacobian.
    _, _dr_dpsi = radial_label(equil; rs_method=rs_method, theta=theta, q_beyond=q_beyond)

    ms, ns, psis, rss = Int[], Int[], Float64[], Float64[]
    dels_m, w_dels, w_visc, w_dr, extraps = Float64[], Float64[], Float64[], Float64[], Bool[]
    unscored = Float64[]

    # Bound the per-surface Riccati work, and say so when the bound bites: a "no overlap"
    # verdict from a truncated list is not conclusive.
    if length(found) > max_layer_solves
        push!(caveats,
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
            separatrix_q_dqdpsi(edge_fit, s.psi)
        end
        da_dpsi = _dr_dpsi(s.psi)
        if !isfinite(da_dpsi) || da_dpsi == 0.0
            push!(notes, "m=$(s.m): da/dψ = $da_dpsi at ψ=$(@sprintf("%.6f", s.psi)); cannot convert width to flux units")
            push!(unscored, s.psi)
            continue
        end

        sing = SingType(; m=[s.m], n=[Int(n_tor)], psifac=s.psi, rho=sqrt(s.psi), q=q_val, q1=q1_val)
        # Built one surface at a time so a single degenerate surface (e.g. ω_*e == ω_*i,
        # which makes iota_e singular) is recorded and skipped rather than aborting the scan.
        params = try
            build_slayer_inputs(equil, [sing], profiles; rs_method=rs_method, q_beyond=q_beyond,
                bt=bt, mu_i=mu_i, zeff=zeff, chi_perp_e=chi_perp_e, chi_perp_i=chi_perp_i, chi_tor=chi_tor,
                tau_E=tau_E, P_perp_model=P_perp_model,
                dr_val=0.0, dc_type=:none, theta=theta,
                resistivity_model=resistivity_model, lnLambda_form=lnLambda_form)
        catch err
            # Any single-surface failure is recorded and skipped, never fatal. A DomainError is
            # the common one: profiles that stop short of the extrapolated surfaces, or a
            # negative-shear surface reaching a sqrt/fractional power in the layer parameters.
            err isa Union{ArgumentError,DomainError} || rethrow()
            msg = err isa ArgumentError ? err.msg : sprint(showerror, err)
            push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): skipped -- $(replace(msg, '\n' => ' '))")
            push!(unscored, s.psi)
            continue
        end

        # The criterion needs only the algebraic DR width, so the Riccati solve never gates it:
        # dropping a surface here would leave _first_overlap_limit comparing its neighbours across
        # a gap twice the physical one, which hides an overlap and reports too wide a domain.
        alg = slayer_algebraic_widths(params[1])
        if !isfinite(alg.delta_dr)
            push!(
                notes,
                "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): Eq. (100) width is not finite, so this " *
                "surface could not be scored"
            )
            push!(unscored, s.psi)
            continue
        end
        # delta_s is reported for comparison only. A failure leaves it NaN, which makes its own
        # channel report no overlap, and leaves the DR criterion untouched.
        delta_s_m = try
            w = abs(riccati_del_s(params[1]) * params[1].d_beta)
            isfinite(w) || push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): del_s Riccati did not converge; " *
                                        "delta_s channel only, criterion unaffected")
            w
        catch err
            err isa Union{ArgumentError,DomainError} || rethrow()
            push!(notes, "m=$(s.m) at ψ=$(@sprintf("%.6f", s.psi)): del_s Riccati failed; delta_s channel only, " *
                         "criterion unaffected")
            NaN
        end
        lw = (; delta_s_m=delta_s_m, delta_dr=alg.delta_dr, delta_visco=alg.delta_visco)

        # Metres → normalized flux, so widths and surface separations are comparable.
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

    ph_dels, _ = _first_overlap_limit(psis, w_dels)
    ph_visc, _ = _first_overlap_limit(psis, w_visc)
    ph_dr, k_dr = _first_overlap_limit(psis, w_dr)
    # The DR channel is the criterion (see the `width_dr` field above); the other two are
    # reported for comparison only.
    recommended = ph_dr
    # A surface that could not be scored hides any overlap with its neighbours, so it matters
    # wherever it sits inside the reported overlap point, or anywhere when none was found.
    n_hidden = count(p -> k_dr === nothing || p < psis[k_dr], unscored)
    n_hidden > 0 &&
        push!(caveats,
            k_dr === nothing ?
            "no overlap was established, with $n_hidden surface(s) that could not be scored (typically " *
            "kinetic profiles that stop short of them); the domain keeps its requested limit" :
            "$n_hidden surface(s) inside the reported overlap point could not be scored (typically " *
            "kinetic profiles that stop short of them); the first overlap may lie further in, so " *
            "this result is an outer bound")
    k_dr !== nothing && (extraps[k_dr] || extraps[k_dr-1]) &&
        push!(caveats,
            "the overlap point is decided by a surface placed on the separatrix law (q = $(ms[k_dr-1])/$n_tor " *
            "against $(ms[k_dr])/$n_tor), not by one traced inside the grid")

    return LayerOverlapScan(ms, ns, psis, rss, dels_m, w_dels, w_visc, w_dr, extraps,
        ph_dels, ph_visc, ph_dr, recommended, k_dr, caveats, notes)
end

# Walk outward and return (recommended psilim, index of first overlapping surface).
# Overlap at k contaminates both k and k-1, so the limit is surface k-1's own position
# (Fitzpatrick 2025 Sect. 5.9 retains 0 < psi < 1 - eps_c with psi[k-1] as 1 - eps_c);
# offsetting by ±w[k-1]/2 either drops the last clean surface or can exceed psi = 1.
function _first_overlap_limit(psi::Vector{Float64}, w::Vector{Float64})
    for k in 2:length(psi)
        if psi[k] - w[k] / 2 < psi[k-1] + w[k-1] / 2
            return (psi[k-1], k)
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

"""
    layer_overlap_scan(equil, model::SLAYERModel, ns) -> Union{Nothing,LayerOverlapScan}

The binding resistive-layer overlap scan over the toroidal mode numbers `ns`: the one with the
innermost overlap point, since the integration domain must stop before the first overlap of any n
(the first scan when none overlaps, so its surfaces are still recorded). The plasma is the
equilibrium's kinetic data and the layer closures are `model`'s, the same inputs the SLAYER
tearing solve uses. Eq. (100) of Fitzpatrick (2025) is anchored to the toroidal-flux label of its
Eq. (30), so the scan runs in that label whatever `model.theta_sample` samples. What qualifies a
scan's verdict is logged as a warning and the per-surface detail as info.
"""
function layer_overlap_scan(equil, model::SLAYERModel, ns)
    data = equil.kinetic
    data === nothing && error("psilim_from_layer_overlap needs kinetic profiles: set kinetic_file in [Equilibrium] or attach_kinetic_profiles!")
    loaded = _slayer_profiles(data, model)
    chi = _slayer_chi_inputs(loaded, model, data; warn=false)
    zeff = loaded.zeff === nothing ? data.zeff_default : loaded.zeff
    scans = [
        resistive_layer_overlap(equil, loaded.profiles; n_tor=n, rs_method=:flux, chi..., tau_E=data.tau_E,
            P_perp_model=model.P_perp_model, mu_i=_layer_ion_mass(data), zeff=zeff, theta=model.theta_sample,
            resistivity_model=_build_resistivity_model(data.resistivity_model), lnLambda_form=data.lnLambda_form)
        for n in ns
    ]
    for (n, scan) in zip(ns, scans)
        foreach(msg -> @warn("Layer-overlap scan (n = $n): $msg"), scan.caveats)
        foreach(msg -> @info("Layer-overlap scan (n = $n): $msg"), scan.notes)
    end
    isempty(scans) && return nothing
    overlapping = filter(sc -> sc.psihigh !== nothing, scans)
    isempty(overlapping) && return first(scans)
    return overlapping[argmin([sc.psihigh for sc in overlapping])]
end
