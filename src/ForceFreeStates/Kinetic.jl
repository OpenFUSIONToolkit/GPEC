
"""
    bisect_refine_kinetic_grid(xs, kw, kt, evaluate, tol, psi_c) → (xs, kw, kt)

One bounded bisection-refinement round for the calculated kinetic matrix grid (experimental).
Every interval midpoint above `2·psi_c` (outside the near-axis validity band) is kernel-evaluated
in one threaded batch; for each family the increment spline on the current knots predicts the
midpoint, and the max-element residual is normalized by a **per-interval local scale** (the max
increment magnitude over the interval's endpoints and midpoint, floored at 1% of the family's
global maximum — the local scale is the lesson from the parked global-scale certificate, the
floor keeps near-zero crossings from flagging on noise). Midpoints whose worst-family residual
exceeds `tol` become knots; their kernel values are reused, so the round costs exactly one extra
kernel pass regardless of how many knots it inserts. The residual distribution is logged
unconditionally — the map of what the ideal-driven grid misses is the diagnostic even when
nothing is inserted.
"""
function bisect_refine_kinetic_grid(xs::Vector{Float64}, kw::Array{ComplexF64,3}, kt::Array{ComplexF64,3},
    evaluate::Function, tol::Float64, psi_c::Float64)
    lo = 2 * psi_c
    cand = [i for i in 1:(length(xs)-1) if xs[i] >= lo]
    isempty(cand) && return xs, kw, kt
    mids = [0.5 * (xs[i] + xs[i+1]) for i in cand]
    kw_m, kt_m = evaluate(mids)
    np2 = size(kw, 2)
    worst = zeros(length(mids))
    buf = Vector{ComplexF64}(undef, np2)
    for ic in 1:6
        gscale_w = maximum(abs, @view(kw[:, :, ic]))
        gscale_t = maximum(abs, @view(kt[:, :, ic]))
        sp_w = cubic_interp(xs, Series(@view(kw[:, :, ic])))
        sp_t = cubic_interp(xs, Series(@view(kt[:, :, ic])))
        for (k, m) in pairs(mids)
            i = cand[k]
            for (sp, arr, arrm, gs) in ((sp_w, kw, kw_m, gscale_w), (sp_t, kt, kt_m, gscale_t))
                gs == 0 && continue
                sp(buf, m)
                scale = max(maximum(abs, @view(arr[i, :, ic])), maximum(abs, @view(arr[i+1, :, ic])),
                    maximum(abs, @view(arrm[k, :, ic])), 0.01 * gs)
                r = maximum(abs(buf[j] - arrm[k, j, ic]) for j in 1:np2) / scale
                worst[k] = max(worst[k], r)
            end
        end
    end
    q = [round(sort(worst)[max(1, ceil(Int, f * length(worst)))]; sigdigits=2) for f in (0.5, 0.9, 0.99, 1.0)]
    ins = findall(>(tol), worst)
    @info "Kinetic bisection map: $(length(mids)) midpoints probed, residual quantiles " *
          "(50/90/99/100%) = $q; $(length(ins)) exceed tol=$tol" *
          (isempty(ins) ? "" : " at ψ=$(round.(mids[ins]; digits=4))")
    isempty(ins) && return xs, kw, kt
    keep = sort(ins)
    nnew = length(xs) + length(keep)
    xs2 = Vector{Float64}(undef, nnew)
    kw2 = Array{ComplexF64,3}(undef, nnew, np2, 6)
    kt2 = Array{ComplexF64,3}(undef, nnew, np2, 6)
    row = 0
    ki = 1
    for i in eachindex(xs)
        row += 1
        xs2[row] = xs[i]
        kw2[row, :, :] .= kw[i, :, :]
        kt2[row, :, :] .= kt[i, :, :]
        while ki <= length(keep) && cand[keep[ki]] == i
            row += 1
            xs2[row] = mids[keep[ki]]
            kw2[row, :, :] .= kw_m[keep[ki], :, :]
            kt2[row, :, :] .= kt_m[keep[ki], :, :]
            ki += 1
        end
    end
    return xs2, kw2, kt2
end

"""
    make_kinetic_matrix(ctrl, equil, ffit, intr, metric;
                        calculated_source=nothing)

Construct kinetic energy (W) and torque (T) matrices, store as splines in `ffit`,
and pre-compute the FKG derived matrices used by `sing_der!`.

Dispatches on `ctrl.kinetic_source`:

  - `"fixed"`: X-shaped test matrices scaled by `ctrl.kinetic_factor` relative to
    ideal matrix Frobenius norms (Ak, Dk, Hk Hermitian; Bk, Ck, Ek non-Hermitian)
  - `"calculated"`: Compute via the `calculated_source` callback. This is
    expected to be `KineticForces.compute_calculated_kinetic_matrices` injected
    by `GeneralizedPerturbedEquilibrium.main`. The callback receives
    `(ctrl, equil, intr, metric, ffit)` and returns `(kw_flat, kt_flat)` of
    shape `(mpsi, np^2, 6)`. Callback injection is used because ForceFreeStates
    is loaded before KineticForces, so a direct import would invert the
    dependency order.

Both paths apply `ctrl.kinetic_factor` as a global scale before the FKG Schur
reduction.
"""
function make_kinetic_matrix(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal,
    metric::MetricData;
    calculated_source::Union{Nothing,Function}=nothing,
    axis_validity_psi_c::Float64=0.0,
    resonance_psis::Vector{Float64}=Float64[]
)
    xs = metric.xs
    mpsi = length(xs)

    # The near-axis validity envelope (KineticForces) has structure on the scale of the
    # suppression boundary; coarse equilibrium grids cannot represent env·(increment), and the
    # spline overshoot can land on a rational surface. Pin the band ends (the smoothstep is
    # only C² there) and resolve the transition with a fixed set of knots.
    # Resonant-layer ladder: the kinetic matrices are near-singular at the located Ω_ℓ = 0
    # surfaces (measured core width ~2e-3 at the DIII-D pedestal ω_E crossing), so the kinetic
    # evaluation grid gets a local knot ladder at each node — kinetic splines only; the
    # equilibrium/Δ′ grids are untouched.
    layer_offsets = [0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.0045, 0.006, 0.008]
    function layer_knots(lo, hi)
        out = Float64[]
        for r in resonance_psis
            r <= max(lo, 2 * axis_validity_psi_c) && continue
            r >= hi && continue
            push!(out, r)
            for w in layer_offsets, sgn in (-1, 1)
                x = r + sgn * w
                lo < x < hi && push!(out, x)
            end
        end
        return out
    end
    band_knots(lo, hi) = axis_validity_psi_c > 0 ?
                         [x for x in range(axis_validity_psi_c, 2 * axis_validity_psi_c; length=9) if lo < x < hi] : Float64[]

    # Get raw kinetic matrices (scaling is baked into each source)
    if ctrl.kinetic_source == "fixed"
        kw_flat, kt_flat = fixed_kinetic_matrices(intr.mpert, mpsi, ctrl.kinetic_factor, intr.mlow, ffit, xs)
    elseif ctrl.kinetic_source == "calculated"
        isnothing(calculated_source) && error(
            "kinetic_source=\"calculated\" requires the KineticForces callback. " *
            "Drive the run via `GeneralizedPerturbedEquilibrium.main` instead of " *
            "calling make_kinetic_matrix directly, or pass " *
            "`calculated_source=KineticForces.compute_calculated_kinetic_matrices` explicitly."
        )
        extra = vcat(band_knots(xs[1], xs[end]), layer_knots(xs[1], xs[end]))
        if isempty(extra)
            kw_flat, kt_flat = calculated_source(ctrl, equil, intr, metric, ffit)
        else
            xs = collect(xs)
            for x in sort!(extra)   # snap-guard: skip knots that would ring against existing ones
                i = searchsortedfirst(xs, x)
                ((i > 1 && x - xs[i-1] < 1e-4) || (i <= length(xs) && xs[i] - x < 1e-4)) && continue
                insert!(xs, i, x)
            end
            mpsi = length(xs)
            isempty(resonance_psis) ||
                @info "Kinetic evaluation grid: resonant-layer ladders at $(round.(filter(r -> r > 2 * axis_validity_psi_c, resonance_psis); digits=3)) -> $mpsi knots"
            kw_flat, kt_flat = calculated_source(ctrl, equil, intr, metric, ffit; psis=xs)
        end
        if ctrl.kinetic_grid_bisect > 0
            xs, kw_flat, kt_flat = bisect_refine_kinetic_grid(
                collect(xs), kw_flat, kt_flat,
                psis -> calculated_source(ctrl, equil, intr, metric, ffit; psis=psis),
                ctrl.kinetic_grid_bisect, axis_validity_psi_c)
            mpsi = length(xs)
        end
        kw_flat .*= ctrl.kinetic_factor
        kt_flat .*= ctrl.kinetic_factor
    else
        error("Unknown kinetic_source: $(ctrl.kinetic_source). Must be \"fixed\" or \"calculated\"")
    end

    # Build splines for each of the 6 components
    for ic in 1:6
        ffit.kwmats[ic] = cubic_interp(xs, Series(@view(kw_flat[:, :, ic])); ffit.itp_opts...)
        ffit.ktmats[ic] = cubic_interp(xs, Series(@view(kt_flat[:, :, ic])); ffit.itp_opts...)
    end

    # Pre-compute FKG derived matrices (corresponds to Fortran method=0)
    _compute_fkg_matrices!(ffit, equil, intr, metric, kw_flat, kt_flat; xs=xs)

    return nothing
end

"""
    _compute_fkg_matrices!(ffit, equil, intr, metric, kw_flat, kt_flat; xs=xs)

Pre-compute the derived F, K, G kinetic matrices at each ψ grid point and store as splines.
This corresponds to `fourfit_kinetic_matrix` method=0 in the Fortran code (Fortran `fourfit.F` lines 1170-1260).

The 9 matrices computed are the Schur complement reductions of ideal (A,B,C,D,E,H) and kinetic (W,T)
primitives into the F, K, G sub-matrices of the non-Hermitian kinetic ODE system (Logan 2015 Appendix C,
Eqs C.1-C.11). These sub-matrices absorb the singfac dependence so that the ODE RHS in `sing_der!`
can be assembled with explicit (m-nq) factors rather than 1/(m-nq), avoiding numerical blow-up at
rational surfaces.
"""
function _compute_fkg_matrices!(
    ffit::FourFitVars,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    metric::MetricData,
    kw_flat::Array{ComplexF64,3},
    kt_flat::Array{ComplexF64,3};
    xs::Vector{Float64}=metric.xs
)
    mpsi = length(xs)
    np = intr.numpert_total
    mpert = intr.mpert
    npert = intr.npert

    # Allocate output arrays — kinetic-modified A/B/C stored for sing_der! FKG path
    ak_flat = zeros(ComplexF64, mpsi, np^2)
    bk_flat = zeros(ComplexF64, mpsi, np^2)
    ck_flat = zeros(ComplexF64, mpsi, np^2)
    f0_flat = zeros(ComplexF64, mpsi, np^2)
    p_flat = zeros(ComplexF64, mpsi, np^2)
    pa_flat = zeros(ComplexF64, mpsi, np^2)
    kk_flat = zeros(ComplexF64, mpsi, np^2)
    kka_flat = zeros(ComplexF64, mpsi, np^2)
    r1_flat = zeros(ComplexF64, mpsi, np^2)
    r2_flat = zeros(ComplexF64, mpsi, np^2)
    r3_flat = zeros(ComplexF64, mpsi, np^2)
    ga_flat = zeros(ComplexF64, mpsi, np^2)

    # ψ-loop is embarrassingly parallel: each iteration writes to a unique
    # ipsi row of the *_flat output arrays. The only thread-shared mutable is
    # the interpolant bracket-search hint; give each thread its own Ref.
    thread_hints = [Ref(1) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for ipsi in 1:mpsi
        hint = thread_hints[Threads.threadid()]
        psi = xs[ipsi]

        # Evaluate ideal and kinetic matrices from splines (full np×np, block-diagonal in n)
        amat_full = reshape(ffit.amats(psi; hint=hint), np, np)
        bmat_full = reshape(ffit.bmats(psi; hint=hint), np, np)
        cmat_full = reshape(ffit.cmats(psi; hint=hint), np, np)
        dmat_full = reshape(ffit.dmats_prim(psi; hint=hint), np, np)
        emat_full = reshape(ffit.emats_prim(psi; hint=hint), np, np)
        hmat_full = reshape(ffit.hmats(psi; hint=hint), np, np)
        fmat_prim_full = reshape(ffit.fmats_prim(psi; hint=hint), np, np)

        kwmat_full = zeros(ComplexF64, np, np, 6)
        ktmat_full = zeros(ComplexF64, np, np, 6)
        for ic in 1:6
            kwmat_full[:, :, ic] .= reshape(@view(kw_flat[ipsi, :, ic]), np, np)
            ktmat_full[:, :, ic] .= reshape(@view(kt_flat[ipsi, :, ic]), np, np)
        end

        # Process each n-block independently (matrices are block-diagonal in n)
        for in_idx in 1:npert
            n = intr.nlow + in_idx - 1
            rng = ((in_idx-1)*mpert+1):(in_idx*mpert)

            # Extract n-block slices
            amat = amat_full[rng, rng]
            bmat = bmat_full[rng, rng]
            cmat = cmat_full[rng, rng]
            dmat = dmat_full[rng, rng]
            emat = emat_full[rng, rng]
            hmat = hmat_full[rng, rng]
            fmat_prim = fmat_prim_full[rng, rng]
            kwmat = zeros(ComplexF64, mpert, mpert, 6)
            ktmat = zeros(ComplexF64, mpert, mpert, 6)
            for ic in 1:6
                kwmat[:, :, ic] .= kwmat_full[rng, rng, ic]
                ktmat[:, :, ic] .= ktmat_full[rng, rng, ic]
            end

            # Add kinetic contributions to ideal matrices [Fortran fourfit.F lines 1153-1158]
            amat_kin = amat .+ kwmat[:, :, 1] .+ ktmat[:, :, 1]
            bmat_kin = bmat .+ kwmat[:, :, 2] .+ ktmat[:, :, 2]
            cmat_kin = cmat .+ kwmat[:, :, 3] .+ ktmat[:, :, 3]
            hmat_kin = hmat .+ kwmat[:, :, 6] .+ ktmat[:, :, 6]
            caat = cmat_kin .- 2 .* ktmat[:, :, 3]  # C† analog for non-Hermitian system

            # b1mat = i*D (Fortran convention)
            b1mat = im .* dmat

            # LU factorization of kinetic A matrix (non-Hermitian)
            amat_lu = lu(amat_kin)

            # f0mat = F_prim - D†A_kin⁻¹D  [Fortran fourfit.F line 1184]
            temp1 = amat_lu \ dmat
            f0mat = fmat_prim .- dmat' * temp1

            # pmat [Fortran lines 1193-1200]
            psio_over_n = equil.psio / n  # chi1/(2πn) = psio/n — toroidal flux per mode number
            bkmat = kwmat[:, :, 2] .+ ktmat[:, :, 2] .+ im * psio_over_n .* (kwmat[:, :, 1] .+ ktmat[:, :, 1])
            bkaat = kwmat[:, :, 2] .- ktmat[:, :, 2] .+ im * psio_over_n .* (kwmat[:, :, 1] .+ ktmat[:, :, 1])
            temp2 = amat_lu \ bkmat
            pmat_val = b1mat' * temp2

            # paat [Fortran lines 1202-1207]
            temp2 = amat_lu \ b1mat
            # Fortran sing.f:1004-1008 computes aamat = amat_kin^H · A_kin⁻¹ via
            #   zgbtrs("C", ..., amatlu, temp2=amat_kin)  → temp2 = A_kin^{-H} · amat_kin
            #   aamat = CONJG(TRANSPOSE(temp2)) = amat_kin^H · A_kin⁻¹
            # For non-Hermitian amat_kin (kwmat Hermitian + ktmat anti-Hermitian),
            # this is NOT the identity. The prior implementation `(amat_lu \ amat_kin)'`
            # gave aamat = I exactly, zeroing umat_diff and dropping the
            # `im·psio_over_n · umat_diff · ...` terms from paat, r1mat, r2mat.
            aamat_temp = amat_lu' \ amat_kin      # = A_kin^{-H} · amat_kin
            aamat = aamat_temp'                   # = amat_kin^H · A_kin⁻¹
            umat_diff = I - aamat
            paat_val = (bkaat' * temp2 .- im * psio_over_n .* umat_diff * b1mat)'

            # r1mat [Fortran lines 1209-1217]
            temp1_r1 = kwmat[:, :, 1] .+ ktmat[:, :, 1]
            temp2 = amat_lu \ bkmat
            r1mat_val =
                kwmat[:, :, 4] .+ ktmat[:, :, 4] .-
                psio_over_n^2 .* temp1_r1' .+
                im * psio_over_n .* bkaat' .-
                im * psio_over_n .* aamat * bkmat .-
                bkaat' * temp2

            # kkmat [Fortran lines 1220-1223]
            temp1 = amat_lu \ cmat_kin
            kkmat_val = emat .- b1mat' * temp1

            # kkaat [Fortran lines 1225-1229]
            temp1 = amat_lu \ b1mat
            kkaat_val = emat' .- caat' * temp1

            # r2mat [Fortran lines 1231-1237]
            temp1_r2 = kwmat[:, :, 5] .+ ktmat[:, :, 5] .- im * psio_over_n .* (kwmat[:, :, 3] .+ ktmat[:, :, 3])
            temp2 = amat_lu \ cmat_kin
            r2mat_val = temp1_r2 .+ im * psio_over_n .* umat_diff * cmat_kin .- bkaat' * temp2

            # r3mat [Fortran lines 1239-1245]
            temp1_r3 = kwmat[:, :, 5] .- ktmat[:, :, 5] .- im * psio_over_n .* (kwmat[:, :, 3] .- ktmat[:, :, 3])
            temp2 = amat_lu \ bkmat
            r3mat_val = temp1_r3' .- caat' * temp2

            # gaat [Fortran lines 1248-1251]
            temp2 = amat_lu \ cmat_kin
            gaat_val = hmat_kin .- caat' * temp2

            # Store n-block results into full flat arrays
            for (j_local, j_global) in enumerate(rng)
                col_offset = (j_global - 1) * np
                for (i_local, i_global) in enumerate(rng)
                    idx = i_global + col_offset
                    ak_flat[ipsi, idx] = amat_kin[i_local, j_local]
                    bk_flat[ipsi, idx] = bmat_kin[i_local, j_local]
                    ck_flat[ipsi, idx] = cmat_kin[i_local, j_local]
                    f0_flat[ipsi, idx] = f0mat[i_local, j_local]
                    p_flat[ipsi, idx] = pmat_val[i_local, j_local]
                    pa_flat[ipsi, idx] = paat_val[i_local, j_local]
                    kk_flat[ipsi, idx] = kkmat_val[i_local, j_local]
                    kka_flat[ipsi, idx] = kkaat_val[i_local, j_local]
                    r1_flat[ipsi, idx] = r1mat_val[i_local, j_local]
                    r2_flat[ipsi, idx] = r2mat_val[i_local, j_local]
                    r3_flat[ipsi, idx] = r3mat_val[i_local, j_local]
                    ga_flat[ipsi, idx] = gaat_val[i_local, j_local]
                end
            end
        end
    end

    # Build FKG splines
    ffit.f0mats = cubic_interp(xs, Series(f0_flat); ffit.itp_opts...)
    ffit.pmats = cubic_interp(xs, Series(p_flat); ffit.itp_opts...)
    ffit.paats = cubic_interp(xs, Series(pa_flat); ffit.itp_opts...)
    ffit.kkmats = cubic_interp(xs, Series(kk_flat); ffit.itp_opts...)
    ffit.kkaats = cubic_interp(xs, Series(kka_flat); ffit.itp_opts...)
    ffit.r1mats = cubic_interp(xs, Series(r1_flat); ffit.itp_opts...)
    ffit.r2mats = cubic_interp(xs, Series(r2_flat); ffit.itp_opts...)
    ffit.r3mats = cubic_interp(xs, Series(r3_flat); ffit.itp_opts...)
    ffit.gaats = cubic_interp(xs, Series(ga_flat); ffit.itp_opts...)

    # Preserve ideal A/B/C splines before overwrite
    ffit.amats_ideal = ffit.amats
    ffit.bmats_ideal = ffit.bmats
    ffit.cmats_ideal = ffit.cmats

    # Overwrite ideal A/B/C splines with kinetic-modified versions for sing_der!
    ffit.amats = cubic_interp(xs, Series(ak_flat); ffit.itp_opts...)
    ffit.bmats = cubic_interp(xs, Series(bk_flat); ffit.itp_opts...)
    ffit.cmats = cubic_interp(xs, Series(ck_flat); ffit.itp_opts...)
    ffit.kinetic_populated = true

    return nothing
end
