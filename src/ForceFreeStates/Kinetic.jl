"""
    certified_kinetic_grid(seed, evaluate, ideal_scales, tol, rationals; kwargs...)
        -> (xs, kw_flat, kt_flat)

Choose the ψ knots for the calculated kinetic matrices by batched certify-or-refine rounds, so the
expensive kernel runs only where the **total** (ideal + kinetic) matrices demand it.

`evaluate(psis) -> (kw, kt)` runs the threaded kernel on a batch of ψ values. Starting from `seed`
(the ideal coefficient-spline knots), each round splines the kinetic increments on the current
knots, proposes the midpoints of all uncertified intervals, evaluates the whole batch, and
classifies: a midpoint whose predicted increments match the fresh evaluation within
`tol · scale` for **every element of every consumed family** certifies its interval and is
discarded (a knot the spline already predicts only adds noise-scale curvature); otherwise it
becomes a knot and its sub-intervals join the queue. The families are the six kinetic components
plus the non-Hermitian adjoint combination `kw₃ − kt₃`, and each `scale` is the largest magnitude
of the corresponding **total** matrix over the seed — so the tolerance is held on what the
Euler-Lagrange solver actually consumes, never on the increments in isolation.

Intervals are never refined below a floor of `0.05·ψ` in the Frobenius core or
`RATIONAL_RES_SPACING` outside it, and refinement stops after `max_rounds`. The returned knot set
always contains the seed, so structure resolved by the ideal grid is retained.
"""
function certified_kinetic_grid(seed::Vector{Float64}, evaluate::Function,
    ideal_scales::NTuple{6,Float64}, tol::Float64, rationals::Vector{Float64};
    max_rounds::Int=6, verbose::Bool=true)

    xs = copy(seed)
    kw, kt = evaluate(xs)
    nevals = length(xs)
    np2 = size(kw, 2)

    # Tolerance scale per family: the total matrix the solver sees (ideal + increments over seed).
    scales = ntuple(ic -> max(ideal_scales[ic],
            maximum(abs, @view(kw[:, :, ic])) + maximum(abs, @view(kt[:, :, ic]))), 6)

    uncertified = trues(length(xs) - 1)
    # Spacing floors: the Frobenius-region cap in the core, RATIONAL_RES_SPACING elsewhere — and a
    # finer floor inside rational windows, so a narrow resonance layer cannot hide behind the
    # coarse floor exactly where layers are expected (the anti-aliasing rule).
    near_rational(x) = any(abs(x - r) <= Equilibrium.RATIONAL_RES_RADIUS for r in rationals)
    floor_at(x) = near_rational(x) ? Equilibrium.RATIONAL_RES_SPACING / 4 :
                  (x < 0.1 ? 0.05 * x : Equilibrium.RATIONAL_RES_SPACING)
    added = 0
    for round in 1:max_rounds
        props = Float64[]
        slots = Int[]
        for i in findall(uncertified)
            m = 0.5 * (xs[i] + xs[i+1])
            if xs[i+1] - xs[i] < 2 * floor_at(m)
                uncertified[i] = false      # certified by the spacing floor
                continue
            end
            push!(props, m)
            push!(slots, i)
        end
        isempty(props) && break
        kwp, ktp = evaluate(props)
        nevals += length(props)

        # Spline each increment family on the current knots to predict the midpoints.
        pred_kw = [cubic_interp(xs, Series(@view(kw[:, :, ic]))) for ic in 1:6]
        pred_kt = [cubic_interp(xs, Series(@view(kt[:, :, ic]))) for ic in 1:6]
        buf = Vector{ComplexF64}(undef, np2)

        fails = Int[]
        for (k, m) in pairs(props)
            ok = true
            for ic in 1:6
                pred_kw[ic](buf, m)
                r = maximum(abs(buf[j] - kwp[k, j, ic]) for j in 1:np2)
                pred_kt[ic](buf, m)
                r = max(r, maximum(abs(buf[j] - ktp[k, j, ic]) for j in 1:np2))
                if r > tol * scales[ic]
                    ok = false
                    break
                end
            end
            if ok  # adjoint combination kw3 - kt3, held against the C-total scale
                pred_kw[3](buf, m)
                a = copy(buf)
                pred_kt[3](buf, m)
                r = maximum(abs((a[j] - buf[j]) - (kwp[k, j, 3] - ktp[k, j, 3])) for j in 1:np2)
                ok = r <= tol * scales[3]
            end
            if ok
                uncertified[slots[k]] = false
            else
                push!(fails, k)
            end
        end
        isempty(fails) && (fill!(uncertified, false); break)

        # Insert failed midpoints as knots (batch merge, preserving order).
        newxs = Float64[]
        newkw = zeros(ComplexF64, length(xs) + length(fails), np2, 6)
        newkt = zeros(ComplexF64, length(xs) + length(fails), np2, 6)
        newunc = Bool[]
        fi = 1
        row = 0
        for i in eachindex(xs)
            row += 1
            push!(newxs, xs[i])
            newkw[row, :, :] .= kw[i, :, :]
            newkt[row, :, :] .= kt[i, :, :]
            i == length(xs) && break
            inserted = false
            while fi <= length(fails) && slots[fails[fi]] == i
                k = fails[fi]
                row += 1
                push!(newxs, props[k])
                newkw[row, :, :] .= kwp[k, :, :]
                newkt[row, :, :] .= ktp[k, :, :]
                inserted = true
                fi += 1
            end
            if inserted
                push!(newunc, true, true)   # both halves of the split interval re-enter the queue
            else
                push!(newunc, uncertified[i])
            end
        end
        added += length(fails)
        xs = newxs
        kw = newkw[1:row, :, :]
        kt = newkt[1:row, :, :]
        uncertified = BitVector(newunc)
    end
    @info "Certified kinetic grid: $(length(seed)) seed -> $(length(xs)) knots " *
          "($nevals kernel evaluations, $added refined, tol=$tol)"
    return xs, kw, kt
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
    calculated_source::Union{Nothing,Function}=nothing
)
    xs = metric.xs
    mpsi = length(xs)

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
        if ctrl.kinetic_grid_tol > 0
            # Certified adaptive grid: seed with the ideal coefficient-spline knots and refine
            # until the total matrices are spline-predictable to kinetic_grid_tol everywhere.
            # Seed with a coarse skeleton of the ideal coefficient-spline knots: the kernel is the
            # expensive part, so the seed must not scale with the equilibrium grid. Every ~seed
            # gap the certificate cannot vouch for is refined, so coarse seeding trades cheap
            # certificates for expensive blanket evaluation. Endpoints and rational-window knots
            # are always retained.
            base = isempty(ffit.matrix_xs) ? metric.xs : ffit.matrix_xs
            seed_max = 49
            if length(base) > seed_max
                rats = [sng.psifac for sng in intr.sing]
                keep_idx = falses(length(base))
                keep_idx[1] = keep_idx[end] = true
                stride = max(1, (length(base) - 1) ÷ (seed_max - 1))
                keep_idx[1:stride:end] .= true
                for (i, x) in pairs(base)
                    any(abs(x - r) <= Equilibrium.RATIONAL_RES_RADIUS for r in rats) && (keep_idx[i] = true)
                end
                seed = base[keep_idx]
            else
                seed = copy(base)
            end
            hint = Ref(1)
            ideal_scales = ntuple(ic -> begin
                    sp = (ffit.amats, ffit.bmats, ffit.cmats, ffit.dmats_prim, ffit.emats_prim, ffit.hmats)[ic]
                    maximum(maximum(abs, sp(x; hint=hint)) for x in seed)
                end, 6)
            rationals = [sng.psifac for sng in intr.sing]
            xs, kw_flat, kt_flat = certified_kinetic_grid(seed,
                psis -> calculated_source(ctrl, equil, intr, metric, ffit; psis=psis),
                ideal_scales, ctrl.kinetic_grid_tol, rationals; verbose=ctrl.verbose)
            mpsi = length(xs)
        else
            kw_flat, kt_flat = calculated_source(ctrl, equil, intr, metric, ffit)
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
    _compute_fkg_matrices!(ffit, equil, intr, metric, kw_flat, kt_flat)

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
