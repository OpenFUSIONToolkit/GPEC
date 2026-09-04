
# Knots laid across the envelope's [ψ_c, 2ψ_c] transition. The envelope is a quintic smoothstep,
# so a cubic spline needs several interior knots to follow it without overshoot; nine keeps the
# residual below the kernel's own tolerance even on coarse decks.
const BAND_KNOTS = 9

"""
    refine_grid_at_fbar_peaks(xs, kw, kt, evaluate, kin, equil, intr, psi_c;
                              ngrid=1000, relaxed_frac=0.01, target=3, max_add=24) → (xs, kw, kt)

Insert kinetic evaluation knots across near-singular structure of F̄ that the grid does not
resolve. Scans cond(F̄) (the same operator `find_kinetic_singular_surfaces!` searches — Park &
Logan Eq. 70, so shifted and split resonances are included), takes peaks between
`relaxed_frac`·threshold and the singular threshold, measures each peak's FWHM, and adds knots
only where fewer than `target` knots lie inside it. New points respect `MIN_KNOT_SPACING`, stay
above the near-axis validity band, and are capped at `max_add`; each costs one kernel evaluation
and the existing values are reused. A well-resolved grid inserts nothing.
"""
function refine_grid_at_fbar_peaks(xs::Vector{Float64}, kw::Array{ComplexF64,3}, kt::Array{ComplexF64,3},
    evaluate::Function, kin::KineticMatrices, equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal, psi_c::Float64;
    ngrid::Int=1000, relaxed_frac::Float64=KINETIC_RELAXED_FRAC, target::Int=3, max_add::Int=24,
    cond_threshold::Float64=KINETIC_SINGULAR_COND)

    lo, hi = xs[1], xs[end]
    scan = collect(range(lo, hi; length=ngrid))
    hint = Ref(1)
    cond_vals = [
        try
            evaluate_fbar_condition(x, kin, equil, intr; hint=hint)
        catch
            Inf
        end for x in scan
    ]

    add = Float64[]
    for i in 2:(ngrid-1)
        c = cond_vals[i]
        (c > cond_vals[i-1] && c > cond_vals[i+1] && relaxed_frac * cond_threshold < c <= cond_threshold) || continue
        l = i
        while l > 1 && cond_vals[l] > c / 2
            l -= 1
        end
        r = i
        while r < ngrid && cond_vals[r] > c / 2
            r += 1
        end
        inside = count(x -> scan[l] <= x <= scan[r], xs)
        inside >= target && continue
        w = (scan[r] - scan[l]) / 3
        for x in (scan[i], scan[i] - w, scan[i] + w)
            (lo < x < hi && x > 2 * psi_c) || continue
            any(y -> abs(y - x) < Equilibrium.MIN_KNOT_SPACING, xs) && continue
            any(y -> abs(y - x) < Equilibrium.MIN_KNOT_SPACING, add) && continue
            push!(add, x)
        end
    end
    isempty(add) && return xs, kw, kt
    length(add) > max_add && (add = sort(add)[1:max_add])

    sort!(add)
    @info "Kinetic grid: $(length(add)) knot(s) added across unresolved near-singular F̄ structure at " *
          "ψ=$(round.(add; digits=4)) (cond peaks below the singular threshold)"
    kw_new, kt_new = evaluate(add)
    allxs = vcat(xs, add)
    perm = sortperm(allxs)
    kw_all = cat(kw, kw_new; dims=1)[perm, :, :]
    kt_all = cat(kt, kt_new; dims=1)[perm, :, :]
    return allxs[perm], kw_all, kt_all
end

"""
    build_kinetic_matrix_splines(ctrl, equil, mats, intr, metric;
                        calculated_source=nothing)

Construct kinetic energy (W) and torque (T) matrices and pre-compute the FKG derived
matrices used by `sing_der!`, returning a new `MatrixSplines` carrying both alongside the
ideal matrices of the input `mats`.

Dispatches on `ctrl.kinetic_source`:

  - `"fixed"`: X-shaped test matrices scaled by `ctrl.kinetic_factor` relative to
    ideal matrix Frobenius norms (Ak, Dk, Hk Hermitian; Bk, Ck, Ek non-Hermitian)
  - `"calculated"`: Compute via the `calculated_source` callback. This is
    expected to be `KineticForces.compute_calculated_kinetic_matrices` injected
    by `GeneralizedPerturbedEquilibrium.main`. The callback receives
    `(ctrl, equil, intr, metric, mats)` and returns `(kw_flat, kt_flat)` of
    shape `(mpsi, np^2, 6)`. Callback injection is used because ForceFreeStates
    is loaded before KineticForces, so a direct import would invert the
    dependency order.

Both paths apply `ctrl.kinetic_factor` as a global scale before the FKG Schur
reduction.
"""
function build_kinetic_matrix_splines(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    metric::MetricData;
    calculated_source::Union{Nothing,Function}=nothing,
    axis_validity_psi_c::Float64=0.0
)
    xs = metric.xs
    mpsi = length(xs)

    # The near-axis validity envelope (KineticForces) has structure on the scale of the
    # suppression boundary; coarse equilibrium grids cannot represent env·(increment), and the
    # spline overshoot can land on a rational surface. Pin the band ends (the smoothstep is
    # only C² there) and resolve the transition with a fixed set of knots.
    band_knots(lo, hi) = axis_validity_psi_c > 0 ?
                         [x for x in range(axis_validity_psi_c, 2 * axis_validity_psi_c; length=BAND_KNOTS) if lo < x < hi] : Float64[]

    # Get raw kinetic matrices (scaling is baked into each source)
    if ctrl.kinetic_source == "fixed"
        kw_flat, kt_flat = fixed_kinetic_matrices(intr.mpert, intr.numpert_total, mpsi, ctrl.kinetic_factor, intr.mlow, mats, xs)
    elseif ctrl.kinetic_source == "calculated"
        isnothing(calculated_source) && error(
            "kinetic_source=\"calculated\" requires the KineticForces callback. " *
            "Drive the run via `GeneralizedPerturbedEquilibrium.main` instead of " *
            "calling build_kinetic_matrix_splines directly, or pass " *
            "`calculated_source=KineticForces.compute_calculated_kinetic_matrices` explicitly."
        )
        band = band_knots(xs[1], xs[end])
        if isempty(band)
            kw_flat, kt_flat = calculated_source(ctrl, equil, intr, metric, mats)
        else
            xs = sort!(unique!(vcat(collect(xs), band)))
            mpsi = length(xs)
            kw_flat, kt_flat = calculated_source(ctrl, equil, intr, metric, mats; psis=xs)
        end
        kw_flat .*= ctrl.kinetic_factor
        kt_flat .*= ctrl.kinetic_factor
    else
        error("Unknown kinetic_source: $(ctrl.kinetic_source). Must be \"fixed\" or \"calculated\"")
    end

    # Pre-compute FKG derived matrices (corresponds to Fortran method=0)
    mats = _compute_fkg_matrices(mats, equil, intr, metric, kw_flat, kt_flat; xs=xs)

    # The FKG splines now exist, so F̄ can be scanned: add knots only where near-singular structure
    # (shifted/split kinetic resonances) falls in an interval that does not resolve it.
    if ctrl.kinetic_source == "calculated" && calculated_source !== nothing && mats.kinetic !== nothing
        xs2, kw_flat, kt_flat = refine_grid_at_fbar_peaks(
            collect(xs), kw_flat, kt_flat,
            psis -> begin
                kwn, ktn = calculated_source(ctrl, equil, intr, metric, mats; psis=psis)
                (kwn .* ctrl.kinetic_factor, ktn .* ctrl.kinetic_factor)
            end,
            mats.kinetic, equil, intr, axis_validity_psi_c)
        if length(xs2) != length(xs)
            xs = xs2
            mats = _compute_fkg_matrices(mats, equil, intr, metric, kw_flat, kt_flat; xs=xs)
        end
    end
    return mats
end

"""
    _compute_fkg_matrices(mats, equil, intr, metric, kw_flat, kt_flat) -> MatrixSplines

Pre-compute the derived F, K, G kinetic matrices at each ψ grid point and return a new `MatrixSplines`
whose `kinetic` field holds them alongside the kinetic-modified A/B/C; `mats.ideal` is carried over
unchanged.
This corresponds to `fourfit_kinetic_matrix` method=0 in the Fortran code (Fortran `fourfit.F` lines 1170-1260).

The 9 matrices computed are the Schur complement reductions of ideal (A,B,C,D,E,H) and kinetic (W,T)
primitives into the F, K, G sub-matrices of the non-Hermitian kinetic ODE system (Logan 2015 Appendix C,
Eqs C.1-C.11). These sub-matrices absorb the singfac dependence so that the ODE RHS in `sing_der!`
can be assembled with explicit (m-nq) factors rather than 1/(m-nq), avoiding numerical blow-up at
rational surfaces.
"""
function _compute_fkg_matrices(
    mats::MatrixSplines,
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
    ideal = mats.ideal

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
        amat_full = reshape(ideal.A_spline(psi; hint=hint), np, np)
        bmat_full = reshape(ideal.B_spline(psi; hint=hint), np, np)
        cmat_full = reshape(ideal.C_spline(psi; hint=hint), np, np)
        dmat_full = reshape(ideal.D_spline_prim(psi; hint=hint), np, np)
        emat_full = reshape(ideal.E_spline_prim(psi; hint=hint), np, np)
        hmat_full = reshape(ideal.H_spline(psi; hint=hint), np, np)
        fmat_prim_full = reshape(ideal.F_spline_prim(psi; hint=hint), np, np)

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

    itp_opts = (; extrap=ExtendExtrap())
    kinetic = KineticMatrices(;
        # kinetic-modified A/B/C consumed by sing_der!; A is non-Hermitian here
        A_spline=cubic_interp(xs, Series(ak_flat); itp_opts...),
        B_spline=cubic_interp(xs, Series(bk_flat); itp_opts...),
        C_spline=cubic_interp(xs, Series(ck_flat); itp_opts...),
        Kw_spline=[cubic_interp(xs, Series(@view(kw_flat[:, :, ic])); itp_opts...) for ic in 1:6],
        Kt_spline=[cubic_interp(xs, Series(@view(kt_flat[:, :, ic])); itp_opts...) for ic in 1:6],
        F0_spline=cubic_interp(xs, Series(f0_flat); itp_opts...),
        P_spline=cubic_interp(xs, Series(p_flat); itp_opts...),
        P_spline_adj=cubic_interp(xs, Series(pa_flat); itp_opts...),
        Kk_spline=cubic_interp(xs, Series(kk_flat); itp_opts...),
        Kk_spline_adj=cubic_interp(xs, Series(kka_flat); itp_opts...),
        R1_spline=cubic_interp(xs, Series(r1_flat); itp_opts...),
        R2_spline=cubic_interp(xs, Series(r2_flat); itp_opts...),
        R3_spline=cubic_interp(xs, Series(r3_flat); itp_opts...),
        G_spline_adj=cubic_interp(xs, Series(ga_flat); itp_opts...))

    return MatrixSplines(mats.ideal, kinetic, mats._hint)
end
