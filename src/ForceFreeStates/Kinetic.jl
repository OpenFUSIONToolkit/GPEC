"""
    make_kinetic_matrix(ctrl, equil, ffit, intr, metric;
                        calculated_source=nothing)

Construct kinetic energy (W) and torque (T) matrices and pre-compute the FKG derived
matrices used by `sing_der!`, returning a new `FourFitVars` carrying both alongside the
ideal matrices of the input `ffit`.

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
        kw_flat, kt_flat = calculated_source(ctrl, equil, intr, metric, ffit)
        kw_flat .*= ctrl.kinetic_factor
        kt_flat .*= ctrl.kinetic_factor
    else
        error("Unknown kinetic_source: $(ctrl.kinetic_source). Must be \"fixed\" or \"calculated\"")
    end

    # Build splines for each of the 6 components
    kwmats = [cubic_interp(xs, Series(@view(kw_flat[:, :, ic])); ffit.itp_opts...) for ic in 1:6]
    ktmats = [cubic_interp(xs, Series(@view(kt_flat[:, :, ic])); ffit.itp_opts...) for ic in 1:6]

    # Pre-compute FKG derived matrices (corresponds to Fortran method=0)
    return _compute_fkg_matrices(ffit, equil, intr, metric, kw_flat, kt_flat, kwmats, ktmats)
end

"""
    _compute_fkg_matrices(ffit, equil, intr, metric, kw_flat, kt_flat, kwmats, ktmats) -> FourFitVars

Pre-compute the derived F, K, G kinetic matrices at each ψ grid point and return a new `FourFitVars`
holding them, the kinetic-modified A/B/C, and the ideal A/B/C of `ffit` preserved as `*_ideal`.
This corresponds to `fourfit_kinetic_matrix` method=0 in the Fortran code (Fortran `fourfit.F` lines 1170-1260).

The 9 matrices computed are the Schur complement reductions of ideal (A,B,C,D,E,H) and kinetic (W,T)
primitives into the F, K, G sub-matrices of the non-Hermitian kinetic ODE system (Logan 2015 Appendix C,
Eqs C.1-C.11). These sub-matrices absorb the singfac dependence so that the ODE RHS in `sing_der!`
can be assembled with explicit (m-nq) factors rather than 1/(m-nq), avoiding numerical blow-up at
rational surfaces.
"""
function _compute_fkg_matrices(
    ffit::FourFitVars,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    metric::MetricData,
    kw_flat::Array{ComplexF64,3},
    kt_flat::Array{ComplexF64,3},
    kwmats::Vector,
    ktmats::Vector
)
    xs = metric.xs
    mpsi = length(xs)
    np = intr.numpert_total
    mpert = intr.mpert
    npert = intr.npert
    ideal = ffit.ideal

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
        amat_full = reshape(ideal.amats(psi; hint=hint), np, np)
        bmat_full = reshape(ideal.bmats(psi; hint=hint), np, np)
        cmat_full = reshape(ideal.cmats(psi; hint=hint), np, np)
        dmat_full = reshape(ideal.dmats_prim(psi; hint=hint), np, np)
        emat_full = reshape(ideal.emats_prim(psi; hint=hint), np, np)
        hmat_full = reshape(ideal.hmats(psi; hint=hint), np, np)
        fmat_prim_full = reshape(ideal.fmats_prim(psi; hint=hint), np, np)

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

    itp_opts = ffit.itp_opts
    kinetic = KineticMatrices(;
        # kinetic-modified A/B/C consumed by sing_der!; A is non-Hermitian here
        amats=cubic_interp(xs, Series(ak_flat); itp_opts...),
        bmats=cubic_interp(xs, Series(bk_flat); itp_opts...),
        cmats=cubic_interp(xs, Series(ck_flat); itp_opts...),
        kwmats,
        ktmats,
        f0mats=cubic_interp(xs, Series(f0_flat); itp_opts...),
        pmats=cubic_interp(xs, Series(p_flat); itp_opts...),
        paats=cubic_interp(xs, Series(pa_flat); itp_opts...),
        kkmats=cubic_interp(xs, Series(kk_flat); itp_opts...),
        kkaats=cubic_interp(xs, Series(kka_flat); itp_opts...),
        r1mats=cubic_interp(xs, Series(r1_flat); itp_opts...),
        r2mats=cubic_interp(xs, Series(r2_flat); itp_opts...),
        r3mats=cubic_interp(xs, Series(r3_flat); itp_opts...),
        gaats=cubic_interp(xs, Series(ga_flat); itp_opts...))

    return FourFitVars(ffit.numpert_total, itp_opts, ffit.ideal, kinetic, ffit._hint)
end
