"""
    _build_x_matrix(mpert, mlow, sigma; hermitian=true)

Build an mpert×mpert X-shaped matrix with diagonal σ and anti-diagonal entries.

If `hermitian=true`, anti-diagonal entries are imaginary: W[i,j] = i·sign(m_i)·σ,
which preserves W = W† (Hermiticity). This is appropriate for components Ak, Ck, Hk
(Eqs 7.30, 7.32, 7.35 of Logan 2015).

If `hermitian=false`, anti-diagonal entries are real: W[i,j] = sign(m_i)·σ,
which breaks Hermiticity (W ≠ W†). This is appropriate for components Bk, Dk, Ek
(Eqs 7.31, 7.33, 7.34 of Logan 2015) which are not self-adjoint in general.
"""
function _build_x_matrix(mpert::Int, mlow::Int, sigma::Float64; hermitian::Bool=true)
    W = zeros(ComplexF64, mpert, mpert)
    for i in 1:mpert
        m_i = mlow + i - 1
        W[i, i] = sigma

        # Anti-diagonal: find j such that m_j = -m_i
        j = -m_i - mlow + 1
        if 1 <= j <= mpert && i != j
            if hermitian
                # Imaginary: W[i,j] = i·sign(m_i)·σ preserves W = W†
                W[i, j] = im * sign(m_i) * sigma
            else
                # Real: W[i,j] = sign(m_i)·σ breaks Hermiticity
                W[i, j] = sign(m_i) * sigma
            end
        end
    end
    return W
end

"""
    dummy_kinetic_matrices(mpert, mpsi, sigma, mlow)

Build X-shaped dummy kinetic energy matrices for testing all 6 components.

Populates all 6 components of the W (energy) matrices with X-shaped patterns
scaled by `sigma`. Components have physically motivated Hermiticity properties
matching the real kinetic matrices (Logan 2015 thesis, Eqs 7.30-7.35):

| Component | Matrix | Coupling | Hermitian | Scale |
|:--------- |:------ |:-------- |:--------- |:----- |
| 1         | Ak     | W*z·Wz   | Yes       | σ     |
| 2         | Bk     | W*z·Wx   | No        | σ     |
| 3         | Ck     | W*z·Wy   | Yes       | 0.5σ  |
| 4         | Dk     | W*x·Wx   | No        | 0.3σ  |
| 5         | Ek     | W*x·Wy   | No        | 0.2σ  |
| 6         | Hk     | W*y·Wy   | Yes       | 0.1σ  |

Hermitian components use imaginary anti-diagonal entries (i·sign(m)·σ);
non-Hermitian components use real anti-diagonal entries (sign(m)·σ).

Torque matrices (T) are all zero (torque requires finite rotation frequency).

Returns `(kw_flat, kt_flat)` where each is `(mpsi, mpert^2, 6)`.
"""
function dummy_kinetic_matrices(mpert::Int, mpsi::Int, sigma::Float64, mlow::Int)
    kw_flat = zeros(ComplexF64, mpsi, mpert^2, 6)
    kt_flat = zeros(ComplexF64, mpsi, mpert^2, 6)

    # Component scaling factors and Hermiticity properties
    # (component_index, scale_factor, is_hermitian)
    component_specs = [
        (1, 1.0, true),   # Ak: Hermitian, full scale
        (2, 1.0, false),  # Bk: non-Hermitian, full scale (drives resonance shift P)
        (3, 0.5, true),   # Ck: Hermitian, half scale
        (4, 0.3, false),  # Dk: non-Hermitian (enters f0mat)
        (5, 0.2, false),  # Ek: non-Hermitian (enters K matrix)
        (6, 0.1, true)   # Hk: Hermitian, small (enters G only)
    ]

    for (ic, scale, is_herm) in component_specs
        W = _build_x_matrix(mpert, mlow, sigma * scale; hermitian=is_herm)
        w_flat = vec(W)
        for ipsi in 1:mpsi
            kw_flat[ipsi, :, ic] .= w_flat
        end
    end

    return kw_flat, kt_flat
end

"""
    make_kinetic_matrix(ctrl, equil, ffit, intr, metric)

Construct kinetic energy (W) and torque (T) matrices and store as splines in `ffit`.

Dispatches on `ctrl.kin_source`:

  - `"dummy"`: X-shaped Hermitian test matrices scaled by `ctrl.kin_dummy_sigma`
  - `"file"`: Load from PENTRC output files (not yet implemented)
  - `"pentrc"`: Compute via PENTRC (not yet implemented)

After constructing raw matrices, applies:

  - `kinfac1`/`kinfac2` scaling
  - `ktanh_flag` core damping: `factor = 0.5*(1 + tanh((ψ - ktc)*ktw))`

Then builds cubic spline interpolants and stores in `ffit.kwmats[1:6]`, `ffit.ktmats[1:6]`.
If `intr.fkg_kmats_flag`, pre-computes the FKG derived matrices and stores those splines too.
"""
function make_kinetic_matrix(
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars,
    intr::ForceFreeStatesInternal,
    metric::MetricData
)
    xs = metric.xs
    mpsi = length(xs)

    # Get raw kinetic matrices
    if ctrl.kin_source == "dummy"
        kw_flat, kt_flat = dummy_kinetic_matrices(intr.mpert, mpsi, ctrl.kin_dummy_sigma, intr.mlow)
    elseif ctrl.kin_source == "file"
        error("kin_source=\"file\" not yet implemented — requires PENTRC output files")
    elseif ctrl.kin_source == "pentrc"
        error("kin_source=\"pentrc\" not yet implemented — requires PENTRC module")
    else
        error("Unknown kin_source: $(ctrl.kin_source). Must be \"dummy\", \"file\", or \"pentrc\"")
    end

    # Apply scaling and optional core damping
    for ipsi in 1:mpsi
        psi = xs[ipsi]
        factor1 = ctrl.kinfac1
        factor2 = ctrl.kinfac2
        if ctrl.ktanh_flag
            tanh_factor = 0.5 * (1 + tanh((psi - ctrl.ktc) * ctrl.ktw))
            factor1 *= tanh_factor
            factor2 *= tanh_factor
        end
        for ic in 1:6
            @views kw_flat[ipsi, :, ic] .*= factor1
            @views kt_flat[ipsi, :, ic] .*= factor2
        end
    end

    # Build splines for each of the 6 components
    for ic in 1:6
        ffit.kwmats[ic] = cubic_interp(xs, @view(kw_flat[:, :, ic]); ffit.itp_opts...)
        ffit.ktmats[ic] = cubic_interp(xs, @view(kt_flat[:, :, ic]); ffit.itp_opts...)
    end

    # Pre-compute FKG derived matrices (default behavior, as in Fortran method=0)
    intr.fkg_kmats_flag = true
    _compute_fkg_matrices!(ffit, equil, intr, metric, kw_flat, kt_flat)

    return nothing
end

"""
    _compute_fkg_matrices!(ffit, equil, intr, metric, kw_flat, kt_flat)

Pre-compute the derived F, K, G kinetic matrices at each ψ grid point and store as splines.
This corresponds to `fourfit_kinetic_matrix` method=0 in the Fortran code.

The matrices computed are: f0mat, pmat, paat, kkmat, kkaat, r1mat, r2mat, r3mat, gaat.
These are combinations of ideal (A,B,C,D,E,H) and kinetic (W,T) matrices that appear
in the non-Hermitian kinetic ODE system.
"""
function _compute_fkg_matrices!(
    ffit::FourFitVars,
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    metric::MetricData,
    kw_flat::Array{ComplexF64,3},
    kt_flat::Array{ComplexF64,3}
)
    xs = metric.xs
    mpsi = length(xs)
    np = intr.numpert_total
    mpert = intr.mpert
    mband = intr.mband
    profiles = equil.profiles
    chi1 = 2π * equil.psio

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

    hint = Ref(1)

    for ipsi in 1:mpsi
        psi = xs[ipsi]

        # Evaluate ideal matrices from splines
        amat = reshape(ffit.amats(psi; hint=hint), np, np)
        bmat = reshape(ffit.bmats(psi; hint=hint), np, np)
        cmat = reshape(ffit.cmats(psi; hint=hint), np, np)
        dmat = reshape(ffit.dmats(psi; hint=hint), np, np)
        emat = reshape(ffit.emats(psi; hint=hint), np, np)
        hmat = reshape(ffit.hmats(psi; hint=hint), np, np)

        # Reshape kinetic matrices at this psi
        kwmat = zeros(ComplexF64, np, np, 6)
        ktmat = zeros(ComplexF64, np, np, 6)
        for ic in 1:6
            kwmat[:, :, ic] .= reshape(@view(kw_flat[ipsi, :, ic]), np, np)
            ktmat[:, :, ic] .= reshape(@view(kt_flat[ipsi, :, ic]), np, np)
        end

        # Add kinetic contributions to ideal matrices [Fortran fourfit.F lines 1153-1158]
        amat_kin = amat .+ kwmat[:, :, 1] .+ ktmat[:, :, 1]
        bmat_kin = bmat .+ kwmat[:, :, 2] .+ ktmat[:, :, 2]
        cmat_kin = cmat .+ kwmat[:, :, 3] .+ ktmat[:, :, 3]
        hmat_kin = hmat .+ kwmat[:, :, 6] .+ ktmat[:, :, 6]
        caat = cmat_kin .- 2 .* ktmat[:, :, 3]  # C† analog for non-Hermitian system

        # Store kinetic-modified A/B/C for sing_der! FKG path
        ak_flat[ipsi, :] .= vec(amat_kin)
        bk_flat[ipsi, :] .= vec(bmat_kin)
        ck_flat[ipsi, :] .= vec(cmat_kin)

        # In Fortran, dbat/ebat/fbat are the primitive D/E/F before Schur complement reduction.
        # In Julia, dmats/emats store the primitive D/E, and fmats_prim stores the primitive F.
        b1mat = im .* dmat  # b1mat = i*D (Fortran convention)

        # Load primitive F directly (stored separately in make_matrix)
        fmat_prim = reshape(ffit.fmats_prim(psi; hint=hint), np, np)

        # LU factorization of kinetic A matrix (non-Hermitian)
        amat_lu = lu(amat_kin)

        # f0mat = F_prim - D†A_kin⁻¹D  [Fortran fourfit.F line 1184]
        temp1 = amat_lu \ dmat
        f0mat = fmat_prim .- dmat' * temp1

        # pmat [Fortran lines 1193-1200]
        n = intr.nlow  # TODO: generalize for multi-n
        bkmat = kwmat[:, :, 2] .+ ktmat[:, :, 2] .+ im * chi1 / (2π * n) .* (kwmat[:, :, 1] .+ ktmat[:, :, 1])
        bkaat = kwmat[:, :, 2] .- ktmat[:, :, 2] .+ im * chi1 / (2π * n) .* (kwmat[:, :, 1] .+ ktmat[:, :, 1])
        temp2 = amat_lu \ bkmat
        pmat_val = b1mat' * temp2

        # paat [Fortran lines 1202-1207]
        temp2 = amat_lu \ b1mat
        aamat = (amat_lu \ amat_kin)'  # close to identity
        umat_diff = I - aamat
        paat_val = (bkaat' * temp2 .- im * chi1 / (2π * n) .* umat_diff * b1mat)'

        # r1mat [Fortran lines 1209-1217]
        temp1_r1 = kwmat[:, :, 1] .+ ktmat[:, :, 1]
        temp2 = amat_lu \ bkmat
        r1mat_val =
            kwmat[:, :, 4] .+ ktmat[:, :, 4] .-
            (chi1 / (2π * n))^2 .* temp1_r1' .+
            im * chi1 / (2π * n) .* bkaat' .-
            im * chi1 / (2π * n) .* aamat * bkmat .-
            bkaat' * temp2

        # kkmat [Fortran lines 1220-1223]
        temp1 = amat_lu \ cmat_kin
        kkmat_val = emat .- b1mat' * temp1

        # kkaat [Fortran lines 1225-1229]
        temp1 = amat_lu \ b1mat
        kkaat_val = emat' .- caat' * temp1

        # r2mat [Fortran lines 1231-1237]
        temp1_r2 = kwmat[:, :, 5] .+ ktmat[:, :, 5] .- im * chi1 / (2π * n) .* (kwmat[:, :, 3] .+ ktmat[:, :, 3])
        temp2 = amat_lu \ cmat_kin
        r2mat_val = temp1_r2 .+ im * chi1 / (2π * n) .* umat_diff * cmat_kin .- bkaat' * temp2

        # r3mat [Fortran lines 1239-1245]
        temp1_r3 = kwmat[:, :, 5] .- ktmat[:, :, 5] .- im * chi1 / (2π * n) .* (kwmat[:, :, 3] .- ktmat[:, :, 3])
        temp2 = amat_lu \ bkmat
        r3mat_val = temp1_r3' .- caat' * temp2

        # gaat [Fortran lines 1248-1251]
        temp2 = amat_lu \ cmat_kin
        gaat_val = hmat_kin .- caat' * temp2

        # Store flattened
        f0_flat[ipsi, :] .= vec(f0mat)
        p_flat[ipsi, :] .= vec(pmat_val)
        pa_flat[ipsi, :] .= vec(paat_val)
        kk_flat[ipsi, :] .= vec(kkmat_val)
        kka_flat[ipsi, :] .= vec(kkaat_val)
        r1_flat[ipsi, :] .= vec(r1mat_val)
        r2_flat[ipsi, :] .= vec(r2mat_val)
        r3_flat[ipsi, :] .= vec(r3mat_val)
        ga_flat[ipsi, :] .= vec(gaat_val)
    end

    # Build FKG splines
    ffit.f0mats = cubic_interp(xs, f0_flat; ffit.itp_opts...)
    ffit.pmats = cubic_interp(xs, p_flat; ffit.itp_opts...)
    ffit.paats = cubic_interp(xs, pa_flat; ffit.itp_opts...)
    ffit.kkmats = cubic_interp(xs, kk_flat; ffit.itp_opts...)
    ffit.kkaats = cubic_interp(xs, kka_flat; ffit.itp_opts...)
    ffit.r1mats = cubic_interp(xs, r1_flat; ffit.itp_opts...)
    ffit.r2mats = cubic_interp(xs, r2_flat; ffit.itp_opts...)
    ffit.r3mats = cubic_interp(xs, r3_flat; ffit.itp_opts...)
    ffit.gaats = cubic_interp(xs, ga_flat; ffit.itp_opts...)

    # Overwrite ideal A/B/C splines with kinetic-modified versions
    # sing_der! loads these when fkg_kmats_flag=true
    ffit.amats = cubic_interp(xs, ak_flat; ffit.itp_opts...)
    ffit.bmats = cubic_interp(xs, bk_flat; ffit.itp_opts...)
    ffit.cmats = cubic_interp(xs, ck_flat; ffit.itp_opts...)

    return nothing
end
