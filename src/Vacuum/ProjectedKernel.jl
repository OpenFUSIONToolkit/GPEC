# Fused kernel assembly + Fourier projection for Galerkin vacuum solve.
#
# Instead of materializing the full M×M kernel matrices and then projecting,
# these functions accumulate the P×P projected matrices row by row as the
# kernel values are computed, reducing memory from O(M²) to O(MP).
#
# K_c = Z^H K Z  and  G_c = Z^H G Z
#
# where Z = C + iS is the [M × P] complex Fourier basis, K is the double-layer
# kernel, and G is the single-layer kernel. For each observer point j, the
# kernel row is projected and accumulated via rank-1 updates:
#
#   K_c += conj(Z[j,:]) ⊗ (K[j,:] · Z)
#
# FLOP cost is identical to the two-step approach O(M²P), but memory drops
# from O(M²) to O(MP + P²).

# ============================================================================
# 2D fused projected kernel
# ============================================================================
"""
    projected_kernel!(K_c, G_c, observer, source, params, exp_mn_basis, Gram)

Compute the Fourier-projected kernel matrices K_c = Z^H K Z and G_c = Z^H G Z
directly, without materializing the full M×M kernel matrices.

Dispatches to the 2D or 3D implementation based on the geometry/params types.

# Arguments

  - `K_c::Matrix{ComplexF64}`: Output P×P projected double-layer kernel [filled in-place]
  - `G_c::Matrix{ComplexF64}`: Output P×P projected single-layer kernel [filled in-place]
  - `observer`: Observer geometry struct
  - `source`: Source geometry struct
  - `params`: Kernel parameters (KernelParams2D or KernelParams3D)
  - `exp_mn_basis::Matrix{ComplexF64}`: [M × P] complex Fourier basis Z = exp(i(mθ − nζ))
  - `Gram::Matrix{ComplexF64}`: [P × P] Gram matrix Z^H Z (needed for diagonal identity term)
"""
function projected_kernel! end

function projected_kernel!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry,WallGeometry},
    source::Union{PlasmaGeometry,WallGeometry},
    params::KernelParams2D,
    exp_mn_basis::AbstractMatrix{ComplexF64},
    Gram::AbstractMatrix{ComplexF64}
)
    _projected_kernel_2D!(K_c, G_c, observer, source, params.n, exp_mn_basis, Gram)
end

function projected_kernel!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    params::KernelParams3D,
    exp_mn_basis::AbstractMatrix{ComplexF64},
    Gram::AbstractMatrix{ComplexF64}
)
    _projected_kernel_3D!(K_c, G_c, observer, source,
        params.PATCH_RAD, params.RAD_DIM, params.INTERP_ORDER,
        exp_mn_basis, Gram)
end
"""
    _projected_kernel_2D!(K_c, G_c, observer, source, n, exp_mn_basis, Gram)

Fused 2D kernel assembly + projection. Mirrors the loop structure of
`compute_2D_kernel_matrices!` but accumulates rank-1 contributions into the
P×P projected matrices instead of filling the M×M kernel matrices.

Memory: O(MP) instead of O(M²).
"""
@with_pool pool function _projected_kernel_2D!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry,WallGeometry},
    source::Union{PlasmaGeometry,WallGeometry},
    n::Int,
    exp_mn_basis::AbstractMatrix{ComplexF64},
    Gram::AbstractMatrix{ComplexF64}
)
    M, P = size(exp_mn_basis)
    Z = exp_mn_basis
    mtheta = length(observer.x)
    dtheta = 2π / mtheta
    theta_grid = range(; start=0, length=mtheta, step=dtheta)

    populate_greenfunction = source isa PlasmaGeometry

    # S₁ᵢ logarithmic correction factors [Chance Phys. Plasmas 1997 2161 eq. 78]
    log_correction_0 = 16.0 * dtheta * (log(2 * dtheta) - 68.0 / 15.0) / 15.0
    log_correction_1 = 128.0 * dtheta * (log(2 * dtheta) - 8.0 / 15.0) / 45.0
    log_correction_2 = 4.0 * dtheta * (7.0 * log(2 * dtheta) - 11.0 / 15.0) / 45.0
    log_correction_array = SVector(log_correction_2, log_correction_1, log_correction_0, log_correction_1, log_correction_2)

    gamma_prefactor = 2 * sqrt(π) * gamma(0.5 - n)

    spline_x = cubic_interp(theta_grid, source.x; bc=PeriodicBC(; endpoint=:exclusive, period=2π))
    spline_z = cubic_interp(theta_grid, source.z; bc=PeriodicBC(; endpoint=:exclusive, period=2π))
    d1_spline_x = deriv1(spline_x)
    d1_spline_z = deriv1(spline_z)

    stencils_left, stencils_right = GL8_LAGRANGE_STENCILS
    sing_idx = zeros!(pool, Int, 5)

    dx_dtheta_grid = acquire!(pool, eltype(source.x), mtheta)
    dz_dtheta_grid = acquire!(pool, eltype(source.z), mtheta)
    d1_spline_x(dx_dtheta_grid, theta_grid)
    d1_spline_z(dz_dtheta_grid, theta_grid)

    # Zero output matrices; we accumulate rank-1 updates (conj(Z[j,:]) ⊗ proj_z)
    fill!(K_c, 0.0)
    fill!(G_c, 0.0)

    # Per-observer projection vectors (P-length complex): proj_z = (kernel row) · Z
    proj_kz = zeros(ComplexF64, P)
    proj_gz = zeros(ComplexF64, P)

    for j in 1:mtheta
        x_obs, z_obs, theta_obs = observer.x[j], observer.z[j], theta_grid[j]

        fill!(proj_kz, 0.0)
        fill!(proj_gz, 0.0)
        diag_accum = 0.0

        # ── Simpson integration for nonsingular source points ──
        @inbounds for k in 1:(mtheta-3)
            isrc = mod1(j + 1 + k, mtheta)
            G_n, gradG_n, gradG_0 = green(x_obs, z_obs,
                source.x[isrc], source.z[isrc],
                dx_dtheta_grid[isrc], dz_dtheta_grid[isrc], n;
                gamma_prefactor)

            wsimpson = dtheta / 3 * ((k == 1 || k == mtheta - 3) ? 1 : (iseven(k) ? 4 : 2))

            if populate_greenfunction
                w_g = G_n * wsimpson
                BLAS.axpy!(ComplexF64(w_g), @view(Z[isrc, :]), proj_gz)
            end
            w_k = gradG_n * wsimpson
            BLAS.axpy!(ComplexF64(w_k), @view(Z[isrc, :]), proj_kz)

            diag_accum -= gradG_0 * wsimpson
        end

        # ── Gaussian quadrature for singular points ──
        for (offset_idx, offset) in enumerate(-2:2)
            sing_idx[offset_idx] = mod1(j + offset + mtheta, mtheta)
        end

        for leftpanel in (true, false)
            gauss_mid = theta_obs + (leftpanel ? -dtheta : dtheta)
            @inbounds for ig in 1:8
                theta_gauss = gauss_mid + GL8.x[ig] * dtheta
                theta_gauss0 = mod(theta_gauss, 2π)
                x_gauss = spline_x(theta_gauss0)
                dx_dtheta_gauss = d1_spline_x(theta_gauss0)
                z_gauss = spline_z(theta_gauss0)
                dz_dtheta_gauss = d1_spline_z(theta_gauss0)
                G_n, gradG_n, gradG_0 = green(x_obs, z_obs,
                    x_gauss, z_gauss, dx_dtheta_gauss, dz_dtheta_gauss, n;
                    gamma_prefactor)

                s = leftpanel ? stencils_left[ig] : stencils_right[ig]
                wgauss = GL8.w[ig] * dtheta

                if populate_greenfunction
                    if observer isa PlasmaGeometry
                        G_n += log((theta_obs - theta_gauss)^2) / x_obs
                    end
                    @inbounds for stencil_idx in 1:5
                        w_g = G_n * s[stencil_idx] * wgauss
                        isrc = sing_idx[stencil_idx]
                        BLAS.axpy!(ComplexF64(w_g), @view(Z[isrc, :]), proj_gz)
                    end
                end

                @inbounds for stencil_idx in 1:5
                    w_k = gradG_n * s[stencil_idx] * wgauss
                    isrc = sing_idx[stencil_idx]
                    BLAS.axpy!(ComplexF64(w_k), @view(Z[isrc, :]), proj_kz)
                end

                diag_accum -= gradG_0 * wgauss
            end
        end

        # Analytic singular integral correction [Chance 1997 eq. 75]
        if populate_greenfunction && observer isa PlasmaGeometry
            @inbounds for stencil_idx in 1:5
                w_g = -log_correction_array[stencil_idx] / x_obs
                isrc = sing_idx[stencil_idx]
                BLAS.axpy!(ComplexF64(w_g), @view(Z[isrc, :]), proj_gz)
            end
        end

        # Fold diagonal accumulation into projection
        BLAS.axpy!(ComplexF64(diag_accum), @view(Z[j, :]), proj_kz)

        # ── Rank-1 accumulate: K_c += conj(Z[j,:]) ⊗ proj_kz ──
        BLAS.geru!(ComplexF64(1.0), conj.(@view(Z[j, :])), proj_kz, K_c)
        if populate_greenfunction
            BLAS.geru!(ComplexF64(1.0), conj.(@view(Z[j, :])), proj_gz, G_c)
        end
    end

    # ── Post-processing (mirrors compute_2D_kernel_matrices!) ──

    # Normals point out of vacuum for wall but inward for plasma → flip sign for plasma source
    if source isa PlasmaGeometry
        K_c .*= -1
    end

    # Diagonal residue: K += residue·I  →  K_c += residue·Gram
    # [Chance Phys. Plasmas 1997 2161 Table I, eq. 69, 89]
    residue = (observer isa WallGeometry) ? 0.0 : (source isa PlasmaGeometry ? 2.0 : -2.0)
    if residue != 0.0
        K_c .+= residue .* Gram
    end

    # 2π𝒢 → 𝒢
    if populate_greenfunction
        G_c ./= 2π
    end
end


# ============================================================================
# 3D fused projected kernel
# ============================================================================

"""
    _projected_kernel_3D!(K_c, G_c, observer, source, PATCH_RAD, RAD_DIM, INTERP_ORDER, exp_mn_basis, Gram)

Fused 3D kernel assembly + projection. Mirrors the loop structure of
`compute_3D_kernel_matrices!` (including multi-threading and BIEST singular correction)
but writes projected P-vectors to per-observer rows of [M × P] buffers instead of
filling the M×M kernel matrices. The P×P assembly is done after the parallel loop
via sequential GEMM calls.

Each observer writes to its own row of the shared buffers, so there are no
cross-thread accumulation races — the same write pattern as the original
`compute_3D_kernel_matrices!`.

Memory: O(2MP + P²) instead of O(M²).
"""
function _projected_kernel_3D!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    PATCH_RAD::Int,
    RAD_DIM::Int,
    INTERP_ORDER::Int,
    exp_mn_basis::AbstractMatrix{ComplexF64},
    Gram::AbstractMatrix{ComplexF64}
)
    M, P = size(exp_mn_basis)
    Z = exp_mn_basis
    num_points = observer.mtheta * observer.nzeta
    dθdζ = 4π^2 / num_points

    populate_greenfunction = source isa PlasmaGeometry3D

    if PATCH_RAD > (min(source.mtheta, source.nzeta) - 1) ÷ 2
        @warn "PATCH_RAD clamped in projected kernel" max_PATCH_RAD=(min(source.mtheta, source.nzeta) - 1) ÷ 2
        PATCH_RAD = (min(source.mtheta, source.nzeta) - 1) ÷ 2
    end
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, ANG_DIM, Ppou, Gpou, P2G) = quad_data

    # [M × P] buffers: row idx_obs holds (kernel row idx_obs) · Z
    KZ = zeros(ComplexF64, M, P)
    GZ = zeros(ComplexF64, M, P)

    # Per-thread workspace (kernel scratch arrays + P-length accumulation vectors)
    max_tid = Threads.maxthreadid()
    workspaces = [KernelWorkspace(PATCH_DIM, RAD_DIM, ANG_DIM) for _ in 1:max_tid]
    proj_kz_all = [zeros(ComplexF64, P) for _ in 1:max_tid]
    proj_gz_all = [zeros(ComplexF64, P) for _ in 1:max_tid]

    Threads.@threads :static for idx_obs in 1:num_points
        tid = Threads.threadid()
        ws = workspaces[tid]
        (; r_patch, dr_dθ_patch, dr_dζ_patch, r_polar, dr_dθ_polar, dr_dζ_polar,
            n_polar, M_polar_single, M_polar_double, M_grid_single_flat, M_grid_double_flat) = ws

        proj_kz = proj_kz_all[tid]
        proj_gz = proj_gz_all[tid]

        fill!(proj_kz, 0.0)
        fill!(proj_gz, 0.0)

        i_obs = mod1(idx_obs, observer.mtheta)
        j_obs = (idx_obs - 1) ÷ observer.mtheta + 1
        r_obs = @view observer.r[idx_obs, :]

        # ── FAR FIELD: Trapezoidal rule ──
        @inbounds for idx_src in 1:num_points
            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            w_double = laplace_double_layer(r_obs, r_src, n_src) * dθdζ
            BLAS.axpy!(ComplexF64(w_double), @view(Z[idx_src, :]), proj_kz)

            if populate_greenfunction
                w_single = laplace_single_layer(r_obs, r_src) * dθdζ
                BLAS.axpy!(ComplexF64(w_single), @view(Z[idx_src, :]), proj_gz)
            end
        end

        # ── NEAR FIELD: Polar quadrature with BIEST singular correction ──
        extract_patch!(r_patch, source.r, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        extract_patch!(dr_dθ_patch, source.dr_dθ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)
        extract_patch!(dr_dζ_patch, source.dr_dζ, i_obs, j_obs, source.mtheta, source.nzeta, PATCH_DIM)

        interpolate_to_polar!(r_polar, r_patch, P2G)
        interpolate_to_polar!(dr_dθ_polar, dr_dθ_patch, P2G)
        interpolate_to_polar!(dr_dζ_polar, dr_dζ_patch, P2G)

        compute_polar_normal!(n_polar, dr_dθ_polar, dr_dζ_polar, source.normal_orient)

        @inbounds for ia in 1:ANG_DIM, ir in 1:RAD_DIM
            r_src = @view r_polar[ir, ia, :]
            n_src = @view n_polar[ir, ia, :]
            M_polar_single[ir, ia] = laplace_single_layer(r_obs, r_src) * Ppou[ir, ia] * dθdζ
            M_polar_double[ir, ia] = laplace_double_layer(r_obs, r_src, n_src) * Ppou[ir, ia] * dθdζ
        end

        mul!(M_grid_single_flat, P2G, vec(M_polar_single))
        mul!(M_grid_double_flat, P2G, vec(M_polar_double))
        M_grid_single = reshape(M_grid_single_flat, PATCH_DIM, PATCH_DIM)
        M_grid_double = reshape(M_grid_double_flat, PATCH_DIM, PATCH_DIM)

        @inbounds for jj in 1:PATCH_DIM, ii in 1:PATCH_DIM
            idx_pol = periodic_wrap(i_obs - PATCH_RAD + ii - 1, source.mtheta)
            idx_tor = periodic_wrap(j_obs - PATCH_RAD + jj - 1, source.nzeta)
            idx_src = idx_pol + source.mtheta * (idx_tor - 1)

            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            far_double = laplace_double_layer(r_obs, r_src, n_src) * Gpou[ii, jj] * dθdζ
            w_double = M_grid_double[ii, jj] + far_double
            BLAS.axpy!(ComplexF64(w_double), @view(Z[idx_src, :]), proj_kz)

            if populate_greenfunction
                far_single = laplace_single_layer(r_obs, r_src) * Gpou[ii, jj] * dθdζ
                w_single = M_grid_single[ii, jj] + far_single
                BLAS.axpy!(ComplexF64(w_single), @view(Z[idx_src, :]), proj_gz)
            end
        end

        # ── Write projected row to buffer (each idx_obs owns its row) ──
        @inbounds KZ[idx_obs, :] .= proj_kz
        if populate_greenfunction
            @inbounds GZ[idx_obs, :] .= proj_gz
        end
    end

    # ── Assemble P×P projected matrices: K_c = Z^H K Z, G_c = Z^H G Z ──
    mul!(K_c, Z', KZ)
    K_c ./= 2π
    if populate_greenfunction
        mul!(G_c, Z', GZ)
        G_c ./= 2π
    else
        fill!(G_c, 0.0)
    end

    # Diagonal: K += I → K_c += Gram [for same-type source/observer]
    if typeof(source) == typeof(observer)
        K_c .+= Gram
    end
end
