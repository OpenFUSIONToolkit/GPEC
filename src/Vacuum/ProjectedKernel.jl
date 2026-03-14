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
    projected_kernel!(K_c, G_c, observer, source, params, cos_basis, sin_basis, Gram)

Compute the Fourier-projected kernel matrices K_c = Z^H K Z and G_c = Z^H G Z
directly, without materializing the full M×M kernel matrices.

Dispatches to the 2D or 3D implementation based on the geometry/params types.

# Arguments

  - `K_c::Matrix{ComplexF64}`: Output P×P projected double-layer kernel [filled in-place]
  - `G_c::Matrix{ComplexF64}`: Output P×P projected single-layer kernel [filled in-place]
  - `observer`: Observer geometry struct
  - `source`: Source geometry struct
  - `params`: Kernel parameters (KernelParams2D or KernelParams3D)
  - `cos_basis::Matrix{Float64}`: [M × P] cosine Fourier basis
  - `sin_basis::Matrix{Float64}`: [M × P] sine Fourier basis
  - `Gram::Matrix{ComplexF64}`: [P × P] Gram matrix Z^H Z (needed for diagonal identity term)
"""
function projected_kernel! end

function projected_kernel!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry,WallGeometry},
    source::Union{PlasmaGeometry,WallGeometry},
    params::KernelParams2D,
    cos_basis::Matrix{Float64},
    sin_basis::Matrix{Float64},
    Gram::AbstractMatrix{ComplexF64}
)
    _projected_kernel_2D!(K_c, G_c, observer, source, params.n, cos_basis, sin_basis, Gram)
end

function projected_kernel!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    params::KernelParams3D,
    cos_basis::Matrix{Float64},
    sin_basis::Matrix{Float64},
    Gram::AbstractMatrix{ComplexF64}
)
    _projected_kernel_3D!(K_c, G_c, observer, source,
        params.PATCH_RAD, params.RAD_DIM, params.INTERP_ORDER,
        cos_basis, sin_basis, Gram)
end
"""
    _projected_kernel_2D!(K_c, G_c, observer, source, n, cos_basis, sin_basis, Gram)

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
    cos_basis::Matrix{Float64},
    sin_basis::Matrix{Float64},
    Gram::AbstractMatrix{ComplexF64}
)
    M, P = size(cos_basis)
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

    # Pre-transpose basis for contiguous column access: Ct[:, k] = C[k, :]
    Ct = acquire!(pool, Float64, P, M)
    St = acquire!(pool, Float64, P, M)
    Ct .= cos_basis'
    St .= sin_basis'

    # Real/imaginary accumulators for P×P projected matrices
    K_re = zeros(P, P)
    K_im = zeros(P, P)
    G_re = zeros(P, P)
    G_im = zeros(P, P)

    # Per-observer projection vectors (P-length)
    proj_kc = zeros(P)
    proj_ks = zeros(P)
    proj_gc = zeros(P)
    proj_gs = zeros(P)

    for j in 1:mtheta
        x_obs, z_obs, theta_obs = observer.x[j], observer.z[j], theta_grid[j]

        fill!(proj_kc, 0.0)
        fill!(proj_ks, 0.0)
        fill!(proj_gc, 0.0)
        fill!(proj_gs, 0.0)
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
                BLAS.axpy!(w_g, @view(Ct[:, isrc]), proj_gc)
                BLAS.axpy!(w_g, @view(St[:, isrc]), proj_gs)
            end
            w_k = gradG_n * wsimpson
            BLAS.axpy!(w_k, @view(Ct[:, isrc]), proj_kc)
            BLAS.axpy!(w_k, @view(St[:, isrc]), proj_ks)

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
                        BLAS.axpy!(w_g, @view(Ct[:, isrc]), proj_gc)
                        BLAS.axpy!(w_g, @view(St[:, isrc]), proj_gs)
                    end
                end

                @inbounds for stencil_idx in 1:5
                    w_k = gradG_n * s[stencil_idx] * wgauss
                    isrc = sing_idx[stencil_idx]
                    BLAS.axpy!(w_k, @view(Ct[:, isrc]), proj_kc)
                    BLAS.axpy!(w_k, @view(St[:, isrc]), proj_ks)
                end

                diag_accum -= gradG_0 * wgauss
            end
        end

        # Analytic singular integral correction [Chance 1997 eq. 75]
        if populate_greenfunction && observer isa PlasmaGeometry
            @inbounds for stencil_idx in 1:5
                w_g = -log_correction_array[stencil_idx] / x_obs
                isrc = sing_idx[stencil_idx]
                BLAS.axpy!(w_g, @view(Ct[:, isrc]), proj_gc)
                BLAS.axpy!(w_g, @view(St[:, isrc]), proj_gs)
            end
        end

        # Fold diagonal accumulation into projection
        BLAS.axpy!(diag_accum, @view(Ct[:, j]), proj_kc)
        BLAS.axpy!(diag_accum, @view(St[:, j]), proj_ks)

        # ── Rank-1 accumulate into P×P projection matrices ──
        # K_c_re += C[j,:] ⊗ proj_kc + S[j,:] ⊗ proj_ks
        BLAS.ger!(1.0, @view(Ct[:, j]), proj_kc, K_re)
        BLAS.ger!(1.0, @view(St[:, j]), proj_ks, K_re)
        # K_c_im += C[j,:] ⊗ proj_ks − S[j,:] ⊗ proj_kc
        BLAS.ger!(1.0, @view(Ct[:, j]), proj_ks, K_im)
        BLAS.ger!(-1.0, @view(St[:, j]), proj_kc, K_im)

        if populate_greenfunction
            BLAS.ger!(1.0, @view(Ct[:, j]), proj_gc, G_re)
            BLAS.ger!(1.0, @view(St[:, j]), proj_gs, G_re)
            BLAS.ger!(1.0, @view(Ct[:, j]), proj_gs, G_im)
            BLAS.ger!(-1.0, @view(St[:, j]), proj_gc, G_im)
        end
    end

    # ── Post-processing (mirrors compute_2D_kernel_matrices!) ──

    # Normals point out of vacuum for wall but inward for plasma → flip sign for plasma source
    if source isa PlasmaGeometry
        K_re .*= -1
        K_im .*= -1
    end

    # Diagonal residue: K += residue·I  →  K_c += residue·Gram
    # [Chance Phys. Plasmas 1997 2161 Table I, eq. 69, 89]
    residue = (observer isa WallGeometry) ? 0.0 : (source isa PlasmaGeometry ? 2.0 : -2.0)
    if residue != 0.0
        K_re .+= residue .* real.(Gram)
        K_im .+= residue .* imag.(Gram)
    end

    # 2π𝒢 → 𝒢
    if populate_greenfunction
        G_re ./= 2π
        G_im ./= 2π
    end

    K_c .= complex.(K_re, K_im)
    G_c .= complex.(G_re, G_im)
end


# ============================================================================
# 3D fused projected kernel
# ============================================================================

"""
    _projected_kernel_3D!(K_c, G_c, observer, source, PATCH_RAD, RAD_DIM, INTERP_ORDER, cos_basis, sin_basis, Gram)

Fused 3D kernel assembly + projection. Mirrors the loop structure of
`compute_3D_kernel_matrices!` (including multi-threading and BIEST singular correction)
but writes projected P-vectors to per-observer rows of [M × P] buffers instead of
filling the M×M kernel matrices. The P×P assembly is done after the parallel loop
via sequential GEMM calls.

Each observer writes to its own row of the shared buffers, so there are no
cross-thread accumulation races — the same write pattern as the original
`compute_3D_kernel_matrices!`.

Memory: O(4MP + P²) instead of O(M²).
"""
function _projected_kernel_3D!(
    K_c::AbstractMatrix{ComplexF64},
    G_c::AbstractMatrix{ComplexF64},
    observer::Union{PlasmaGeometry3D,WallGeometry3D},
    source::Union{PlasmaGeometry3D,WallGeometry3D},
    PATCH_RAD::Int,
    RAD_DIM::Int,
    INTERP_ORDER::Int,
    cos_basis::Matrix{Float64},
    sin_basis::Matrix{Float64},
    Gram::AbstractMatrix{ComplexF64}
)
    M, P = size(cos_basis)
    num_points = observer.mtheta * observer.nzeta
    dθdζ = 4π^2 / num_points

    populate_greenfunction = source isa PlasmaGeometry3D

    if PATCH_RAD > (min(source.mtheta, source.nzeta) - 1) ÷ 2
        @warn "PATCH_RAD clamped in projected kernel" max_PATCH_RAD=(min(source.mtheta, source.nzeta) - 1) ÷ 2
        PATCH_RAD = (min(source.mtheta, source.nzeta) - 1) ÷ 2
    end
    quad_data = get_singular_quadrature(PATCH_RAD, RAD_DIM, INTERP_ORDER)
    (; PATCH_DIM, ANG_DIM, Ppou, Gpou, P2G) = quad_data

    # Pre-transpose basis for contiguous column access in the inner loop
    Ct = Matrix(cos_basis')   # [P × M]
    St = Matrix(sin_basis')   # [P × M]

    # [M × P] buffers for projected kernel rows.
    # Row idx_obs = Σ_k K[idx_obs, k] · basis[k, :] — each observer writes to
    # its own row, so no cross-thread races.
    KZ_c = zeros(M, P)
    KZ_s = zeros(M, P)
    GZ_c = zeros(M, P)
    GZ_s = zeros(M, P)

    # Per-thread workspace (kernel scratch arrays + P-length accumulation vectors)
    max_tid = Threads.maxthreadid()
    workspaces = [KernelWorkspace(PATCH_DIM, RAD_DIM, ANG_DIM) for _ in 1:max_tid]
    proj_kc_all = [zeros(P) for _ in 1:max_tid]
    proj_ks_all = [zeros(P) for _ in 1:max_tid]
    proj_gc_all = [zeros(P) for _ in 1:max_tid]
    proj_gs_all = [zeros(P) for _ in 1:max_tid]

    Threads.@threads :static for idx_obs in 1:num_points
        tid = Threads.threadid()
        ws = workspaces[tid]
        (; r_patch, dr_dθ_patch, dr_dζ_patch, r_polar, dr_dθ_polar, dr_dζ_polar,
            n_polar, M_polar_single, M_polar_double, M_grid_single_flat, M_grid_double_flat) = ws

        proj_kc = proj_kc_all[tid]
        proj_ks = proj_ks_all[tid]
        proj_gc = proj_gc_all[tid]
        proj_gs = proj_gs_all[tid]

        fill!(proj_kc, 0.0)
        fill!(proj_ks, 0.0)
        fill!(proj_gc, 0.0)
        fill!(proj_gs, 0.0)

        i_obs = mod1(idx_obs, observer.mtheta)
        j_obs = (idx_obs - 1) ÷ observer.mtheta + 1
        r_obs = @view observer.r[idx_obs, :]

        # ── FAR FIELD: Trapezoidal rule ──
        @inbounds for idx_src in 1:num_points
            r_src = @view source.r[idx_src, :]
            n_src = @view source.normal[idx_src, :]
            w_double = laplace_double_layer(r_obs, r_src, n_src) * dθdζ
            @inbounds @simd for m in 1:P
                proj_kc[m] += w_double * Ct[m, idx_src]
                proj_ks[m] += w_double * St[m, idx_src]
            end

            if populate_greenfunction
                w_single = laplace_single_layer(r_obs, r_src) * dθdζ
                @inbounds @simd for m in 1:P
                    proj_gc[m] += w_single * Ct[m, idx_src]
                    proj_gs[m] += w_single * St[m, idx_src]
                end
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
            @simd for m in 1:P
                proj_kc[m] += w_double * Ct[m, idx_src]
                proj_ks[m] += w_double * St[m, idx_src]
            end

            if populate_greenfunction
                far_single = laplace_single_layer(r_obs, r_src) * Gpou[ii, jj] * dθdζ
                w_single = M_grid_single[ii, jj] + far_single
                @simd for m in 1:P
                    proj_gc[m] += w_single * Ct[m, idx_src]
                    proj_gs[m] += w_single * St[m, idx_src]
                end
            end
        end

        # ── Write projected row to buffer (each idx_obs owns its row) ──
        @inbounds for m in 1:P
            KZ_c[idx_obs, m] = proj_kc[m]
            KZ_s[idx_obs, m] = proj_ks[m]
        end
        if populate_greenfunction
            @inbounds for m in 1:P
                GZ_c[idx_obs, m] = proj_gc[m]
                GZ_s[idx_obs, m] = proj_gs[m]
            end
        end
    end

    # ── Assemble P×P projected matrices via GEMM (sequential, after barrier) ──
    # K_c = Z^H K Z = (C'·KZ_c + S'·KZ_s) + i(C'·KZ_s − S'·KZ_c)
    K_re = zeros(P, P)
    K_im = zeros(P, P)
    mul!(K_re, cos_basis', KZ_c)
    mul!(K_re, sin_basis', KZ_s, 1.0, 1.0)
    mul!(K_im, cos_basis', KZ_s)
    mul!(K_im, sin_basis', KZ_c, -1.0, 1.0)

    G_re = zeros(P, P)
    G_im = zeros(P, P)
    if populate_greenfunction
        mul!(G_re, cos_basis', GZ_c)
        mul!(G_re, sin_basis', GZ_s, 1.0, 1.0)
        mul!(G_im, cos_basis', GZ_s)
        mul!(G_im, sin_basis', GZ_c, -1.0, 1.0)
    end

    # ── Post-processing (mirrors compute_3D_kernel_matrices!) ──
    K_re ./= 2π
    K_im ./= 2π
    G_re ./= 2π
    G_im ./= 2π

    # Diagonal: K += I → K_c += Gram [for same-type source/observer]
    if typeof(source) == typeof(observer)
        K_re .+= real.(Gram)
        K_im .+= imag.(Gram)
    end

    K_c .= complex.(K_re, K_im)
    G_c .= complex.(G_re, G_im)
end
