"""
    laplace_single_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}) -> Float64

Evaluate the Laplace single-layer (FxU) kernel between two 3D points.

The single-layer kernel φ is the fundamental solution to Laplace's equation:

```
φ(x_obs, x_src) = 1 / (4π |x_obs - x_src|)
```

# Arguments

  - `x_obs::Vector{Float64}`: Observation point (3D Cartesian coordinates)
  - `x_src::Vector{Float64}`: Source point (3D Cartesian coordinates)

# Returns

  - `Float64`: Kernel value φ(x_obs, x_src)
"""
function laplace_single_layer(x_obs::Vector{Float64}, x_src::Vector{Float64})::Float64

    # Compute separation vector
    dx = x_obs[1] - x_src[1]
    dy = x_obs[2] - x_src[2]
    dz = x_obs[3] - x_src[3]

    # Distance
    r = sqrt(dx^2 + dy^2 + dz^2)

    # Single-layer kernel
    return 1.0 / (4π * r)
end

"""
    laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64}) -> Float64

Evaluate the Laplace double-layer (DxU) kernel between a point and a surface element.

The double-layer kernel K is the normal derivative of the fundamental solution:

```
K(x_obs, x_src, n_src) = ∇_{x_src} φ · n̂_src
                       = -1/(4π) * (x_obs - x_src) · n̂_src / |x_obs - x_src|³
```

# Arguments

  - `x_obs::Vector{Float64}`: Observation point (3D Cartesian coordinates)
  - `x_src::Vector{Float64}`: Source point on surface (3D Cartesian coordinates)
  - `n_src::Vector{Float64}`: Outward UNIT normal at source point (must be normalized!)

# Returns

  - `Float64`: Kernel value K(x_obs, x_src, n_src)
"""
function laplace_double_layer(x_obs::Vector{Float64}, x_src::Vector{Float64}, n_src::Vector{Float64})::Float64

    # Compute separation vector
    dx = x_obs[1] - x_src[1]
    dy = x_obs[2] - x_src[2]
    dz = x_obs[3] - x_src[3]

    # Distance
    r = sqrt(dx^2 + dy^2 + dz^2)

    # Dot product: (x_obs - x_src) · n_src
    r_dot_n = dx * n_src[1] + dy * n_src[2] + dz * n_src[3]

    # Double-layer kernel: -1/(4π) * (r·n) / r³
    return -r_dot_n / (4π * r^3)
end

function compute_3D_kernel_matrix!(
    grad_greenfunction::Matrix{Float64},
    greenfunction::Matrix{Float64},
    observer::PlasmaGeometry3D,
    source::PlasmaGeometry3D;
    PATCH_DIM=3
)

    # Zero out greenfunction at start of each kernel call (matches Fortran behavior)
    fill!(grad_greenfunction, 0.0)
    fill!(greenfunction, 0.0)
    @inline periodic_dist(i, j, n) = min(abs(i - j), n - abs(i - j))


    # Loop through observer points
    for i_obs in 1:observer.ntheta, j_obs in 1:observer.nzeta
        idx_obs = i_obs + (j_obs - 1) * observer.ntheta
        # Initialize variables
        r_obs = observer.r[idx_obs, :]

        # Get indices excluding the singular region (±PATCH_DIM around observer point)
        # Only include points where AT LEAST ONE theta and zeta are outside the patch
        nonsing_idx = Vector{Int}(undef, 0)
        sizehint!(nonsing_idx, observer.ntheta * observer.nzeta)
        for j in 1:observer.nzeta, i in 1:observer.ntheta
            if periodic_dist(i, i_obs, observer.ntheta) > PATCH_DIM ||
               periodic_dist(j, j_obs, observer.nzeta) > PATCH_DIM
                push!(nonsing_idx, i + (j - 1) * observer.ntheta)
            end
        end

        # Perform 2D periodic trapezoidal integration for nonsingular source points
        # Note that all trapezoidal weights are 1 since we perform the full periodic integration
        # However, we are actually integrating the kernel times the POU function, which is 1 at nonsingular points
        for idx_src in nonsing_idx
            greenfunction[idx_obs, idx_src] += laplace_single_layer(r_obs, source.r[idx_src, :]) * source.dA[idx_src]
            grad_greenfunction[idx_obs, idx_src] += laplace_double_layer(r_obs, source.r[idx_src, :], source.n[idx_src, :]) * source.dA[idx_src]
        end

        # TODO: singular region treatment!!
        # Perform Gaussian quadrature for singular points (source = obs point)
        # Get indices of the singularity region, [j-2, j-1, j, j+1, j+2]
        # sing_idx = mod1.(j .+ ((mtheta-2):(mtheta+2)), mtheta)
        # # Integrate region of length 2 * dtheta on left/right of singularity
        # for region in ["left", "right"]
        #     gauss_mid = theta_obs + (region == "left" ? -dtheta : dtheta)
        #     theta_gauss = gauss_mid .+ GAUSSIANPOINTS .* dtheta
        #     wgauss = GAUSSIANWEIGHTS .* dtheta
        #     for ig in 1:8 # 8-point Gaussian quadrature
        #         # Compute green function for this Gaussian point
        #         theta_gauss0 = mod(theta_gauss[ig], 2π)
        #         x_gauss = spline_x(theta_gauss0)
        #         dx_dtheta_gauss = Interpolations.gradient(spline_x, theta_gauss0)[1]
        #         z_gauss = spline_z(theta_gauss0)
        #         dz_dtheta_gauss = Interpolations.gradient(spline_z, theta_gauss0)[1]
        #         G_n, coupling_n, coupling_0 = green(x_obs, z_obs, x_gauss, z_gauss, dx_dtheta_gauss, dz_dtheta_gauss, n)

        #         # Add logarithm to G_n to analytically isolate the singularity (first type), Chance eq.(75)
        #         if observer isa PlasmaGeometry && source isa PlasmaGeometry # previously iops
        #             G_n += log((theta_obs - theta_gauss[ig])^2) / x_obs
        #         end

        #         p = (theta_gauss[ig] - theta_obs) / dtheta # p = θ/Δ = (θⱼ - θ')/Δ
        #         stencil_points = SVector(-2, -1, 0, 1, 2)
        #         lagrange_stencil = ntuple(5) do i
        #             xi = stencil_points[i]
        #             prod(j -> j == i ? 1.0 : (p - stencil_points[j])/(xi - stencil_points[j]), 1:5)
        #         end |> SVector

        #         # First type of singularity: 𝒢ⁿ, occurs plasma as source only (see RHS of Chance eqs. 26/27)
        #         if source isa PlasmaGeometry # previously iopw
        #             @. @views greenfunction_mat[j, sing_idx] += wgauss[ig] * G_n * lagrange_stencil
        #         end

        #         # Second type of singularity: 𝒦ⁿ (Eq. 86: 𝒦ⁿαᵢ - δⱼᵢK⁰)
        #         @. @views grad_greenfunction_block[j, sing_idx] += wgauss[ig] * coupling_n * lagrange_stencil
        #         # Subtract off the diverging singular n=0 component
        #         grad_greenfunction_block[j, j] -= coupling_0 * wgauss[ig]
        #     end
        # end

        # # Subtract off analytic singular integral from Chance eq.(75) if plasma-plasma block
        # if observer isa PlasmaGeometry && source isa PlasmaGeometry
        #     @. @views greenfunction_mat[j, sing_idx] -= log_correction / x_obs
        # end
    end

    # Account for normal direction pointing out of vacuum integration region in 𝒦ⁿ ⋅ dS, previously isgn
    # Negative for plasma since dS = ∇ψ J dθdζ and ∇ψ points outward but outward normal is inward
    @views grad_greenfunction .*= (source isa PlasmaGeometry3D ? -1 : 1)
end
