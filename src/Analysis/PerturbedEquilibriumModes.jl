"""
    PerturbedEquilibriumModes

Post-processing utilities for converting modal GPEC outputs to theta-space
and (theta, phi) grids.
"""
module PerturbedEquilibriumModes

using HDF5
using FastInterpolations
using ...Utilities.FourierTransforms: FourierTransform, inverse

"""
    modes_to_theta(h5_file, variable; mtheta=nothing, keep_sfl_phi=true)

Convert modal output `(npsi, numpert_total)` from a gpec.h5 file to theta-space
`(npsi, mtheta, npert)`.

Reads all required metadata (mlow, nlow, mpert, npert) and spline data (ν) from
the HDF5 file.

# Arguments
- `h5_file::String`: Path to gpec.h5 output file
- `variable::String`: HDF5 dataset path, e.g. `"perturbed_equilibrium/response/xi_R"`

# Keyword arguments
- `mtheta::Int`: theta grid resolution (default: `max(2*(|mlow|+mpert), 512)`)
- `keep_sfl_phi::Bool`: if `true` (default), output in SFL toroidal angle;
  if `false`, apply `exp(i*n*ν(ψ,θ))` to convert to machine toroidal angle
  and conjugate if `helicity > 0` (matches Fortran `gpout_xbrzphifun`)

# Returns
- `theta_data::Array{ComplexF64,3}`: `[npsi × mtheta × npert]`
- `theta_grid::Vector{Float64}`: `[mtheta]` SFL theta ∈ [0, 1)
- `n_vals::Vector{Int}`: `[npert]` toroidal mode numbers

!!! note
    When used on the cylindrical components `xi_R`, `xi_Z`, `xi_phi`, `b_R`, `b_Z`,
    `b_phi` these are currently in beta and show up to ~20% discrepancies vs Fortran.
"""
function modes_to_theta(h5_file::String, variable::String;
                        mtheta::Union{Int,Nothing}=nothing,
                        keep_sfl_phi::Bool=true)
    h5open(h5_file, "r") do f
        modes = read(f, variable)  # (npsi, numpert_total)
        npsi, numpert_total = size(modes)

        mlow  = read(f, "info/mlow")
        nlow  = read(f, "info/nlow")
        mpert = read(f, "info/mpert")
        npert = read(f, "info/npert")
        @assert numpert_total == mpert * npert "Expected numpert_total=$(mpert*npert), got $numpert_total"

        n_vals = [nlow + k - 1 for k in 1:npert]
        mth = isnothing(mtheta) ? max(2 * (abs(mlow) + mpert), 512) : mtheta
        ft = FourierTransform(mth, mpert, mlow)
        theta_grid = [(j - 1) / mth for j in 1:mth]

        theta_data = zeros(ComplexF64, npsi, mth, npert)
        for k in 1:npert
            col_range = (k-1)*mpert+1 : k*mpert
            for ipsi in 1:npsi
                theta_data[ipsi, :, k] .= inverse(ft, view(modes, ipsi, col_range))
            end
        end

        if !keep_sfl_phi
            # Reconstruct ν spline from stored grid + nodal values (FastInterpolations v0.4 API)
            rzphi_xs = read(f, "splines/rzphi/xs")
            rzphi_ys = read(f, "splines/rzphi/ys")
            nu_vals  = read(f, "splines/rzphi/nu")
            nu_spline = cubic_interp(
                (rzphi_xs, rzphi_ys), nu_vals;
                bc=(CubicFit(), PeriodicBC(; check=false)),
                extrap=(ExtendExtrap(), WrapExtrap())
            )

            psi_grid = read(f, "integration/psi")

            bt_sign  = haskey(f, "equil/bt_sign") ? read(f, "equil/bt_sign") : 1
            crnt     = haskey(f, "equil/crnt")     ? read(f, "equil/crnt")    : 1.0
            helicity = bt_sign * Int(sign(crnt))

            hint = (Ref(1), Ref(1))
            for k in 1:npert
                nn = n_vals[k]
                for ipsi in 1:npsi
                    psi = psi_grid[ipsi]
                    for itheta in 1:mth
                        nu = nu_spline((psi, theta_grid[itheta]); hint=hint)
                        theta_data[ipsi, itheta, k] *= exp(im * nn * nu)
                    end
                end

                if helicity > 0
                    theta_data[:, :, k] .= conj.(theta_data[:, :, k])
                end
            end
        end

        return theta_data, theta_grid, n_vals
    end
end

"""
    theta_to_thetaphi(theta_data, n_vals; nphi=nothing)

Extend theta-space data to a `(θ, φ)` grid via toroidal inverse DFT.

    f(θ, φ) = Σₙ fₙ(θ) exp(i·n·φ)

# Arguments
- `theta_data::Array{ComplexF64,3}`: `[npsi × mtheta × npert]` from `modes_to_theta`
- `n_vals::Vector{Int}`: toroidal mode numbers

# Keyword arguments
- `nphi::Int`: toroidal grid points (default: `max(4*maximum(abs.(n_vals)), 64)`)

# Returns
- `full_data::Array{ComplexF64,3}`: `[npsi × mtheta × nphi]`
- `phi_grid::Vector{Float64}`: `[nphi]` in radians ∈ [0, 2π)
"""
function theta_to_thetaphi(theta_data::Array{ComplexF64,3}, n_vals::Vector{Int};
                           nphi::Union{Int,Nothing}=nothing)
    npsi, mth, npert = size(theta_data)
    np = isnothing(nphi) ? max(4 * maximum(abs.(n_vals)), 64) : nphi
    phi_grid = [(j - 1) * 2π / np for j in 1:np]

    full_data = zeros(ComplexF64, npsi, mth, np)
    for k in 1:npert
        nn = n_vals[k]
        for jphi in 1:np
            phase = exp(im * nn * phi_grid[jphi])
            full_data[:, :, jphi] .+= theta_data[:, :, k] .* phase
        end
    end

    return full_data, phi_grid
end

end # module PerturbedEquilibriumModes
