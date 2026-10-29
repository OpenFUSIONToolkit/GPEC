"""
ψ-resolved torque response matrices of a kinetic Euler-Lagrange solve.

The complex plasma energy inside a flux surface ψ of the kinetic solve is the quadratic form
δW(ψ) = ξ†·U₂(ψ)·U₁(ψ)⁻¹·ξ/(2μ₀) of the displacement at ψ; its anti-Hermitian part is the
neoclassical toroidal viscosity torque [Logan 2013, eq. 19]. Referring the displacement back
to the applied control-surface field turns it into a matrix over the applied spectrum, the
torque response matrix, whose Hermitian part is the torque quadratic form and whose
eigenvectors are the applied spectra of extremal torque [Logan 2015, ch. 7].
"""

"""
    torque_response_matrix(ffs, permeability, flux_conform) -> Array{ComplexF64,3}

ψ-resolved torque response `T_xe` of a forward kinetic solve from its stored solution.
`permeability` is the flux-space plasma response operator `P` (Φ_tot = P·Φ_x) and `flux_conform`
the b̃ → Φ operator `R = S·A`, both at the control surface.

Per stored node `k`, with `U₁`, `U₂` the displacement and momentum blocks of `u_store` and
`U₁ₗ` the displacement block at the control surface:

    T_xe(ψ_k) = K†·[U₁(ψ_k)†·U₂(ψ_k)]·K · 2n·i/(2μ₀),   K = U₁ₗ⁻¹·D·P·R,   D = diag(1/(χ₁·(m − n·q_lim)·2π·i))

`K` maps an applied b̃ to the coefficients of the solution basis (b̃ → Φ_x → Φ_tot → boundary ξ
→ coefficients), so `T_xe` is the Fortran `gresp` chain (`dwks`, `bsurfmat`, `gind`, `gres`,
`ptof`) with the single edge inverse the stored, globally consistent basis allows.

Errors unless the solve is a single-`n` forward integration with kinetic matrices; the ideal
Euler-Lagrange energy is Hermitian and carries no torque. Warns when the kinetic matrices are
the synthetic `"fixed"` test matrices.
"""
function torque_response_matrix(ffs::ForceFreeStatesResult, permeability::AbstractMatrix{ComplexF64}, flux_conform::AbstractMatrix{ComplexF64})
    ffs.integrator === :forward ||
        throw(ArgumentError("torque response matrices need the dense forward-integrated solution; this result came from the $(ffs.integrator) integrator"))
    ffs.solution === nothing && throw(ArgumentError("torque response matrices need a ξ solution; the forward integrator produced none"))
    ffs.mats.kinetic === nothing &&
        throw(ArgumentError("torque response matrices need a kinetic solve; the ideal Euler-Lagrange energy is Hermitian and carries no torque"))
    ffs.npert == 1 || throw(ArgumentError("torque response matrices are defined per toroidal harmonic; this solve stacks $(ffs.npert)"))
    ffs.control.kinetic isa ForceFreeStates.TestKineticMatrices &&
        @warn "TestKineticMatrices: the torque response is built from synthetic test matrices and has no physical meaning"

    solution = ffs.solution
    N = ffs.numpert_total
    npsi = solution.step
    nn = ffs.nlow
    μ₀ = 4π * 1e-7
    chi1 = 2π * ffs.equil.psio
    m_modes = [(i - 1) % ffs.mpert + ffs.mlow for i in 1:N]

    # Boundary flux → boundary displacement, Φ_tot = χ₁·(m − n·q_lim)·2π·i·ξ (the same factor Λ carries).
    t = [-im / (chi1 * (m_modes[i] - nn * ffs.qlim) * 2π) for i in 1:N]
    U1_lim = solution.u_store[:, :, 1, npsi]
    K = U1_lim \ (Diagonal(t) * permeability * flux_conform)

    T_xe = Array{ComplexF64}(undef, N, N, npsi)
    prefactor = 2 * nn * im / (2 * μ₀)
    W = Matrix{ComplexF64}(undef, N, N)
    for k in 1:npsi
        mul!(W, adjoint(@view(solution.u_store[:, :, 1, k])), @view(solution.u_store[:, :, 2, k]))
        W .*= prefactor
        T_xe[:, :, k] = K' * W * K
    end
    return T_xe
end

"""
    torque_profile(T_xe::AbstractArray{ComplexF64,3}, b̃) -> Vector{ComplexF64}
    torque_profile(tr::TorqueResponse, b̃) -> Vector{ComplexF64}

Contract an applied root-area-weighted spectrum `b̃` with a torque response, giving the
cumulative complex torque `T(ψ_k) = b̃†·T_xe(ψ_k)·b̃/2` at every stored node: `real` is the
toroidal torque on the plasma inside ψ_k in N·m and `imag` is `2n·δW` of that plasma in J.

The factor 1/2 converts the stored "2×" matrix (the Fortran GPEC `T_xe` array, built on the
quadratic form of the `+n` harmonic alone) to the physical torque of the real field; the
last entry then equals the boundary-response `toroidal_torque` of the same solve. Only the
Hermitian part of `T_xe` contributes to the real part, so the torque is the same whether or
not the matrix is symmetrized first.
"""
function torque_profile(T_xe::AbstractArray{ComplexF64,3}, b̃::AbstractVector{<:Number})
    N = size(T_xe, 1)
    length(b̃) == N || throw(DimensionMismatch("applied spectrum has $(length(b̃)) entries; the response matrix has $N rows"))
    x = Vector{ComplexF64}(b̃)
    return [dot(x, @view(T_xe[:, :, k]), x) / 2 for k in 1:size(T_xe, 3)]
end

torque_profile(tr::TorqueResponse, b̃::AbstractVector{<:Number}) = torque_profile(tr.T_xe, b̃)

"""
    coil_flux_spectra(ffs, coil_sets, cfg) -> Matrix{ComplexF64}

Unit-norm control-surface flux spectrum `Φ_x` of every coil set as built, one column per set
on the solve's (m, n) ordering `[numpert_total × ncoil_set]`: Biot-Savart on the control
surface `ffs.psilim` sampled with the run's own `cfg` grid, so the columns sum to the run's
coil forcing.
"""
function coil_flux_spectra(ffs::ForceFreeStatesResult, coil_sets::Vector{CoilSet}, cfg::ForcingTerms.CoilConfig)
    grids = [(n, ForcingTerms.CoilForcingGrid(ffs.equil, cfg, n; psi=ffs.psilim)) for n in ffs.nlow:ffs.nhigh]
    M = zeros(ComplexF64, ffs.numpert_total, length(coil_sets))
    for (j, cs) in enumerate(coil_sets)
        modes = ForcingMode[]
        for (n, grid) in grids
            append!(modes, ForcingTerms.coil_forcing_modes(cs, grid, n, ffs.mlow, ffs.mhigh))
        end
        M[:, j] = map_forcing_to_eigenmodes(modes, ffs)
    end
    return M
end

"""
    build_torque_response!(resp; verbose=false)

Store the torque response `T_xe` of the solve on `resp`, from its flux-space permeability.
Errors when `resp` carries no plasma response operators.
"""
function build_torque_response!(resp::PlasmaResponse; verbose::Bool=false)
    isempty(resp.permeability_flux) &&
        throw(ArgumentError("compute_torque_response needs the plasma response operators, which this result could not supply"))
    verbose && @info "Computing torque response matrices"
    resp.T_xe = torque_response_matrix(resp.ffs, resp.permeability_flux, resp.rootarea_to_area_weight .* resp.surface_area)
    return resp
end

"""
    apply_torque_response!(state, forcing; verbose=false)

Contract the torque response of `state.response` with the forcing of `state` into
`state.torque_response`. Every forcing source is one column of `T_source`: each coil set
(`state.coil_sets`, on the grid `forcing` carries) when the forcing is coils, otherwise the
applied spectrum as a single source named after its forcing file. A source's Φ_x column is
referred to b̃ through `R = S·A`, so `T_source = B̃†·T_xe·B̃` with `B̃ = R⁻¹·Φ_source` (the Fortran `gcoil`).
"""
function apply_torque_response!(
    state::PerturbedEquilibriumState,
    forcing::Union{ForcingTerms.ForcingTermsControl,ForcingTerms.RMPField};
    verbose::Bool=false
)
    resp = state.response
    ffs = resp.ffs
    ft_ctrl = forcing isa ForcingTerms.ForcingTermsControl ? forcing :
              forcing isa ForcingTerms.RMPSource ? forcing.ctrl : nothing
    source_b = zeros(ComplexF64, ffs.numpert_total, 0)
    source_names = String[]
    if !isempty(state.coil_sets) && ft_ctrl !== nothing && ft_ctrl.forcing_data_format == "coil"
        flux_conform = resp.rootarea_to_area_weight .* resp.surface_area
        source_b = flux_conform \ coil_flux_spectra(ffs, state.coil_sets, ForcingTerms.CoilConfig(ft_ctrl))
        source_names = [cs.name for cs in state.coil_sets]
    elseif !isempty(state.forcing_b_rootarea)
        source_b = reshape(state.forcing_b_rootarea, :, 1)
        source_names = [ft_ctrl === nothing ? "forcing" : basename(ft_ctrl.forcing_data_file)]
    end

    T_xe = resp.T_xe
    npsi = size(T_xe, 3)
    T_source = Array{ComplexF64}(undef, length(source_names), length(source_names), npsi)
    for k in 1:npsi
        T_source[:, :, k] = source_b' * @view(T_xe[:, :, k]) * source_b
    end
    T_applied = isempty(state.forcing_b_rootarea) ? ComplexF64[] : torque_profile(T_xe, state.forcing_b_rootarea)
    state.torque_response = TorqueResponse(ffs.solution.psi_store[1:npsi], copy(resp.m_modes), copy(resp.n_modes), T_xe, source_names, T_source, T_applied)

    if verbose && !isempty(T_applied)
        total = T_applied[end]
        @info "Torque response complete: applied-forcing torque $(@sprintf("%.4e", real(total))) N·m, 2n·δW $(@sprintf("%.4e", imag(total))) J"
    end
    return state
end
