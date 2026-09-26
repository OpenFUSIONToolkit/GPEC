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
    torque_response_matrices(ffs, permeability, flux_conform;
        forcing_b_rootarea=ComplexF64[], coil_flux=zeros(ComplexF64, N, 0), coil_names=String[]) -> TorqueResponse

Build the ψ-resolved torque response of a forward kinetic solve from its stored solution.
`permeability` is the flux-space plasma response operator `P` (Φ_tot = P·Φ_x) and `flux_conform`
the b̃ → Φ operator `R = S·A`, both at the control surface; `coil_flux` holds one unit-norm
Φ_x column per coil set and `forcing_b_rootarea` the run's applied b̃ for `T_applied`.

Per stored node `k`, with `U₁`, `U₂` the displacement and momentum blocks of `u_store` and
`U₁ₗ` the displacement block at the control surface:

    T_xe(ψ_k) = K†·[U₁(ψ_k)†·U₂(ψ_k)]·K · 2n·i/(2μ₀),   K = U₁ₗ⁻¹·D·P·R,   D = diag(1/(χ₁·(m − n·q_lim)·2π·i))

`K` maps an applied b̃ to the coefficients of the solution basis (b̃ → Φ_x → Φ_tot → boundary ξ
→ coefficients), so `T_xe` is the Fortran `gresp` chain (`dwks`, `bsurfmat`, `gind`, `gres`,
`ptof`) with the single edge inverse the stored, globally consistent basis allows.
`T_coil = Kc†·[U₁†U₂]·Kc · 2n·i/(2μ₀)` with `Kc = U₁ₗ⁻¹·D·P·coil_flux` is the Fortran `gcoil`.

Errors unless the solve is a single-`n` forward integration with kinetic matrices; the ideal
Euler-Lagrange energy is Hermitian and carries no torque. Warns when the kinetic matrices are
the synthetic `"fixed"` test matrices.
"""
function torque_response_matrices(
    ffs::ForceFreeStatesResult,
    permeability::AbstractMatrix{ComplexF64},
    flux_conform::AbstractMatrix{ComplexF64};
    forcing_b_rootarea::AbstractVector{ComplexF64}=ComplexF64[],
    coil_flux::AbstractMatrix{ComplexF64}=zeros(ComplexF64, ffs.numpert_total, 0),
    coil_names::Vector{String}=String[]
)::TorqueResponse
    ffs.integrator === :forward ||
        throw(ArgumentError("torque response matrices need the dense forward-integrated solution; this result came from the $(ffs.integrator) integrator"))
    ffs.solution === nothing && throw(ArgumentError("torque response matrices need a ξ solution; the forward integrator produced none"))
    ffs.mats.kinetic === nothing &&
        throw(ArgumentError("torque response matrices need a kinetic solve (kinetic_factor > 0); the ideal Euler-Lagrange energy is Hermitian and carries no torque"))
    ffs.npert == 1 || throw(ArgumentError("torque response matrices are defined per toroidal harmonic; this solve stacks $(ffs.npert) (nn_low != nn_high)"))
    ffs.control.kinetic_source == "fixed" &&
        @warn "kinetic_source = \"fixed\": the torque response is built from synthetic test matrices and has no physical meaning"
    size(coil_flux, 2) == length(coil_names) ||
        throw(DimensionMismatch("coil_flux has $(size(coil_flux, 2)) columns but $(length(coil_names)) coil names were given"))

    solution = ffs.solution
    N = ffs.numpert_total
    npsi = solution.step
    nn = ffs.nlow
    μ₀ = 4π * 1e-7
    chi1 = 2π * ffs.equil.psio
    m_modes = [(i - 1) % ffs.mpert + ffs.mlow for i in 1:N]
    n_modes = fill(nn, N)

    # Boundary flux → boundary displacement, Φ_tot = χ₁·(m − n·q_lim)·2π·i·ξ (the same factor Λ carries).
    t = [-im / (chi1 * (m_modes[i] - nn * ffs.qlim) * 2π) for i in 1:N]
    U1_lim = solution.u_store[:, :, 1, npsi]
    boundary_xi = Diagonal(t) * permeability
    K = U1_lim \ (boundary_xi * flux_conform)
    Kc = U1_lim \ (boundary_xi * coil_flux)

    ncoil = size(coil_flux, 2)
    T_xe = Array{ComplexF64}(undef, N, N, npsi)
    T_coil = Array{ComplexF64}(undef, ncoil, ncoil, npsi)
    prefactor = 2 * nn * im / (2 * μ₀)
    W = Matrix{ComplexF64}(undef, N, N)
    for k in 1:npsi
        mul!(W, adjoint(@view(solution.u_store[:, :, 1, k])), @view(solution.u_store[:, :, 2, k]))
        W .*= prefactor
        T_xe[:, :, k] = K' * W * K
        ncoil > 0 && (T_coil[:, :, k] = Kc' * W * Kc)
    end

    T_applied = isempty(forcing_b_rootarea) ? ComplexF64[] : torque_profile(T_xe, forcing_b_rootarea)
    return TorqueResponse(solution.psi_store[1:npsi], m_modes, n_modes, T_xe, coil_names, T_coil, T_applied)
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
    compute_torque_response!(state, ffs, forcing, intr, ctrl)

Build the torque response of the solve from the plasma response already on `state` and
`intr` and store it in `state.torque_response`. The coil-space matrix is added when the run's
forcing came from coil sets (`intr.coil_sets`) whose grid configuration `forcing` carries;
file forcing and summed fields leave it empty. Errors when the plasma response was not computed.
"""
function compute_torque_response!(
    state::PerturbedEquilibriumState,
    ffs::ForceFreeStatesResult,
    forcing::Union{ForcingTerms.ForcingTermsControl,ForcingTerms.RMPField},
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    isempty(intr.plasma_response) &&
        throw(ArgumentError("compute_torque_response needs the plasma response (compute_response = true)"))
    ctrl.verbose && @info "Computing torque response matrices"
    flux_conform = state.rootarea_to_area_weight .* state.surface_area

    ft_ctrl = forcing isa ForcingTerms.ForcingTermsControl ? forcing :
              forcing isa ForcingTerms.RMPSource ? forcing.ctrl : nothing
    coil_flux = zeros(ComplexF64, ffs.numpert_total, 0)
    coil_names = String[]
    if !isempty(intr.coil_sets) && ft_ctrl !== nothing && ft_ctrl.forcing_data_format == "coil"
        coil_flux = coil_flux_spectra(ffs, intr.coil_sets, ForcingTerms.CoilConfig(ft_ctrl))
        coil_names = [cs.name for cs in intr.coil_sets]
    else
        ctrl.verbose && @info "No coil-set forcing: the coil-space torque response is skipped"
    end

    state.torque_response = torque_response_matrices(ffs, intr.plasma_response, flux_conform;
        forcing_b_rootarea=state.forcing_b_rootarea, coil_flux=coil_flux, coil_names=coil_names)

    if ctrl.verbose
        total = state.torque_response.T_applied[end]
        @info "Torque response complete: applied-forcing torque $(@sprintf("%.4e", real(total))) N·m, 2n·δW $(@sprintf("%.4e", imag(total))) J"
    end
    return state
end
