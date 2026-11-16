# The solutions of the Δ' boundary-value problem on the stored radial grid.

"""
    outer_solution_basis(coefficients, propagators, chunks, columns, bases, surfaces, equil, mats, modes) -> OuterSolutionBasis

Rebuild the solutions of the Δ' boundary-value problem [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eq. 37]
at every stored step of the chunk propagators. The cut solution leaves out the leading-order big and small terms
over the two chunks next to each surface.

### Arguments

  - `coefficients`: The BVP's solution vectors, one column per solution, from `compute_delta_prime_matrix`.
  - `propagators`, `chunks`, `columns`, `bases`: From `riccati_eulerlagrange_integration`.
  - `surfaces`: The singular surfaces `chunks` index.
  - `modes`: Mode indexing.
"""
function outer_solution_basis(coefficients::Matrix{ComplexF64}, propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk}, columns, bases::AbstractDict, surfaces::Vector{SingType},
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, modes::ModeIndices)
    (; stored, slots) = columns
    N = modes.numpert_total
    nsol = size(coefficients, 2)
    ngrid = slots[end]
    chunk_coefficients = basis_chunk_coefficients(coefficients, propagators, chunks, columns)

    psi = Vector{Float64}(undef, ngrid)
    q = Vector{Float64}(undef, ngrid)
    xi = Array{ComplexF64}(undef, N, ngrid, nsol)
    xi_deriv = Array{ComplexF64}(undef, N, ngrid, nsol)
    nblas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in eachindex(chunks)
            u = zeros(ComplexF64, N, nsol, 2)
            du = zeros(ComplexF64, N, nsol, 2)
            spline_hint, mats_hint = Ref(1), Ref(1)
            for (j, k) in enumerate(stored[i])
                node = slots[i] + j
                psi[node] = propagators[i].psi[k]
                @views for half in 1:2
                    mul!(u[:, :, half], propagators[i].steps[k][:, :, half], chunk_coefficients[i])
                end
                q[node] = el_derivatives!(du, u, false, equil, mats, modes, psi[node], spline_hint, mats_hint)
                @views xi[:, node, :] .= u[:, :, 1]
                @views xi_deriv[:, node, :] .= du[:, :, 1]
            end
        end
    finally
        BLAS.set_num_threads(nblas)
    end

    i_crossings = findall(c -> c.needs_crossing, chunks)
    xi_cut = copy(xi)
    cut_range = Matrix{Float64}(undef, length(i_crossings), 2)
    for (j, ic) in enumerate(i_crossings)
        surface = surfaces[chunks[ic].ising]
        ipert = 1 + surface.m[1] - modes.mlow + (surface.n[1] - modes.nlow) * modes.mpert
        cut_range[j, 1] = chunks[ic].psi_start
        cut_range[j, 2] = chunks[ic+1].psi_end
        for (i, series) in ((ic, bases[chunks[ic].ising].left_series), (ic + 1, bases[chunks[ic].ising].right_series))
            big = transpose(@view chunk_coefficients[i][ipert, :])
            small = transpose(@view chunk_coefficients[i][ipert+N, :])
            for node in (slots[i]+1):slots[i+1]
                leading = sing_get_ua_res_cut(series, abs(psi[node] - surface.psifac))
                @views xi_cut[:, node, :] .-= leading[:, 1, 1] .* big .+ leading[:, 2, 1] .* small
            end
        end
    end
    return OuterSolutionBasis(psi, q, fill(false, ngrid), xi, xi_deriv, xi_cut, cut_range)
end

"""
    basis_chunk_coefficients(coefficients, propagators, chunks, columns) -> Vector{Matrix{ComplexF64}}

Per chunk, the 2N × nsol matrix that its propagator's stored steps multiply to give the solutions `coefficients`
of the Δ' boundary-value problem there.
"""
function basis_chunk_coefficients(coefficients::Matrix{ComplexF64}, propagators::Vector{ChunkPropagator},
    chunks::Vector{IntegrationChunk}, columns)
    (; axis_state, starts) = columns
    N = size(axis_state, 2)
    matrix(i) = [propagators[i].u[:, :, 1]; propagators[i].u[:, :, 2]]
    i_crossings = findall(c -> c.needs_crossing, chunks)
    msing = length(i_crossings)
    chunk_coefficients = Vector{Matrix{ComplexF64}}(undef, length(chunks))

    # Axis side: the BVP's axis coefficients multiply the orthonormalized `axis_state`; trace them to each chunk's
    # start columns as `restore_solution_columns!` does.
    axis_columns = qr(axis_state).R \ coefficients[1:N, :]
    for i in (i_crossings[1]-1):-1:1
        chunk_end = matrix(i) * starts[i]
        axis_columns = chunk_end[(N+1):(2N), :] \ axis_columns
        chunk_coefficients[i] = starts[i] * axis_columns
    end

    for j in 1:msing
        ic = i_crossings[j]
        # The crossing chunk starts from the left basis; the forward chunks before it, back to the matching point,
        # from the state at their lower end.
        chunk_coefficients[ic] = coefficients[_col_left(j, N), :]
        state = matrix(ic) * chunk_coefficients[ic]
        range_L = j == 1 ? (1:0) : _midpoint_shoot_range(chunks, i_crossings, j, msing; side=:left)
        for i in reverse(range_L)
            state = matrix(i) \ state
            chunk_coefficients[i] = state
        end
        # The chunk after the surface starts from the right basis; the rest multiply on.
        range_R = _midpoint_shoot_range(chunks, i_crossings, j, msing; side=:right)
        chunk_coefficients[ic+1] = coefficients[_col_right(j, N), :]
        for i in range_R[2:end]
            chunk_coefficients[i] = matrix(i - 1) * chunk_coefficients[i-1]
        end
    end
    return chunk_coefficients
end
