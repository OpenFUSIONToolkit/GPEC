"""
    _read_fortran_reals(io, n) -> Vector{Float64}

Read one little-endian Fortran sequential unformatted record of `n` reals, stored as real*4 or real*8
(told apart by the record length).
"""
function _read_fortran_reals(io::IO, n::Int)
    nbytes = ltoh(read(io, Int32))
    T = nbytes == 8n ? Float64 : nbytes == 4n ? Float32 : error("Fortran record holds $nbytes bytes, expected $n reals")
    data = ltoh.(read!(io, Vector{T}(undef, n)))
    ltoh(read(io, Int32)) == nbytes || error("Fortran record end marker does not match its start")
    return Float64.(data)
end

"""
    read_ifile(config) -> InverseRunInput

Read an inverse i-file, such as TokaMaker `save_ifile` writes: little-endian Fortran unformatted records,
reals as real*8 or real*4, axis first, θ fastest.

  - `mx, my` (int32); `psi(mx)` [Wb/rad], `f(mx)` = R·Bt [T·m], `p(mx)` [Pa], `q(mx)`
  - `r(my, mx)`, `z(my, mx)` [m], with the periodic θ point duplicated
  - optional `FF′(mx)`, `p′(mx)` per Wb/rad, skipped
  - optional `[ψ_axis, ψ_LCFS]`: ψ_N is then normalized to this LCFS, so the last file surface (written just
    inside it) sits below ψ_N = 1

Port of Fortran `read_eq_ldp_i`.
"""
function read_ifile(config::EquilibriumConfig)
    @info "Reading inverse i-file: $(config.eq_filename)"

    psi, f, p, q, r, z, bounds = open(config.eq_filename, "r") do io
        ltoh(read(io, Int32)) == 8 || error("i-file header record is not two int32")
        mx, my = Int.(ltoh.(read!(io, Vector{Int32}(undef, 2))))
        read(io, Int32)
        profiles = [_read_fortran_reals(io, mx) for _ in 1:4]
        r, z = (reshape(_read_fortran_reals(io, my * mx), my, mx) for _ in 1:2)
        # Optional records, told apart by length: FF′ and p′ (mx each, unused), then the ψ bounds (2)
        bounds = nothing
        while !eof(io)
            nbytes = ltoh(read(io, Int32))
            if nbytes in (4mx, 8mx)
                skip(io, nbytes + 4)
            else
                skip(io, -4)
                bounds = _read_fortran_reals(io, 2)
            end
        end
        return (profiles..., r, z, bounds)
    end
    mx, my = length(psi), size(r, 1)
    @info "Parsed from header: mx = $mx surfaces, my = $my poloidal points"

    psi_axis, psi_lcfs = bounds === nothing ? (psi[1], psi[end]) : bounds
    psio_signed = psi_lcfs - psi_axis
    psio = abs(psio_signed)
    xs = (psi .- psi_axis) ./ psio_signed
    bounds === nothing ||
        @info "i-file ψ bounds: ψ_axis = $psi_axis, ψ_LCFS = $psi_lcfs; file surfaces span ψ_N ∈ [$(xs[1]), $(xs[end])]"

    sq_fs = hcat(abs.(f), p .* mu0, q, sqrt.(xs))
    sq_in = cubic_interp(xs, Series(sq_fs); extrap=ExtendExtrap())

    R_data = Matrix(transpose(r))
    Z_data = Matrix(transpose(z))
    ro, zo = R_data[1, 1], Z_data[1, 1]
    rz_in_ys = collect(range(0, 1; length=my))

    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((xs, rz_in_ys), R_data; opts2d...)
    rz_in_Z = cubic_interp((xs, rz_in_ys), Z_data; opts2d...)
    ingest = InverseIngest(xs, sq_fs, xs, rz_in_ys, R_data, Z_data, ro, zo, psio)

    @info "Finished reading i-file. Magnetic axis at (ro=$(@sprintf("%.3f", ro)), zo=$(@sprintf("%.3f", zo))), psio=$(@sprintf("%.3e", psio))"
    return InverseRunInput(config, sq_in, xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio, ingest)
end
