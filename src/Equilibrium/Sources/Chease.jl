"""
    read_chease_binary(equil_config)

Parses a binary CHEASE file, creates initial 1D and 2D splines with proper
normalization (R0, B0 scaling), and bundles them into a `InverseRunInput` object.
"""
function read_chease_binary(config::EquilibriumConfig)
    @info "Reading CHEASE file (Binary): $(config.eq_filename)"

    R0EXP = config.r0_norm
    B0EXP = config.b0_norm

    open(config.eq_filename, "r") do io
        seekstart(io)
        read(io, UInt32)
        ntnova = read(io, Int32)
        npsi1 = read(io, Int32)
        nsym = read(io, Int32)
        read(io, UInt32)

        read(io, UInt32)
        axx = [read(io, Float64) for _ in 1:5]
        read(io, UInt32)

        # 1D array allocation
        zcpr, zcppr, zq, zdq, ztmf, ztp, zfb, zfbp, zpsi, zpsim =
            [zeros(i == 1 || i == 10 ? npsi1 - 1 : npsi1) for i in 1:10]

        for arr in (zcpr, zcppr, zq, zdq, ztmf, ztp, zfb, zfbp, zpsi, zpsim)
            read(io, UInt32)
            read!(io, arr)
            read(io, UInt32)
        end

        # --- Normalization (1D) ---
        ztmf .*= (R0EXP * B0EXP)
        zcppr .*= (B0EXP / R0EXP^2)
        psio_norm = zpsi[npsi1] - zpsi[1]
        psio = psio_norm * R0EXP^2 * B0EXP

        ma = npsi1 - 1
        xs = (zpsi .- zpsi[1]) ./ psio_norm

        fs = zeros(npsi1, 4)
        fs[:, 1] .= ztmf
        fs[:, 2] .= zcppr
        fs[:, 3] .= zq

        # Compute cumulative integral of pressure column for normalization using FastInterpolations
        itp_pressure = cubic_interp(xs, fs[:, 2])
        fsi_pressure = FastInterpolations.cumulative_integrate(itp_pressure)
        # Make a writable copy and normalize pressure integral
        fs_copy = copy(fs)
        fs_copy[:, 2] .= (fsi_pressure .- fsi_pressure[ma]) .* psio
        # Create final spline with modified data
        sq_in = cubic_interp(xs, Series(fs_copy); extrap=ExtendExtrap())

        # --- 2D Geometry ---
        mtau = ntnova + 1  # Same with ASCII
        poloidal_start = 3 # This 3 is to skip ghost datas, which are used for derivatives
        poloidal_stop = ntnova + 3

        fs_2d = zeros(npsi1, mtau, 2)
        buffer = zeros(Float64, ntnova + 3, npsi1) # CHEASE binary record size

        # Reading R data
        read(io, UInt32)
        read!(io, buffer)
        read(io, UInt32)
        buffer .*= R0EXP

        # sart reading with ASCII
        ro = buffer[poloidal_start, 1]
        fs_2d[:, :, 1] .= transpose(buffer[poloidal_start:poloidal_stop, :])

        # Reading Z data
        read(io, UInt32)
        read!(io, buffer)
        read(io, UInt32)
        buffer .*= R0EXP

        zo = buffer[poloidal_start, 1]
        fs_2d[:, :, 2] .= transpose(buffer[poloidal_start:poloidal_stop, :])

        # Create separate interpolants for R and Z coordinates
        rz_in_xs = xs
        rz_in_ys = range(0, 1; length=mtau) |> collect
        rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), fs_2d[:, :, 1]; bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
        rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), fs_2d[:, :, 2]; bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

        # Capture the raw arrays that reconstruct sq_in, rz_in_R, rz_in_Z.
        ingest = InverseIngest(collect(xs), fs_copy, collect(rz_in_xs), collect(rz_in_ys), fs_2d[:, :, 1], fs_2d[:, :, 2], ro, zo, psio)

        @info "Finished reading CHEASE equilibrium (Binary)"
        return InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio, ingest)
    end
end


"""
    read_chease_ascii(config)

Parses a ascii CHEASE file, creates initial 1D and 2D splines, finds magnetic axis, and bundles
them into a `InverseRunInput` object.

## Arguments:

  - `config`: The `EquilibriumConfig` object containing the filename and parameters.

## Returns:

  - A `InverseRunInput` object ready for the inverse solver.
"""
function read_chease_ascii(config::EquilibriumConfig)
    @info "Reading CHEASE file (ASCII): $(config.eq_filename)"
    lines = readlines(config.eq_filename)
    R0EXP = config.r0_norm
    B0EXP = config.b0_norm

    # --- Parse Header (FORMAT 10: 3I5) ---
    header_parts = split(lines[1])
    ntnova = parse(Int, header_parts[1])
    npsi1 = parse(Int, header_parts[2])
    nsym = parse(Int, header_parts[3])

    # --- Parse axx (FORMAT 20: 1E22.15) ---
    axx = parse(Float64, split(lines[2])[1])   # RBOXLEN - compuational box lenth
    # --- Pre-allocate Arrays ---
    zcpr = zeros(npsi1 - 1) # normalized P(ψ)
    zcppr = zeros(npsi1) # normalized dP/dψ
    zq = zeros(npsi1) # q(ψ)
    zdq = zeros(npsi1) # dq/dψ
    ztmf = zeros(npsi1) # normalized F(ψ)
    ztp = zeros(npsi1) # normalized dF/dψ
    zfb = zeros(npsi1) # normazlied F(ψ)/q(ψ)
    zfbp = zeros(npsi1) # d/dψ [F(ψ)/q(ψ) ]
    zpsi = zeros(npsi1) # ψ Poloidal flux
    zpsim = zeros(npsi1 - 1) # ψ mid

    zrcp = zeros(ntnova + 3, npsi1) # normalized R
    zzcp = zeros(ntnova + 3, npsi1) # normalized Z
    zjacm = zeros(ntnova + 3, npsi1) # 𝒥(Jacobian)
    zjac = zeros(ntnova + 3, npsi1) # 𝒥(Jacobian

    # --- Helper to parse 5E22.15 data per line ---
    function parse_floats(lines_range)
        data = Float64[]
        for line in lines[lines_range]
            for i in 0:4
                s = strip(line[(22*i+1):min(end, 22 * (i + 1))])
                if !isempty(s)
                    push!(data, parse(Float64, s))
                end
            end
        end
        return data
    end

    # --- Compute line offsets ---
    line_idx = 3  # Start after header (line 1) and axx (line 2)

    function load_vector!(vec)
        count = length(vec)
        lines_needed = cld(count, 5)
        vec .= parse_floats(line_idx:(line_idx+lines_needed-1))
        return line_idx += lines_needed
    end

    function load_matrix!(mat)
        count = size(mat, 1) * size(mat, 2)
        lines_needed = cld(count, 5)
        data = parse_floats(line_idx:(line_idx+lines_needed-1))
        line_idx += lines_needed
        # Fill column-major (Fortran-style)
        for j in 1:size(mat, 2)
            for i in 1:size(mat, 1)
                mat[i, j] = data[(j-1)*size(mat, 1)+i]
            end
        end
    end

    # --- Read Vectors ---
    load_vector!(zcpr)
    load_vector!(zcppr)
    load_vector!(zq)
    load_vector!(zdq)
    load_vector!(ztmf)
    load_vector!(ztp)
    load_vector!(zfb)
    load_vector!(zfbp)
    load_vector!(zpsi)
    load_vector!(zpsim)

    # --- Read Matrices ---
    load_matrix!(zrcp)
    load_matrix!(zzcp)
    load_matrix!(zjacm)
    load_matrix!(zjac)
    @info "Parsed from header: ntnova = $ntnova, npsi1 = $npsi1, nsym = $nsym"

    # --- Apply Normalization ---
    # Scale geometry
    zrcp .*= R0EXP
    zzcp .*= R0EXP

    # Scale flux
    # Psi_phys = Psi_norm * R0^2 * B0
    psio_norm = zpsi[end] - zpsi[1]
    psio = psio_norm * R0EXP^2 * B0EXP

    # Scale Toroidal Field Function F
    # zfb .*= (R0EXP * B0EXP)
    ztmf .*= (R0EXP * B0EXP)

    # Scale Pressure Gradient P'
    zcppr .*= (B0EXP / R0EXP^2)

    # Number of spline intervals
    ma = npsi1 - 1

    # Normalize ψ to [0, 1]
    xs = (zpsi .- zpsi[1]) ./ psio_norm # Use normalized range for x axis [0,1]

    fs = zeros(npsi1, 4)
    # Both zq * zfb and ztmf are the same ! But I don't know why current GPEC follows zq .*zfb - JB.Cho
    # fs[:, 1] .= zq .* zfb
    fs[:, 1] .= ztmf
    fs[:, 2] .= zcppr # normalized Pressure
    fs[:, 3] .= zq # q profile
    # Fit spline with extrapolation boundary condition (bctype = 3)
    # Compute cumulative integral of pressure column for normalization using FastInterpolations
    itp_pressure = cubic_interp(xs, fs[:, 2])
    fsi_pressure = FastInterpolations.cumulative_integrate(itp_pressure)
    # Make a writable copy and normalize pressure integral
    fs_copy = copy(fs)
    fs_copy[:, 2] .= (fsi_pressure .- fsi_pressure[ma]) .* psio
    # Create final spline with modified data
    sq_in = cubic_interp(xs, Series(fs_copy); extrap=ExtendExtrap())

    # --- Copy 2D geometry arrays ---
    mtau = ntnova + 1
    poloidal_start = 3
    poloidal_stop = ntnova + 3
    ro = zrcp[poloidal_start, 1] # Already scaled
    zo = zzcp[poloidal_start, 1] # Already scaled
    rz_in_ys = range(0, 1; length=mtau) |> collect
    # CHEASE includes 2 ghost points at the start (wrap-around); drop them.
    R_data = transpose(zrcp[poloidal_start:poloidal_stop, :])
    Z_data = transpose(zzcp[poloidal_start:poloidal_stop, :])

    # Create separate interpolants for R and Z coordinates
    rz_in_xs = xs

    @views R_data[:, end] .= R_data[:, 1]
    @views Z_data[:, end] .= Z_data[:, 1]

    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), R_data; opts2d...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), Z_data; opts2d...)
    # Capture the raw arrays that reconstruct sq_in, rz_in_R, rz_in_Z.
    ingest = InverseIngest(collect(xs), fs_copy, collect(rz_in_xs), collect(rz_in_ys), Matrix(R_data), Matrix(Z_data), ro, zo, psio)

    @info "Finished reading CHEASE equilibrium. Magnetic axis at (ro=$(@sprintf("%.3f", ro)), zo=$(@sprintf("%.3f", zo))), psio=$(@sprintf("%.3e", psio))"
    return InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio, ingest)
end
