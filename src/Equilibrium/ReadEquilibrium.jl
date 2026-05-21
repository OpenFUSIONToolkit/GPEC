"""
_read_1d_gfile_format(lines_block, num_values)

Internal helper function to parse Fortran-style fixed-width numerical blocks
from a vector of strings.

## Arguments:

  - `lines_block`: A `Vector{String}` containing the lines to parse.
  - `num_values`: The total number of `Float64` values to read from the block.

## Returns:

  - A `Vector{Float64}` containing the parsed values.
"""
function _read_1d_gfile_format(lines_block::Vector{String}, num_values::Int)
    data_str = join(lines_block)
    field_width = 16
    parsed_values = Float64[]
    num_read = 0

    # Ensure the string length is a multiple of the field width for safe processing
    safe_len = (length(data_str) ÷ field_width) * field_width
    for i in 1:field_width:safe_len
        num_read >= num_values && break
        val_str = strip(@view(data_str[i:(i+field_width-1)]))
        if !isempty(val_str)
            try
                push!(parsed_values, parse(Float64, val_str))
                num_read += 1
            catch e
                @warn "Parsing error for substring: '$val_str'. Error: $e. Skipping."
            end
        end
    end

    if num_read < num_values
        @warn "Expected $num_values values, but only read $num_read."
    end
    return parsed_values
end

"""
    _read_efit(equil_in)

Parses an EFIT g-file, creates initial 1D and 2D splines, and bundles
them into a `DirectRunInput` object.

## Arguments:

  - `equil_in`: The `EquilInput` object containing the filename and parameters.

## Returns:

  - A `DirectRunInput` object ready for the direct solver.
"""
function read_efit(config::EquilibriumConfig)
    @info "Processing EFIT g-file: $(config.eq_filename)"
    lines = readlines(config.eq_filename)

    # --- Parse Header ---
    header1_parts = split(lines[1])
    nw = parse(Int, header1_parts[end-1])
    nh = parse(Int, header1_parts[end])
    @info "Parsed from header: nw=$nw, nh=$nh"

    header_vals = _read_1d_gfile_format(lines[2:5], 20)
    rdim, zdim, rcentr, rleft, zmid = header_vals[1:5]
    rmaxis, zmaxis, simag, sibry = header_vals[6:9]

    # --- Parse Data Blocks ---
    current_line_idx = 6
    function parse_block(num_pts)
        num_lines = ceil(Int, num_pts / 5)
        block = lines[current_line_idx:(current_line_idx+num_lines-1)]
        data = _read_1d_gfile_format(block, num_pts)
        current_line_idx += num_lines
        return data
    end

    fpol_data = parse_block(nw)
    fpol_sign = Int(sign(fpol_data[end]))  # sign of toroidal field (before abs is applied below)
    pres_data = parse_block(nw)
    ffprime_data = parse_block(nw)
    pprime_data = parse_block(nw)
    psi_flat_vec = parse_block(nw * nh)
    qprof_data = parse_block(nw)

    psi_rz = reshape(psi_flat_vec, nw, nh)

    # --- Create 1D Profile Spline (sq_in) ---
    psi_norm_grid = range(0.0, 1.0; length=nw)
    sq_fs_nodes = hcat(
        abs.(fpol_data),
        max.(pres_data .* mu0, 0.0),
        qprof_data,
        sqrt.(psi_norm_grid)
    )
    sq_xs = collect(psi_norm_grid)
    sq_in = cubic_interp(sq_xs, Series(sq_fs_nodes); extrap=ExtendExtrap())

    # --- Process and Normalize 2D Psi Data ---
    psio_signed = sibry - simag
    psi_proc = (sibry .- psi_rz)
    psio = abs(psio_signed)
    # Ensure psi at the magnetic axis is positive relative to the boundary
    if psio_signed < 0.0
        psi_proc .*= -1.0
    end

    # --- Create 2D Psi interpolant (psi_in) ---
    r_grid = range(rleft, rleft + rdim; length=nw)
    z_grid = range(zmid - zdim / 2, zmid + zdim / 2; length=nh)
    rmin, rmax = extrema(r_grid)
    zmin, zmax = extrema(z_grid)

    psi_in_xs = collect(r_grid)
    psi_in_ys = collect(z_grid)
    psi_in = cubic_interp((psi_in_xs, psi_in_ys), psi_proc; extrap=ExtendExtrap())

    # --- Bundle everything for the solver ---
    return DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio, fpol_sign)
end


"""
    read_chease_binary(equil_config)

Parses a binary CHEASE file, creates initial 1D and 2D splines with proper
normalization (R0, B0 scaling), and bundles them into a `InverseRunInput` object.
"""
function read_chease_binary(config::EquilibriumConfig)
    @info "Reading CHEASE file (Binary): $(config.eq_filename)"

    R0EXP = config.r0exp
    B0EXP = config.b0exp

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
            [zeros(i == 1 || i == 10 ? npsi1-1 : npsi1) for i in 1:10]

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
        rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), fs_2d[:, :, 1]; bc=(CubicFit(), PeriodicBC(; check=false)), extrap=(ExtendExtrap(), WrapExtrap()))
        rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), fs_2d[:, :, 2]; bc=(CubicFit(), PeriodicBC(; check=false)), extrap=(ExtendExtrap(), WrapExtrap()))

        @info "Finished reading CHEASE equilibrium (Binary)"
        return InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio)
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
    R0EXP = config.r0exp
    B0EXP = config.b0exp

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
                s = strip(line[(22*i+1):min(end, 22*(i+1))])
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

    opts2d = (bc=(CubicFit(), PeriodicBC(; check=false)), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), R_data; opts2d...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), Z_data; opts2d...)
    @info "Finished reading CHEASE equilibrium. Magnetic axis at (ro=$(@sprintf("%.3f", ro)), zo=$(@sprintf("%.3f", zo))), psio=$(@sprintf("%.3e", psio))"
    return InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio)
end

"""
    read_imas(config::EquilibriumConfig, dd)

Load an equilibrium from an IMAS data dictionary and return a `DirectRunInput`.

The `dd.equilibrium.time_slice[]` is used (active time slice). Poloidal flux is
converted from the IMAS COCOS convention (set by `config.imas_cocos`) to the
internal COCOS 2 convention:
  - `imas_cocos = 11` (default, IMAS standard): divide ψ by 2π
  - `imas_cocos = 2` (GPEC internal): no conversion

## Arguments
- `config`: `EquilibriumConfig` with `eq_type = "imas"` and `imas_cocos` set.
- `dd`: populated `IMASdd.dd` with `dd.equilibrium.time_slice[]` containing:
  - `global_quantities.psi_axis`, `global_quantities.psi_boundary`
  - `profiles_1d.psi`, `profiles_1d.f`, `profiles_1d.pressure`, `profiles_1d.q`
  - `profiles_2d[1].grid.dim1` (R), `profiles_2d[1].grid.dim2` (Z), `profiles_2d[1].psi`
"""
function read_imas(config::EquilibriumConfig, dd)
    @info "Processing IMAS equilibrium at global_time = $(dd.global_time) s"

    eqt = dd.equilibrium.time_slice[]

    # COCOS conversion: IMAS standard is COCOS 11 (ψ_IMAS = 2π × ψ_internal)
    # config.imas_cocos controls the expected convention:
    #   11 (default) → divide psi by 2π to get internal (COCOS 2) values
    #    2           → data is already in internal convention, no conversion needed
    if config.imas_cocos == 11
        cocos_factor = 1.0 / (2π)
        @info "Converting IMAS data from COCOS 11 to internal (COCOS 2)"
    elseif config.imas_cocos == 2
        cocos_factor = 1.0
        @info "IMAS data in COCOS 2 (no conversion needed)"
    else
        error("read_imas: unsupported imas_cocos = $(config.imas_cocos). Use 2 or 11.")
    end

    psi_axis = eqt.global_quantities.psi_axis * cocos_factor
    psi_boundary = eqt.global_quantities.psi_boundary * cocos_factor

    psio = abs(psi_boundary - psi_axis)

    if psio < 1e-10
        error("read_imas: |psi_axis - psi_boundary| = $psio is too small. " *
              "Check that dd.equilibrium is properly populated.")
    end

    # Extract 1D profiles, converting psi from COCOS 11 to internal
    psi_1d = eqt.profiles_1d.psi .* cocos_factor
    f_1d = eqt.profiles_1d.f          # F(ψ) = R·Bt [T·m], COCOS-independent
    fpol_sign = Int(sign(f_1d[end])) # F sign at edge → toroidal field direction
    p_1d = eqt.profiles_1d.pressure   # plasma pressure P(ψ) [Pa], COCOS-independent
    q_1d = eqt.profiles_1d.q          # safety factor, COCOS-independent

    nw = length(psi_1d)
    psi_norm_grid = range(0.0, 1.0; length=nw)

    # Build equilibrium source terms for spline interpolation
    # abs(f_1d): F(ψ) can be negative depending on toroidal field direction convention;
    #            take absolute value since GPEC uses magnitude F = R·|Bt|
    sq_fs_nodes = hcat(
        abs.(f_1d),
        max.(p_1d .* mu0, 0.0),
        q_1d,
        sqrt.(psi_norm_grid)
    )
    sq_xs = collect(psi_norm_grid)
    sq_in = cubic_interp(sq_xs, Series(sq_fs_nodes); extrap=ExtendExtrap())

    # 2D ψ(R,Z) map
    if isempty(eqt.profiles_2d)
        error("read_imas: no profiles_2d found in equilibrium time slice. " *
              "Ensure the 2D ψ(R,Z) map is stored in dd.equilibrium.")
    end
    prof2d = eqt.profiles_2d[1]
    r_grid = prof2d.grid.dim1
    z_grid = prof2d.grid.dim2

    psi_rz = prof2d.psi .* cocos_factor

    # Shift so psi_in = 0 at boundary, psi_in = psio at axis
    psi_proc = psi_boundary .- psi_rz
    if psi_boundary - psi_axis < 0.0
        psi_proc .*= -1.0
    end

    rmin, rmax = extrema(r_grid)
    zmin, zmax = extrema(z_grid)

    psi_in_xs = collect(r_grid)
    psi_in_ys = collect(z_grid)
    psi_in = cubic_interp((psi_in_xs, psi_in_ys), psi_proc; extrap=ExtendExtrap())

    @info "IMAS equilibrium loaded:" *
          "\n    psio = $(round(psio; sigdigits=5)) Wb" *
          "\n    1D profile points: nw = $nw" *
          "\n    2D grid: nR = $(length(r_grid)), nZ = $(length(z_grid))" *
          "\n    R ∈ [$(round(rmin; sigdigits=4)), $(round(rmax; sigdigits=4))] m" *
          "\n    Z ∈ [$(round(zmin; sigdigits=4)), $(round(zmax; sigdigits=4))] m"

    return DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio, fpol_sign)
end
