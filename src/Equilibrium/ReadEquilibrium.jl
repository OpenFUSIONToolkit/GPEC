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
        val_str = strip(data_str[i:(i+field_width-1)])
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
    println("--> Processing EFIT g-file: $(config.control.eq_filename)")
    lines = readlines(config.control.eq_filename)

    # --- Parse Header ---
    header1_parts = split(lines[1])
    nw = parse(Int, header1_parts[end-1])
    nh = parse(Int, header1_parts[end])
    println("--> Parsed from header: nw=$nw, nh=$nh")

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
    sq_in = cubic_interp(sq_xs, sq_fs_nodes; bc=Spl.extrap_bc(sq_xs, sq_fs_nodes[:, 1]), extrap=:extension)

    # --- Process and Normalize 2D Psi Data ---
    psio_signed = sibry - simag
    psi_proc = (sibry .- psi_rz)
    psio = abs(psio_signed)
    # Ensure psi at the magnetic axis is positive relative to the boundary
    if psio_signed < 0.0
        psi_proc .*= -1.0
    end

    # --- Create 2D Psi Spline (psi_in) ---
    r_grid = range(rleft, rleft + rdim; length=nw)
    z_grid = range(zmid - zdim / 2, zmid + zdim / 2; length=nh)
    rmin, rmax = extrema(r_grid)
    zmin, zmax = extrema(z_grid)

    psi_proc_3d = reshape(psi_proc, (nw, nh, 1))
    psi_in = Spl.BicubicSpline(collect(r_grid), collect(z_grid), psi_proc_3d, :extrap, :extrap)

    # --- Bundle everything for the solver ---
    return DirectRunInput(config, sq_in, psi_in, rmin, rmax, zmin, zmax, psio)
end

"""
    _read_chease2(equil_in)

Parses a chease2 file, creates initial 1D and 2D splines, finds magnetic axis, and bundles
them into a `InverseRunInput` object.

## Arguments:

  - `equil_in`: The `EquilInput` object containing the filename and parameters.

## Returns:

  - A `InverseRunInput` object ready for the inverse solver.
"""
function read_chease2(config::EquilibriumConfig)
    println("--> Reading CHEASE file: $(config.control.eq_filename)")
    lines = readlines(config.control.eq_filename)

    # --- Parse Header (FORMAT 10: 3I5) ---
    header_parts = split(lines[1])
    ntnova = parse(Int, header_parts[1])
    npsi1 = parse(Int, header_parts[2])
    nsym = parse(Int, header_parts[3])

    # --- Parse axx (FORMAT 20: 1E22.15) ---
    axx = parse(Float64, split(lines[2])[1])

    # --- Pre-allocate Arrays ---
    zcpr = zeros(npsi1 - 1)
    zcppr = zeros(npsi1)
    zq = zeros(npsi1)
    zdq = zeros(npsi1)
    ztmf = zeros(npsi1)
    ztp = zeros(npsi1)
    zfb = zeros(npsi1)
    zfbp = zeros(npsi1)
    zpsi = zeros(npsi1)
    zpsim = zeros(npsi1 - 1)

    zrcp = zeros(ntnova + 3, npsi1)
    zzcp = zeros(ntnova + 3, npsi1)
    zjacm = zeros(ntnova + 3, npsi1)
    zjac = zeros(ntnova + 3, npsi1)

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

    println("--> Parsed from header:  ntnova = $ntnova, npsi1 = $npsi1, nsym = $nsym")

    # Number of spline intervals
    ma = npsi1 - 1
    # Total ψ range for normalization
    psio = zpsi[end] - zpsi[1]
    # Normalize ψ to [0, 1]
    xs = (zpsi .- zpsi[1]) ./ psio
    # Construct fs matrix: (npsi1 rows, 4 columns)
    fs = zeros(npsi1, 4)
    fs[:, 1] .= zq .* zfb
    fs[:, 2] .= zcppr
    fs[:, 3] .= zq
    # Fit spline with extrapolation boundary condition (bctype = 3)
    # Compute cumulative integral of pressure column for normalization
    fsi_pressure = Spl.cumulative_integral(xs, fs[:, 2]; bc=Spl.extrap_bc(xs, fs[:, 2]))
    # Make a writable copy and normalize pressure integral column (2nd column)
    fs_copy = copy(fs)
    fs_copy[:, 2] .= (fsi_pressure .- fsi_pressure[ma]) .* psio
    # Create final spline with modified data
    sq_in = cubic_interp(xs, fs_copy; bc=Spl.extrap_bc(xs, fs_copy[:, 1]), extrap=:extension)

    # --- Copy 2D geometry arrays ---
    mtau = ntnova + 1
    ro = zrcp[1, 1]
    zo = zzcp[1, 1]
    ys = range(0, 2π; length=mtau) |> collect
    # Allocate and fill fs array (radial × poloidal × 2 quantities)
    fs = zeros(length(xs), length(ys), 2)
    fs[:, :, 1] .= transpose(zrcp[1:(ntnova+1), :])
    fs[:, :, 2] .= transpose(zzcp[1:(ntnova+1), :])


    # Setup bicubic spline with periodic boundary conditions
    rz_in = Spl.BicubicSpline(xs, ys, fs, :extrap, Spl.PeriodicBC())
    println("--> Finished reading CHEASE equilibrium.")
    println("    Magnetic axis at (ro=$ro, zo=$zo), psio=$psio")
    return InverseRunInput(config, sq_in, rz_in, ro, zo, psio)
end

"""
    _read_chease(equil_config)

Parses a binary CHEASE file, creates initial 1D and 2D splines, and bundles
them into a `InverseRunInput` object.

## Arguments:

  - `equil_config`: The `EquilConfig` object containing the filename and parameters.

## Returns:

  - A `InverseRunInput` object ready for the inverse solver.
"""
function read_chease(config::EquilibriumConfig)
    println("--> Reading CHEASE file: $(config.control.eq_filename)")
    diagnostics = false # Set to true to enable detailed print output
    open(config.control.eq_filename, "r") do io
        # Read first 3 integers
        seekstart(io)
        read(io, UInt32)  # skip record length at start
        ntnova = read(io, Int32)
        npsi1 = read(io, Int32)
        nsym = read(io, Int32)
        read(io, UInt32)  # skip record length at end

        if diagnostics
            println("Header:")
            println("  ntnova = $ntnova   Type=$(typeof(ntnova))  Bytes=$(sizeof(ntnova))")
            println("  npsi1  = $npsi1   Type=$(typeof(npsi1))  Bytes=$(sizeof(npsi1))")
            println("  nsym   = $nsym   Type=$(typeof(nsym))  Bytes=$(sizeof(nsym))")
        end

        # Read next 5 Float64 values (axx)
        read(io, UInt32)  # skip record length at start
        axx = [read(io, Float64) for _ in 1:5]
        if diagnostics
            println("\naxx array (expected 5 Float64 values):")
            for (i, val) in enumerate(axx)
                println("  axx[$i] = $val   Type=$(typeof(val))  Bytes=$(sizeof(val))")
            end
        end
        read(io, UInt32)  # skip record length at end

        # --- Helper function ---
        function print_summary(name, arr)
            n = length(arr)
            first5 = arr[1:min(5, n)]
            last5 = arr[max(1, n-4):end]
            println("$name first 5: ", first5)
            return println("$name last 5:  ", last5)
        end

        # --- Pre-allocate Arrays ---
        zcpr = zeros(npsi1 - 1)
        zcppr = zeros(npsi1)
        zq = zeros(npsi1)
        zdq = zeros(npsi1)
        ztmf = zeros(npsi1)
        ztp = zeros(npsi1)
        zfb = zeros(npsi1)
        zfbp = zeros(npsi1)
        zpsi = zeros(npsi1)
        zpsim = zeros(npsi1 - 1)

        # --- Read 1D arrays from file ---
        for (name, arr) in zip(
            ("zcpr", "zcppr", "zq", "zdq", "ztmf", "ztp", "zfb", "zfbp", "zpsi", "zpsim"),
            (zcpr, zcppr, zq, zdq, ztmf, ztp, zfb, zfbp, zpsi, zpsim)
        )
            read(io, UInt32)  # skip record length at start
            read!(io, arr)
            read(io, UInt32)  # skip record length at end
            if diagnostics
                print_summary(name, arr)
            end
        end

        # --- Prepare spline & geometry ---
        ma = npsi1 - 1
        psio = zpsi[npsi1] - zpsi[1]
        xs = (zpsi .- zpsi[1]) ./ psio

        fs = zeros(npsi1, 4)
        fs[:, 1] .= ztmf
        fs[:, 2] .= zcppr
        fs[:, 3] .= zq

        # Compute cumulative integral of pressure column for normalization
        fsi_pressure = Spl.cumulative_integral(xs, fs[:, 2]; bc=Spl.extrap_bc(xs, fs[:, 2]))
        # Make a writable copy and normalize pressure integral column (2nd column)
        fs_copy = copy(fs)
        fs_copy[:, 2] .= (fsi_pressure .- fsi_pressure[ma]) .* psio
        # Create final spline with modified data
        sq_in = cubic_interp(xs, fs_copy; bc=Spl.extrap_bc(xs, fs_copy[:, 1]), extrap=:extension)

        # --- Setup parameters ---
        # PeriodicBC requires a closed grid where fs[:, end, :] == fs[:, 1, :]
        # The binary CHEASE format stores ntnova points (open grid), so we add
        # one extra point to close the grid
        mtau = ntnova + 1

        # Allocate fs array (radial × poloidal × 2) with closed grid
        fs = zeros(npsi1, mtau, 2)

        # Allocate buffer (Fortran: ALLOCATE(buffer(ntnova+3, npsi1)))
        buffer = zeros(Float64, ntnova + 3, npsi1)

        # --- First read (R data) ---
        read(io, UInt32)  # skip record length at start
        read!(io, buffer)                         # READ(in_unit) buffer
        read(io, UInt32)  # skip record length at end
        ro = buffer[1, 1]                         # ro = buffer(1,1)
        if diagnostics
            println("ro = $ro")
        end

        # Fill with r-coordinates (open grid from file)
        fs[:, 1:ntnova, 1] .= transpose(buffer[1:ntnova, :])
        # Close the grid: θ=2π has same values as θ=0
        fs[:, mtau, 1] .= fs[:, 1, 1]

        # --- Second read (Z data) ---
        read(io, UInt32)  # skip record length at start
        read!(io, buffer)                         # READ(in_unit) buffer
        read(io, UInt32)  # skip record length at end
        zo = buffer[1, 1]                         # zo = buffer(1,1)
        if diagnostics
            println("zo = $zo")
        end

        # Fill with z-coordinates (open grid from file)
        fs[:, 1:ntnova, 2] .= transpose(buffer[1:ntnova, :])
        # Close the grid: θ=2π has same values as θ=0
        fs[:, mtau, 2] .= fs[:, 1, 2]

        # Construct ys grid (0..2π, closed with mtau points)
        ys = range(0, 2π; length=mtau) |> collect

        # Setup bicubic spline with periodic boundary conditions
        rz_in = Spl.BicubicSpline(xs, ys, fs, :extrap, Spl.PeriodicBC())

        if diagnostics
            # --- Print first 5 and last 5 entries of each slice ---
            for k in 1:2
                flat = vec(fs[:, :, k])  # flatten to 1D
                n = length(flat)
                println("Slice $k:")
                println("  First 5 entries: ", flat[1:5])
                println("  Last  5 entries: ", flat[(n-4):n])
            end
        end
        println("--> Finished reading CHEASE equilibrium.")
        println("    Magnetic axis at (ro=$ro, zo=$zo), psio=$psio")
        return InverseRunInput(config, sq_in, rz_in, ro, zo, psio)
    end
end
