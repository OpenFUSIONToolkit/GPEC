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
    ip_sign = Int(sign(header_vals[11]))  # the g-file's plasma current; its sign sets the helicity
    ip_sign == 0 && (ip_sign = 1)

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

    # Capture the raw arrays that reconstruct sq_in and psi_in, so the rerun path
    # (gpec.h5 → PlasmaEquilibrium) can skip the g-file parse entirely.
    ingest = DirectIngest(sq_xs, sq_fs_nodes, psi_in_xs, psi_in_ys, psi_proc, rmin, rmax, zmin, zmax, psio, fpol_sign, ip_sign)

    return DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio, fpol_sign, ip_sign, ingest)
end
