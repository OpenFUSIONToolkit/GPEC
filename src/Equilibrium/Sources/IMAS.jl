"""
    read_imas(config::EquilibriumConfig, dd) -> DirectRunInput

Load an equilibrium from an IMAS data dictionary and return a `DirectRunInput`.

The `dd.equilibrium.time_slice[]` is used (active time slice). Poloidal flux is
converted from the IMAS COCOS convention (set by `config.imas_cocos`) to the
internal COCOS 2 convention:

  - `imas_cocos = 11` (default, IMAS standard): divide ψ by 2π
  - `imas_cocos = 2` (GPEC internal): no conversion

## Arguments

  - `config`: `EquilibriumConfig` with `eq_type = "imas"` and `imas_cocos` set.

  - `dd`: populated `IMASdd.dd` with `dd.equilibrium.time_slice[]` containing:

      + `global_quantities.psi_axis`, `global_quantities.psi_boundary`
      + `profiles_1d.psi`, `profiles_1d.f`, `profiles_1d.pressure`, `profiles_1d.q`
      + the `profiles_2d` entry with `grid_type.index == 1` (rectangular), supplying `grid.dim1` (R), `grid.dim2` (Z) and `psi`
"""
function read_imas(config::EquilibriumConfig, dd)
    @info "Processing IMAS equilibrium at global_time = $(dd.global_time) s"

    eqt = dd.equilibrium.time_slice[]

    # COCOS 11 (IMAS standard) stores ψ_IMAS = 2π × ψ_internal
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

    psi_1d = eqt.profiles_1d.psi .* cocos_factor
    f_1d = eqt.profiles_1d.f          # F(ψ) = R·Bt [T·m], COCOS-independent
    p_1d = eqt.profiles_1d.pressure   # plasma pressure P(ψ) [Pa], COCOS-independent
    q_1d = eqt.profiles_1d.q          # safety factor, COCOS-independent

    # Toroidal-field sign from the boundary F, before abs() below
    bt_sign = isempty(f_1d) ? 1 : Int(sign(f_1d[end]))
    bt_sign == 0 && (bt_sign = 1)
    # Plasma-current sign (missing or zero → +1); getproperty with a default because reading an absent ip throws
    ip_imas = getproperty(eqt.global_quantities, :ip, 0.0)
    ip_sign = ip_imas == 0 ? 1 : Int(sign(ip_imas))

    nw = length(psi_1d)
    psi_norm_grid = (psi_1d .- psi_axis) ./ (psi_boundary - psi_axis)
    @assert all(diff(psi_norm_grid) .> 0) "read_imas: profiles_1d.psi must increase strictly from psi_axis to psi_boundary"
    @assert 0 <= psi_norm_grid[1] && psi_norm_grid[end] <= 1 "read_imas: profiles_1d.psi must lie between psi_axis and psi_boundary"

    # GPEC uses |F| = R·|Bt|; the sign is kept in bt_sign
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
    prof2d = findfirst(:rectangular, eqt.profiles_2d)
    prof2d === nothing && error("read_imas: no rectangular profiles_2d found.")
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

    # Raw arrays so an IMAS run can be replayed from gpec.h5 without the dd
    ingest = DirectIngest(sq_xs, sq_fs_nodes, psi_in_xs, psi_in_ys, Matrix(psi_proc), rmin, rmax, zmin, zmax, psio, bt_sign, ip_sign)

    return DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio, bt_sign, ip_sign, ingest)
end
