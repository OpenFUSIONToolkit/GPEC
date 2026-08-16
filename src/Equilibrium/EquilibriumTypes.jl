function symbolize_keys(dict::Dict{String,Any})
    return Dict(Symbol(k) => v for (k, v) in dict)
end

"""
    EquilibriumConfig(...)

An immutable struct containing configuration parameters for equilibrium reconstruction
specified in the input.

## Fields

  - `eq_type::String` - Type of equilibrium file ("efit", "solovev", "lar", etc.)
  - `eq_filename::String` - Path to equilibrium input file
  - `jac_type::String` - Jacobian coordinate type ("hamada", "pest", "equal_arc", "boozer", "park", "custom")
  - `power_bp::Int` - Poloidal field power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `power_b::Int` - Total field power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `power_r::Int` - Major radius power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `power_rc::Int` - Minor radius (rfac = √((R-R₀)²+(Z-Z₀)²)) power exponent for Jacobian (internal, derived from `jac_type`; deprecated as a TOML input key)
  - `jac_custom_power_bp::Int` - Poloidal-field exponent for a custom Jacobian `J = Bp^bp · B^b / (R^r · rfac^rc)`; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_b::Int` - Total-field exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_r::Int` - Major-radius exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `jac_custom_power_rc::Int` - Minor-radius (rfac) exponent for a custom Jacobian; read only when `jac_type = "custom"`, ignored for named types
  - `r0exp::Float64` - Major radius normalization for CHEASE/EQDSK [m]
  - `b0exp::Float64` - On-axis toroidal field normalization for CHEASE/EQDSK [T]
  - `grid_type::String` - Grid type for flux surface discretization ("auto" — two-pass measured-curvature
    refinement when mpsi=0, three-region log layout when mpsi>0; "ldp", "pow1", "uniform";
    "log_asymptotic" is a legacy alias for "auto")
  - `psilow::Float64` - Lower limit of normalized flux coordinate
  - `psihigh::Float64` - Requested upper limit of normalized flux coordinate; the value the
    equilibrium is actually formed on is `EquilibriumParameters.psihigh_resolved`.
  - `mpsi::Int` - Number of radial grid intervals; 0 with grid_type="auto" selects the
    two-pass auto grid: the main driver forms a coarse pass-1 equilibrium, measures its curvature,
    pins knots on rational surfaces, and re-forms on the refined grid. Standalone `setup_equilibrium`
    callers get the coarse pass-1 grid unless they refine via `refined_psi_grid` + `override_psi_nodes`.
  - `psi_accuracy::Float64` - Target relative accuracy τ of splined profile derivatives for the
    two-pass auto grid (knot count scales as τ^(-1/3))
  - `mtheta::Int` - Number of poloidal grid points
  - `newq0::Float64` - Target on-axis safety factor q(0); the q and F profiles are rescaled to
    meet it (0 = use input value, -1 = use the axis extrapolation with its sign flipped)
  - `etol::Float64` - Error tolerance for equilibrium solver
  - `force_termination::Bool` - Terminate after equilibrium setup (skip stability calculations)
  - `use_galgrid::Bool` - Use the same grid as galerkin method
"""
@kwdef struct EquilibriumConfig
    eq_type::String = "efit"
    eq_filename::String = "mypath"
    r0exp::Float64 = 1.0
    b0exp::Float64 = 1.0

    jac_type::String = "hamada"
    power_bp::Int = 0
    power_b::Int = 0
    power_r::Int = 0
    power_rc::Int = 0

    jac_custom_power_bp::Int = 0
    jac_custom_power_b::Int = 0
    jac_custom_power_r::Int = 0
    jac_custom_power_rc::Int = 0

    grid_type::String = "auto"
    psilow::Float64 = 1e-2
    psihigh::Float64 = 0.9995
    mpsi::Int = 0
    psi_accuracy::Float64 = 0.001
    mtheta::Int = 512

    newq0::Float64 = 0.0
    etol::Float64 = 1e-10

    force_termination::Bool = false
    use_galgrid::Bool = true

    # IMAS-specific: expected COCOS convention of the input dd.equilibrium (11=IMAS standard, 2=GPEC internal)
    imas_cocos::Int = 11

    """
    Modified internal constructor that enforces self consistency within the inputs
    """
    # The four `_` slots are the `power_bp/power_b/power_r/power_rc` struct fields, which @kwdef
    # forwards positionally. They are always derived below from `jac_type` (or `jac_custom_power_*`),
    # so their incoming values are ignored (hence `_`).
    function EquilibriumConfig(eq_type, eq_filename, r0exp, b0exp, jac_type, _, _, _, _,
        jac_custom_power_bp, jac_custom_power_b, jac_custom_power_r, jac_custom_power_rc,
        grid_type, psilow, psihigh, mpsi, psi_accuracy, mtheta, newq0, etol,
        force_termination, use_galgrid, imas_cocos)
        if jac_type == "hamada"
            @info "Forcing hamada coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 0
            power_r = 0
            power_rc = 0
        elseif jac_type == "pest"
            @info "Forcing pest coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 0
            power_r = 2
            power_rc = 0
        elseif jac_type == "equal_arc"
            @info "Forcing equal_arc coordinate jacobian exponents: power_*"
            power_b = 0
            power_bp = 1
            power_r = 0
            power_rc = 0
        elseif jac_type == "boozer"
            @info "Forcing boozer coordinate jacobian exponents: power_*"
            power_b = 2
            power_bp = 0
            power_r = 0
            power_rc = 0
        elseif jac_type == "park"
            @info "Forcing park coordinate jacobian exponents: power_*"
            power_b = 1
            power_bp = 0
            power_r = 0
            power_rc = 0
        elseif jac_type == "custom"
            # Custom Jacobian: the user-facing knobs define the exponents.
            power_bp = jac_custom_power_bp
            power_b = jac_custom_power_b
            power_r = jac_custom_power_r
            power_rc = jac_custom_power_rc
            # Normalize to a named type when the powers match, so fast paths are taken.
            if power_b == 0 && power_bp == 0 && power_r == 0 && power_rc == 0
                jac_type = "hamada"
                @info "Recognized hamada jacobian from power exponents"
            elseif power_b == 0 && power_bp == 0 && power_r == 2 && power_rc == 0
                jac_type = "pest"
                @info "Recognized pest jacobian from power exponents"
            elseif power_b == 0 && power_bp == 1 && power_r == 0 && power_rc == 0
                jac_type = "equal_arc"
                @info "Recognized equal_arc jacobian from power exponents"
            elseif power_b == 2 && power_bp == 0 && power_r == 0 && power_rc == 0
                jac_type = "boozer"
                @info "Recognized boozer jacobian from power exponents"
            elseif power_b == 1 && power_bp == 0 && power_r == 0 && power_rc == 0
                jac_type = "park"
                @info "Recognized park jacobian from power exponents"
            else
                @info "Using manual jacobian exponents: power b, bp, r, rc = $(power_b), $(power_bp), $(power_r), $(power_rc)"
            end
        else
            error("Cannot recognize jac_type = $(jac_type)")
        end
        if psihigh > 1.0
            @warn "psihigh = $psihigh exceeds 1.0 (separatrix); clamping to 1.0"
        end
        psihigh = min(psihigh, 1.0)
        return new(eq_type, eq_filename, r0exp, b0exp, jac_type, power_bp, power_b, power_r, power_rc,
            jac_custom_power_bp, jac_custom_power_b, jac_custom_power_r, jac_custom_power_rc,
            grid_type, psilow, psihigh, mpsi, psi_accuracy, mtheta, newq0, etol,
            force_termination, use_galgrid, imas_cocos)
    end
end

"""
Outer constructor for EquilibriumConfig from a parsed TOML dictionary
"""
function EquilibriumConfig(equil_dict::Dict{String,Any}, base_path::String="./")
    # `eq_type` is always required.  `eq_filename` is required for file-based
    # equilibria (efit, chease, …) but optional for analytic types whose
    # parameters live in an embedded `[TJ_ANALYTIC_INPUT]` / `[SOL_INPUT]` /
    # `[LAR_INPUT]` section of the parent gpec.toml.
    if !haskey(equil_dict, "eq_type")
        error("Missing required key in [Equilibrium]: eq_type")
    end

    # Filter to only known parameters
    config_fields = Set(String.(fieldnames(EquilibriumConfig)))
    config_data = Dict{String,Any}()

    for (k, v) in equil_dict
        if k in config_fields
            config_data[k] = v
        else
            @warn "Unknown equilibrium parameter: $k"
        end
    end

    # Only resolve `eq_filename` against `base_path` if the user actually supplied one
    # (otherwise leave the kwdef sentinel for the embedded path). The empty string is the
    # rerun path's "no input file" marker and must stay empty, not become `base_path`.
    if haskey(config_data, "eq_filename") && !isempty(config_data["eq_filename"]) && !isabspath(config_data["eq_filename"])
        config_data["eq_filename"] = normpath(joinpath(base_path, config_data["eq_filename"]))
    end

    return EquilibriumConfig(; symbolize_keys(config_data)...)
end

"""
Outer constructor for EquilibriumConfig that enables a toml file
interface for specifying the configuration settings

DEPRECATED: Use [Equilibrium] section in gpec.toml instead
"""
function EquilibriumConfig(path::String)
    raw = TOML.parsefile(path)

    # Extract EQUIL_CONTROL with default fallback
    config_data = get(raw, "EQUIL_CONTROL", Dict())

    # Check for required fields
    required_keys = ("eq_filename", "eq_type")
    missingkeys = filter(k -> !haskey(config_data, k), required_keys)

    if !isempty(missingkeys)
        error("Missing required key(s) in [EQUIL_CONTROL]: $(join(missingkeys, ", "))")
    end

    # Construct validated struct
    return EquilibriumConfig(Dict{String,Any}(config_data), dirname(path))
end

"""
    LargeAspectRatioConfig(...)

A mutable struct holding parameters for the Large Aspect Ratio (LAR) plasma equilibrium model.

## Fields:

  - `lar_r0`: The major radius of the plasma [m].
  - `lar_a`: The minor radius of the plasma [m].
  - `beta0`: The beta value on axis (normalized pressure).
  - `q0`: The safety factor on axis.
  - `p_pres`: The exponent for the pressure profile, defined as `p00 * (1 - (r / a)^2)^p_pres`.
  - `p_sig`: The exponent that determines the shape of the current-related function profile.
  - `sigma_type`: The type of sigma profile, can be "default" or "wesson". If "wesson", the sigma profile is defined as `sigma0 * (1 - (r / a)^2)^p_sig`.
  - `mtau`: The number of grid points in the poloidal direction.
  - `ma`: The number of grid points in the radial direction.
  - `zeroth`: If set to true, it neglects the Shafranov shift
"""
@kwdef mutable struct LargeAspectRatioConfig
    lar_r0::Float64 = 10.0
    lar_a::Float64 = 1.0
    beta0::Float64 = 1e-3
    q0::Float64 = 1.5
    qa::Float64 = 3.6        # Edge safety factor (legacy field; not consumed by current sigma_type options)
    B0::Float64 = 1.0        # On-axis toroidal field [T] (scales F and P)
    p_pres::Float64 = 2.0
    p_sig::Float64 = 1.0
    sigma_type::String = "default"
    mtau::Int = 128
    ma::Int = 128
    zeroth::Bool = false
end

"Build a `LargeAspectRatioConfig` from a parsed `[LAR_INPUT]` TOML table."
function LargeAspectRatioConfig(input_dict::Dict{String,Any})
    return LargeAspectRatioConfig(; symbolize_keys(input_dict)...)
end

"""
    TJAnalyticConfig(...)

Parameters for the **TJ-analytic** cylindrical large-aspect-ratio equilibrium
model — a GPEC adaptation of the analytic profile family used by
R. Fitzpatrick's TJ code (https://github.com/rfitzp/TJ).  We follow the
same analytic-profile parameterization (ψ-ODE in dimensionless r/a, f₁
for q, power-law pressure) for the inner cylindrical core and connect it
to GPEC's direct-GS pipeline; this is NOT a re-implementation of TJ.

The model uses analytic profiles with exact control of both the on-axis
and edge safety factors. The q profile is determined by:

    f1(r) = [1 - (1-r²)^ν] / (ν·qc)
    q(r)  = r² / f1(r)

where ν = qa/qc is the current peaking parameter, qc is the axis q, and qa
is the edge q. All lengths are normalized to R₀, fields to B₀. The pressure
profile is p₂(r) = pc·(1-r²)^μ.

Reference: R. Fitzpatrick, TJ code, https://github.com/rfitzp/TJ
"""
@kwdef mutable struct TJAnalyticConfig
    lar_r0::Float64 = 10.0     # Major radius R₀ [m]
    lar_a::Float64 = 1.0       # Minor radius a [m] (ε = a/R₀)
    qc::Float64 = 1.5          # On-axis safety factor
    qa::Float64 = 3.6          # Edge safety factor
    pc::Float64 = 0.001        # Normalized on-axis pressure
    mu::Float64 = 2.0          # Pressure peaking exponent: p₂ = pc·(1-r²)^μ
    B0::Float64 = 12.0         # On-axis toroidal field [T]
    ma::Int = 128              # Radial grid points
    mtau::Int = 128            # Poloidal grid points
    zeroth::Bool = false       # If true, suppress Shafranov shift
end

"Build a `TJAnalyticConfig` from a parsed `[TJ_ANALYTIC_INPUT]` TOML table."
function TJAnalyticConfig(input_dict::Dict{String,Any})
    return TJAnalyticConfig(; symbolize_keys(input_dict)...)
end

"""
    SolovevConfig(...)

A mutable struct holding parameters for the Solev'ev (SOL) plasma equilibrium model.

## Fields:

  - `mr`: number of radial grid zones
  - `mz`: number of axial grid zones
  - `ma`: number of flux grid zones
  - `e`:  elongation
  - `a`: minor radius
  - `r0`: major radius
  - `q0`: safety factor at the o-point
  - `p0fac`: scale on-axis pressure (P-> P+P0*p0fac. beta changes. Phi,q constant)
  - `b0fac`: scale toroidal field at constant beta (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
  - `f0fac`: scale toroidal field at constant pressure (s*f. beta,q changes. Phi,p,bp constant)
"""
@kwdef mutable struct SolovevConfig
    mr::Int = 128      # number of radial grid zones
    mz::Int = 128      # number of axial grid zones
    ma::Int = 128      # number of flux grid zones
    e::Float64 = 1.6       # elongation
    a::Float64 = 0.33      # minor radius
    r0::Float64 = 1.0      # major radius
    q0::Float64 = 1.9      # safety factor at the o-point
    p0fac::Float64 = 1       # scale on-axis pressure (P-> P+P0*p0fac. beta changes. Phi,q constant)
    b0fac::Float64 = 1       # scale toroidal field at constant beta (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
    f0fac::Float64 = 1       # scale toroidal field at constant pressure (s*f. beta,q changes. Phi,p,bp constant)
end

"Build a `SolovevConfig` from a parsed `[SOL_INPUT]` TOML table."
function SolovevConfig(input_dict::Dict{String,Any})
    return SolovevConfig(; symbolize_keys(input_dict)...)
end

"""
    DirectIngest

The serializable raw arrays and scalars captured by a direct-equilibrium reader
(`read_efit`, `read_imas`, `sol_run` is analytic and does not use this) — everything
needed to rebuild a `DirectRunInput`'s splines without re-reading the original g-file.
Stored on `DirectRunInput.ingest` / `PlasmaEquilibrium.ingest` and dumped to
`gpec.h5` so a run can be replayed from any directory. The interpolants themselves are
not serializable; they are reconstructed from these nodes by `build_direct_from_ingest`.

## Fields

  - `sq_xs::Vector{Float64}` — normalized-ψ knots for the 1D profile spline
  - `sq_fs::Matrix{Float64}` — 1D profile node values (F, μ₀P, q, √ψ_norm)
  - `psi_xs::Vector{Float64}` — R grid for the 2D flux map [m]
  - `psi_ys::Vector{Float64}` — Z grid for the 2D flux map [m]
  - `psi_rz::Matrix{Float64}` — processed poloidal flux on the (R, Z) grid [Wb/rad]
  - `rmin/rmax/zmin/zmax::Float64` — computational-grid bounds [m]
  - `psio::Float64` — total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `bt_sign::Int` — sign of the toroidal field (+1 or -1)
"""
struct DirectIngest
    sq_xs::Vector{Float64}
    sq_fs::Matrix{Float64}
    psi_xs::Vector{Float64}
    psi_ys::Vector{Float64}
    psi_rz::Matrix{Float64}
    rmin::Float64
    rmax::Float64
    zmin::Float64
    zmax::Float64
    psio::Float64
    bt_sign::Int
end

"""
    InverseIngest

The serializable raw arrays and scalars captured by an inverse-equilibrium reader
(`read_chease_ascii`, `read_chease_binary`) — everything needed to rebuild an
`InverseRunInput`'s splines without re-reading the original CHEASE file. Stored on
`InverseRunInput.ingest` / `PlasmaEquilibrium.ingest` and reconstructed by
`build_inverse_from_ingest`. See [`DirectIngest`](@ref) for the role this plays in
the `gpec.h5` rerun snapshot.

## Fields

  - `sq_xs::Vector{Float64}` — normalized-ψ knots for the 1D profile spline
  - `sq_fs::Matrix{Float64}` — 1D profile node values
  - `rz_xs::Vector{Float64}` — ψ grid for the R, Z maps
  - `rz_ys::Vector{Float64}` — θ grid for the R, Z maps
  - `R_nodes::Matrix{Float64}` — R node values on the (ψ, θ) grid [m]
  - `Z_nodes::Matrix{Float64}` — Z node values on the (ψ, θ) grid [m]
  - `ro::Float64` — R of magnetic axis [m]
  - `zo::Float64` — Z of magnetic axis [m]
  - `psio::Float64` — total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
"""
struct InverseIngest
    sq_xs::Vector{Float64}
    sq_fs::Matrix{Float64}
    rz_xs::Vector{Float64}
    rz_ys::Vector{Float64}
    R_nodes::Matrix{Float64}
    Z_nodes::Matrix{Float64}
    ro::Float64
    zo::Float64
    psio::Float64
end

# Equilibria captured for replay carry one of these; analytic equilibria carry `nothing`
# and are regenerated from their TOML section instead.
const EquilibriumIngest = Union{Nothing,DirectIngest,InverseIngest}

"""
    DirectRunInput(...)

A container struct that bundles all necessary inputs for the `direct_run` function.
It is created by one of the equilibrium file read-in functions after processing the
raw equilibrium data and preparing the initial splines.

## Fields

  - `config::EquilibriumConfig`
    The equilibrium configuration object.

  - `sq_in`
    1D spline data versus normalized poloidal flux `psin`.
    Quantities:

     1. `F = R * B_t` — toroidal flux function [m·T]
     2. `μ₀ * Pressure` — plasma pressure (non-negative) [T²]
     3. `q` — safety factor profile
     4. `√ψ_norm` — square root of normalized flux
  - `psi_in`
    2D cubic interpolant on the (R, Z) grid [m].
    The values correspond to the **poloidal flux** adjusted to be zero at the boundary [Wb/rad].
    Definitions:

     1. `ψ(R, Z) = ψ_boundary - ψ(R, Z)`

     2. `ψ = ψ * sign(ψ(centerR, centerZ))`

          * 1D profiles are represented by `CubicInterpolant` or `CubicSeriesInterpolant`
          * 2D flux surfaces by `CubicInterpolantND`
  - `psi_in_xs::Vector{Float64}` — R coordinate grid for psi_in [m]
  - `psi_in_ys::Vector{Float64}` — Z coordinate grid for psi_in [m]
  - `rmin::Float64` — Minimum R-coordinate of the computational grid [m]
  - `rmax::Float64` — Maximum R-coordinate of the computational grid [m]
  - `zmin::Float64` — Minimum Z-coordinate of the computational grid [m]
  - `zmax::Float64` — Maximum Z-coordinate of the computational grid [m]
  - `psio::Float64` — Total flux difference `|ψ_axis - ψ_boundary|` [Wb/rad]
  - `bt_sign::Int` — Sign of the toroidal field (+1 or -1); read from fpol sign in EFIT g-files
  - `ingest::EquilibriumIngest` — captured raw arrays for the `gpec.h5` rerun snapshot
    (a [`DirectIngest`](@ref) for file-based reads, or `nothing` for analytic equilibria)
  - `psihigh_resolved::Float64` — outer flux limit the equilibrium is formed on: `config.psihigh`
    clamped to the outermost closed flux surface by [`resolve_psihigh!`](@ref). Defaults to
    `config.psihigh` and only differs for efit-family equilibria whose requested limit falls
    outside the closed-flux region. The solvers build their ψ grid from this field. IMAS
    equilibria are read into this struct but are not in `EFIT_KINDS`, so they are never
    clamped and always keep the request.
"""
mutable struct DirectRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S       # 1D profile spline for F, P, q
    psi_in::I2D    # 2D flux interpolant (CubicInterpolantND)
    psi_in_xs::Vector{Float64}  # R coordinates
    psi_in_ys::Vector{Float64}  # Z coordinates
    rmin::Float64    # Minimum R-coordinate of the computational grid [m].
    rmax::Float64    # Maximum R-coordinate of the computational grid [m].
    zmin::Float64    # Minimum Z-coordinate of the computational grid [m].
    zmax::Float64    # Maximum Z-coordinate of the computational grid [m].
    psio::Float64    # The total flux difference |ψ_axis - ψ_boundary| [Weber / radian].
    bt_sign::Int     # Sign of the toroidal field: +1 or -1 (from fpol sign in g-file)
    ingest::EquilibriumIngest
    psihigh_resolved::Float64
end

# Readers construct without a resolved psihigh; it starts at the request and `resolve_psihigh!`
# clamps it for efit-family equilibria.
DirectRunInput(config::EquilibriumConfig, sq_in, psi_in, psi_in_xs, psi_in_ys,
    rmin, rmax, zmin, zmax, psio, bt_sign, ingest) =
    DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys,
        rmin, rmax, zmin, zmax, psio, bt_sign, ingest, config.psihigh)

"""
    InverseRunInput(...)

A container struct for inputs to the `inverse_run` function.

## Fields

  - `config::EquilibriumConfig` - The equilibrium configuration object
  - `sq_in::CubicSeriesInterpolant` - 1D profile spline for F, P, q
  - `rz_in_xs::Vector{Float64}` - ψ coordinate grid for rz_in
  - `rz_in_ys::Vector{Float64}` - θ coordinate grid for rz_in
  - `rz_in_R::CubicInterpolantND` - R coordinate interpolant [m]
  - `rz_in_Z::CubicInterpolantND` - Z coordinate interpolant [m]
  - `ro::Float64` - R-coordinate of magnetic axis [m]
  - `zo::Float64` - Z-coordinate of magnetic axis [m]
  - `psio::Float64` - Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `ingest::EquilibriumIngest` - captured raw arrays for the `gpec.h5` rerun snapshot
    (an [`InverseIngest`](@ref) for file-based reads, or `nothing` for analytic equilibria)
  - `psihigh_resolved::Float64` - outer flux limit the equilibrium is formed on; see
    [`DirectRunInput`](@ref). Equals `config.psihigh` for every inverse reader (CHEASE,
    analytic); `efit_by_inversion` forwards the clamped value from its `DirectRunInput`.
"""
mutable struct InverseRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S   # 1D profile spline for F, P, q
    rz_in_xs::Vector{Float64}   # ψ coordinates
    rz_in_ys::Vector{Float64}   # θ coordinates
    rz_in_R::I2D                # R coordinate interpolant
    rz_in_Z::I2D                # Z coordinate interpolant
    ro::Float64                 # R axis location
    zo::Float64                 # Z axis location
    psio::Float64               # Total flux difference |psi_axis - psi_boundary|
    ingest::EquilibriumIngest
    psihigh_resolved::Float64
end

InverseRunInput(config::EquilibriumConfig, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z,
    ro, zo, psio, ingest) =
    InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z,
        ro, zo, psio, ingest, config.psihigh)

"""
    EquilibriumParameters

A mutable struct containing computed equilibrium parameters and diagnostic flags.

## Fields

  - `ro::Union{Nothing,Float64}` - R-coordinate of the magnetic axis [m]
  - `zo::Union{Nothing,Float64}` - Z-coordinate of the magnetic axis [m]
  - `psio::Union{Nothing,Float64}` - Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `psihigh_resolved::Union{Nothing,Float64}` - Outer flux limit the equilibrium was formed on
    (the outermost ψ node); the plasma edge downstream of `setup_equilibrium`.
  - `rsep::Union{Nothing,Vector{Float64}}` - R-coordinates of the plasma boundary [m]
  - `zsep::Union{Nothing,Vector{Float64}}` - Z-coordinates of the plasma boundary [m]
  - `rext::Union{Nothing,Vector{Float64}}` - R-coordinates of the plasma edge [m]
  - `zext::Union{Nothing,Vector{Float64}}` - Z-coordinates of the plasma edge [m]
  - `psi0::Union{Nothing,Float64}` - Normalized poloidal flux at reference location
  - `b0::Union{Nothing,Float64}` - Total magnetic field strength at the axis [T]
  - `q0::Union{Nothing,Float64}` - Safety factor at the axis
  - `qmin::Union{Nothing,Float64}` - Minimum safety factor in the plasma
  - `qmax::Union{Nothing,Float64}` - Maximum safety factor in the plasma
  - `qa::Union{Nothing,Float64}` - Safety factor at the plasma edge
  - `q95::Union{Nothing,Float64}` - Safety factor at 95% flux surface
  - `qextrema_psi::Union{Nothing,Vector{Float64}}` - Normalized flux values at q extrema
  - `qextrema_q::Union{Nothing,Vector{Float64}}` - Safety factor values at extrema
  - `mextrema::Union{Nothing,Int}` - Number of extrema in q-profile
  - `psi_norm::Union{Nothing,Float64}` - Normalized poloidal flux
  - `b_norm::Union{Nothing,Float64}` - Normalized magnetic field strength
  - `psi_axis::Union{Nothing,Float64}` - Poloidal flux at the axis
  - `psi_boundary::Union{Nothing,Float64}` - Poloidal flux at the boundary
  - `psi_boundary_norm::Union{Nothing,Float64}` - Normalized boundary flux
  - `psi_axis_norm::Union{Nothing,Float64}` - Normalized axis flux
  - `psi_boundary_offset::Union{Nothing,Float64}` - Boundary flux offset
  - `psi_axis_offset::Union{Nothing,Float64}` - Axis flux offset
  - `psi_boundary_sign::Union{Nothing,Int}` - Sign of boundary flux
  - `psi_axis_sign::Union{Nothing,Int}` - Sign of axis flux
  - `psi_boundary_zero::Union{Nothing,Bool}` - Whether boundary flux is zero
  - `rmean::Union{Nothing,Float64}` - Mean major radius [m]
  - `amean::Union{Nothing,Float64}` - Mean minor radius [m]
  - `aratio::Union{Nothing,Float64}` - Aspect ratio (R0/a)
  - `kappa::Union{Nothing,Float64}` - Plasma elongation
  - `delta1::Union{Nothing,Float64}` - Upper triangularity
  - `delta2::Union{Nothing,Float64}` - Lower triangularity
  - `bt0::Union{Nothing,Float64}` - Toroidal field at axis [T]
  - `crnt::Union{Nothing,Float64}` - Plasma current [A]
  - `bwall::Union{Nothing,Float64}` - Toroidal field at wall [T]
  - `verbose::Bool` - Enable verbose output
  - `diagnose_src::Bool` - Enable source data diagnostics
  - `diagnose_maxima::Bool` - Enable extrema diagnostics
  - `volume::Union{Nothing,Float64}` - Plasma volume [m³]
  - `betat::Union{Nothing,Float64}` - Toroidal beta
  - `betan::Union{Nothing,Float64}` - Normalized beta
  - `betaj::Union{Nothing,Float64}` - Total beta
  - `betap1::Union{Nothing,Float64}` - Poloidal beta (definition 1)
  - `betap2::Union{Nothing,Float64}` - Poloidal beta (definition 2)
  - `betap3::Union{Nothing,Float64}` - Poloidal beta (definition 3)
  - `li1::Union{Nothing,Float64}` - Internal inductance (definition 1)
  - `li2::Union{Nothing,Float64}` - Internal inductance (definition 2)
  - `li3::Union{Nothing,Float64}` - Internal inductance (definition 3)
"""
@kwdef mutable struct EquilibriumParameters
    ro::Union{Nothing,Float64} = nothing # R-coordinate of the magnetic axis [m]
    zo::Union{Nothing,Float64} = nothing # Z-coordinate of the magnetic axis [m]
    psio::Union{Nothing,Float64} = nothing # Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
    psihigh_resolved::Union{Nothing,Float64} = nothing # Outer flux limit actually formed on (clamped config.psihigh)
    rsep::Union{Nothing,Vector{Float64}} = nothing # R-coordinates of the plasma boundary [m]
    zsep::Union{Nothing,Vector{Float64}} = nothing # Z-coordinates of the plasma boundary [m]
    rext::Union{Nothing,Vector{Float64}} = nothing # R-coordinates of the plasma edge [m]
    zext::Union{Nothing,Vector{Float64}} = nothing # Z-coordinates of the plasma edge [m]
    psi0::Union{Nothing,Float64} = nothing # Normalized poloidal flux
    b0::Union{Nothing,Float64} = nothing # Total magnetic field strength at the axis [T]
    q0::Union{Nothing,Float64} = nothing # Safety factor at the axis
    qmin::Union{Nothing,Float64} = nothing # Minimum safety factor in the plasma
    qmax::Union{Nothing,Float64} = nothing # Maximum safety factor in the plasma
    qa::Union{Nothing,Float64} = nothing # Safety factor at the plasma edge
    q95::Union{Nothing,Float64} = nothing # Safety factor at 95% flux surface
    qextrema_psi::Union{Nothing,Vector{Float64}} = nothing # Normalized poloidal flux values where q has extrema
    qextrema_q::Union{Nothing,Vector{Float64}} = nothing # Safety factor values at the extrema points
    mextrema::Union{Nothing,Int} = nothing # Number of extrema points in the q-profile
    psi_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    b_norm::Union{Nothing,Float64} = nothing # Normalized total magnetic field strength at the axis
    psi_axis::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    psi_boundary::Union{Nothing,Float64} = nothing # Poloidal flux at the boundary
    psi_boundary_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the boundary
    psi_axis_norm::Union{Nothing,Float64} = nothing # Normalized poloidal flux at the axis
    psi_boundary_offset::Union{Nothing,Float64} = nothing # Offset for the boundary poloidal flux
    psi_axis_offset::Union{Nothing,Float64} = nothing  # Offset for the axis poloidal flux
    psi_boundary_sign::Union{Nothing,Int} = nothing # Sign of the boundary poloidal flux
    psi_axis_sign::Union{Nothing,Int} = nothing # Sign of the axis poloidal flux
    psi_boundary_zero::Union{Nothing,Bool} = nothing # Whether the boundary poloidal flux is zero
    rmean::Union{Nothing,Float64} = nothing # Mean R-coordinate of the plasma [m]
    amean::Union{Nothing,Float64} = nothing # Mean minor radius of the plasma [m]
    aratio::Union{Nothing,Float64} = nothing # Aspect ratio of the plasma (R0/a)
    kappa::Union{Nothing,Float64} = nothing # Elongation of the plasma cross-section
    delta1::Union{Nothing,Float64} = nothing # Triangularity of the plasma cross-section (upper triangularity)
    delta2::Union{Nothing,Float64} = nothing # Triangularity of the plasma cross-section (lower triangularity)
    bt0::Union{Nothing,Float64} = nothing # Toroidal magnetic field at the axis [T] (always positive; sign in bt_sign)
    crnt::Union{Nothing,Float64} = nothing # Plasma current at the axis [A]
    bt_sign::Int = 1 # Sign of the toroidal field: +1 (positive Bt) or -1 (negative Bt, e.g. DIII-D standard)
    bwall::Union{Nothing,Float64} = nothing # Toroidal magnetic field at the wall [T]
    verbose::Bool = false # Whether to print verbose output
    diagnose_src::Bool = false # Whether to diagnose source data
    diagnose_maxima::Bool = false # Whether to diagnose maxima in the equilibrium
    volume::Union{Nothing,Float64} = nothing # Plasma volume [m³]
    betat::Union{Nothing,Float64} = nothing # Toroidal beta (normalized pressure) at the axis
    betan::Union{Nothing,Float64} = nothing # Normalized beta at the axis
    betaj::Union{Nothing,Float64} = nothing # Total beta at the axis
    betap1::Union{Nothing,Float64} = nothing # Pressure beta at the axis
    betap2::Union{Nothing,Float64} = nothing # Toroidal beta at the axis
    betap3::Union{Nothing,Float64} = nothing # Poloidal beta at the axis
    li1::Union{Nothing,Float64} = nothing  # Internal inductance at the axis
    li2::Union{Nothing,Float64} = nothing  # External inductance at the axis
    li3::Union{Nothing,Float64} = nothing  # Total inductance at the axis
end

"""
    ProfileSplines

Named 1D cubic spline interpolants for equilibrium profiles.
Each profile is stored as a separate spline for code clarity.

# Fields

  - `xs::Vector{Float64}`: Shared x-axis (normalized psi)
  - `F_spline`: 2π*F (toroidal flux function, where F = R * B_toroidal)
  - `P_spline`: μ₀*P (plasma pressure × μ₀)
  - `dVdpsi_spline`: dV/dψ (volume derivative)
  - `q_spline`: q (safety factor)

# Derivative Interpolants (for continuous derivative evaluation)

  - `F_deriv`, `P_deriv`, `dVdpsi_deriv`, `q_deriv`: First derivative interpolants

# Notes

  - Node values at grid points: Access via `spline.y[i]`
  - Derivative at any point: Call `deriv(x)` (derivative views are callable)
  - Grid: Access via `xs` field or `spline.cache.x`
"""
struct ProfileSplines{S,D}
    xs::Vector{Float64}
    npts::Int          # length(xs), avoids redundant length() calls
    npts_minus_1::Int  # npts - 1, for hint at last interval
    # Value interpolants
    F_spline::S
    P_spline::S
    dVdpsi_spline::S
    q_spline::S
    # Derivative views (callable, share data with value interpolants)
    F_deriv::D
    P_deriv::D
    dVdpsi_deriv::D
    q_deriv::D
end

"""
    ProfileSplines(xs, F_vals, P_vals, dVdpsi_vals, q_vals; extrap=ExtendExtrap())

Create ProfileSplines from arrays of profile values.
Uses CubicFit boundary conditions with extension extrapolation.
"""
function ProfileSplines(xs::Vector{Float64},
    F_vals::Vector{Float64},
    P_vals::Vector{Float64},
    dVdpsi_vals::Vector{Float64},
    q_vals::Vector{Float64};
    extrap::AbstractExtrap=ExtendExtrap())
    npts = length(xs)
    npts_minus_1 = npts - 1
    @assert length(F_vals) == npts
    @assert length(P_vals) == npts
    @assert length(dVdpsi_vals) == npts
    @assert length(q_vals) == npts

    # Create value interpolants with CubicFit BC (default) for sequential psi access
    F_spline = cubic_interp(xs, F_vals; extrap=extrap)
    P_spline = cubic_interp(xs, P_vals; extrap=extrap)
    dVdpsi_spline = cubic_interp(xs, dVdpsi_vals; extrap=extrap)
    q_spline = cubic_interp(xs, q_vals; extrap=extrap)

    # Create derivative views (these share data with value interpolants, no extra storage)
    F_deriv = deriv1(F_spline)
    P_deriv = deriv1(P_spline)
    dVdpsi_deriv = deriv1(dVdpsi_spline)
    q_deriv = deriv1(q_spline)

    ProfileSplines{typeof(F_spline),typeof(F_deriv)}(
        xs, npts, npts_minus_1,
        F_spline, P_spline, dVdpsi_spline, q_spline,
        F_deriv, P_deriv, dVdpsi_deriv, q_deriv
    )
end

"""
    GeometryProfileSplines

Named 1D cubic spline interpolants for flux-surface-averaged geometric quantities.
Built once during equilibrium construction so they are available to every downstream
module without per-caller recomputation.

# Fields

  - `xs::Vector{Float64}`: Shared ψ axis (matches `rzphi_xs`)
  - `area_spline`: Flux-surface area ∫ dA(ψ) [m²]
  - `avg_r_spline`: Surface-average minor radius ⟨r⟩(ψ) [m]
  - `avg_R_spline`: Surface-average major radius ⟨R⟩(ψ) [m]
"""
struct GeometryProfileSplines{S}
    xs::Vector{Float64}
    npts::Int
    npts_minus_1::Int
    area_spline::S
    avg_r_spline::S
    avg_R_spline::S
end

"""
    GeometryProfileSplines(xs, area_vals, avg_r_vals, avg_R_vals; extrap=ExtendExtrap())

Construct `GeometryProfileSplines` from arrays of surface-averaged values defined
on the shared ψ grid `xs`. Uses CubicFit boundary conditions to match `ProfileSplines`.
"""
function GeometryProfileSplines(xs::Vector{Float64},
    area_vals::Vector{Float64},
    avg_r_vals::Vector{Float64},
    avg_R_vals::Vector{Float64};
    extrap::AbstractExtrap=ExtendExtrap())
    npts = length(xs)
    @assert length(area_vals) == npts
    @assert length(avg_r_vals) == npts
    @assert length(avg_R_vals) == npts

    area_spline = cubic_interp(xs, area_vals; extrap=extrap)
    avg_r_spline = cubic_interp(xs, avg_r_vals; extrap=extrap)
    avg_R_spline = cubic_interp(xs, avg_R_vals; extrap=extrap)

    GeometryProfileSplines{typeof(area_spline)}(
        xs, npts, npts - 1,
        area_spline, avg_r_spline, avg_R_spline,
    )
end

"""
    KineticProfileSplines

Named 1D cubic spline interpolants for kinetic profiles (densities, temperatures,
ExB rotation, collisional diagnostics) loaded from an external `kinetic.dat`
file. Mirrors the `ProfileSplines` pattern: each quantity is its own named
spline plus a paired derivative view for the species the NTV kernel needs
(densities and temperatures, used by `wdian`/`wdiat` in `KineticForces/Torque.jl`).

# Fields

  - `xs::Vector{Float64}`: Shared ψ axis (the regular kinetic grid)
  - `ni_spline`, `ne_spline`: Ion / electron number densities [m⁻³]
  - `Ti_spline`, `Te_spline`: Ion / electron temperatures [J]
  - `omegaE_spline`: ExB rotation ω_E [rad/s]
  - `loglam_spline`: Coulomb logarithm
  - `nui_spline`, `nue_spline`: Krook collision frequencies [s⁻¹]
  - `zeff_spline`: Effective charge Z_eff
  - `ni_deriv`, `ne_deriv`, `Ti_deriv`, `Te_deriv`: Derivative views for ψ-derivative access
"""
struct KineticProfileSplines{S,D}
    xs::Vector{Float64}
    npts::Int
    npts_minus_1::Int
    ni_spline::S
    ne_spline::S
    Ti_spline::S
    Te_spline::S
    omegaE_spline::S
    loglam_spline::S
    nui_spline::S
    nue_spline::S
    zeff_spline::S
    ni_deriv::D
    ne_deriv::D
    Ti_deriv::D
    Te_deriv::D
end

"""
    KineticProfileSplines(xs, ni, ne, Ti, Te, omegaE, loglam, nui, nue, zeff;
                          extrap=ExtendExtrap())

Build the kinetic profile splines from arrays of values defined on the shared
ψ grid `xs`. Uses CubicFit boundary conditions to match `ProfileSplines`.
Temperatures must already be in Joules.
"""
function KineticProfileSplines(xs::Vector{Float64},
    ni::Vector{Float64}, ne::Vector{Float64},
    Ti::Vector{Float64}, Te::Vector{Float64},
    omegaE::Vector{Float64}, loglam::Vector{Float64},
    nui::Vector{Float64}, nue::Vector{Float64}, zeff::Vector{Float64};
    extrap::AbstractExtrap=ExtendExtrap())
    npts = length(xs)
    @assert all(length(v) == npts for v in (ni, ne, Ti, Te, omegaE, loglam, nui, nue, zeff))

    ni_spline = cubic_interp(xs, ni; extrap=extrap)
    ne_spline = cubic_interp(xs, ne; extrap=extrap)
    Ti_spline = cubic_interp(xs, Ti; extrap=extrap)
    Te_spline = cubic_interp(xs, Te; extrap=extrap)
    omegaE_spline = cubic_interp(xs, omegaE; extrap=extrap)
    loglam_spline = cubic_interp(xs, loglam; extrap=extrap)
    nui_spline = cubic_interp(xs, nui; extrap=extrap)
    nue_spline = cubic_interp(xs, nue; extrap=extrap)
    zeff_spline = cubic_interp(xs, zeff; extrap=extrap)

    ni_deriv = deriv1(ni_spline)
    ne_deriv = deriv1(ne_spline)
    Ti_deriv = deriv1(Ti_spline)
    Te_deriv = deriv1(Te_spline)

    KineticProfileSplines{typeof(ni_spline),typeof(ni_deriv)}(
        xs, npts, npts - 1,
        ni_spline, ne_spline, Ti_spline, Te_spline,
        omegaE_spline, loglam_spline, nui_spline, nue_spline, zeff_spline,
        ni_deriv, ne_deriv, Ti_deriv, Te_deriv,
    )
end

"""
    PlasmaEquilibrium(...)

The final, self-contained result of the equilibrium reconstruction.
This object provides a complete representation of the processed plasma equilibrium in flux coordinates.

# Fields

  - `config::EquilibriumConfig`:
    The equilibrium configuration object used for the reconstruction.

  - `params::EquilibriumParameters`:
    Computed equilibrium parameters and diagnostics.
  - `profiles::ProfileSplines`:
    Named 1D profile splines (F, P, dV/dψ, q) on normalized psi grid.
    Access values at grid points via `profiles.F_spline.y[i]`, etc.
    Access derivatives via `profiles.F_deriv.y[i]` or `profiles.F_deriv(psi)`.
  - `geometry::GeometryProfileSplines`:
    Named 1D splines for flux-surface-averaged geometry (area, ⟨r⟩, ⟨R⟩),
    populated automatically by `compute_geometry_profiles` during construction.
  - **Grid coordinates (shared by all rzphi/eqfun interpolants):**

      + `rzphi_xs::Vector{Float64}`: ψ coordinates (length mpsi+1)
      + `rzphi_ys::Vector{Float64}`: θ coordinates (length mtheta+1)
  - **Geometric quantities (rzphi, 4 interpolants):**
    2D cubic interpolants for flux-coordinate mapping with periodic BC in theta.

      + **x value:** normalized ψ
      + **y value:** SFL poloidal angle ∈ [0, 1]
      + `rzphi_rsquared::CubicInterpolantND`: r_coord² = (R - ro)² + (Z - zo)²
      + `rzphi_offset::CubicInterpolantND`: η/(2π) - θₙₑw (angle offset)
      + `rzphi_nu::CubicInterpolantND`: ν in ϕ = 2πζ + ν(ψ, θ)
      + `rzphi_jac::CubicInterpolantND`: Jacobian
  - **Physics quantities (eqfun, 3 interpolants):**
    2D cubic interpolants storing local physics and geometric quantities.

      + **x value:** normalized ψ
      + **y value:** SFL poloidal angle θₙₑw
      + `eqfun_B::CubicInterpolantND`: Total magnetic field strength [T]
      + `eqfun_metric1::CubicInterpolantND`: (e₁⋅e₂ + q⋅e₃⋅e₁)/(J⋅B²)
      + `eqfun_metric2::CubicInterpolantND`: (e₂⋅e₃ + q⋅e₃⋅e₃)/(J⋅B²)
  - `ro::Float64`: R-coordinate of the magnetic axis [m]
  - `zo::Float64`: Z-coordinate of the magnetic axis [m]
  - `psio::Float64`: Total flux difference |Ψ_axis - Ψ_boundary| [Weber/radian]
  - `ingest::EquilibriumIngest`: raw arrays forwarded from the equilibrium input for the
    `gpec.h5` rerun snapshot — a [`DirectIngest`](@ref)/[`InverseIngest`](@ref) for file-based
    equilibria, or `nothing` for analytic ones (regenerated from their TOML section on replay)
"""
mutable struct PlasmaEquilibrium{P<:ProfileSplines,G<:GeometryProfileSplines,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    params::EquilibriumParameters
    profiles::P
    geometry::G

    # Grid coordinates (shared by all 2D interpolants)
    rzphi_xs::Vector{Float64}
    rzphi_ys::Vector{Float64}

    # Geometric quantities (4 interpolants)
    rzphi_rsquared::I2D
    rzphi_offset::I2D
    rzphi_nu::I2D
    rzphi_jac::I2D

    # Physics quantities (3 interpolants)
    eqfun_B::I2D
    eqfun_metric1::I2D
    eqfun_metric2::I2D

    ro::Float64
    zo::Float64
    psio::Float64

    ingest::EquilibriumIngest
end

# Solvers build the equilibrium before setup_equilibrium forwards eq_input.ingest, so allow
# construction without it; ingest defaults to nothing and is assigned post-construction.
PlasmaEquilibrium(config, params, profiles, geometry, rzphi_xs, rzphi_ys,
    rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
    eqfun_B, eqfun_metric1, eqfun_metric2, ro, zo, psio) =
    PlasmaEquilibrium(config, params, profiles, geometry, rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2, ro, zo, psio, nothing)
