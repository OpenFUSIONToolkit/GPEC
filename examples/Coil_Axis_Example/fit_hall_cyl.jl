# fit_hall_cyl.jl — pose-blind iterative fitter for cylindrical Hall files.
#
# Consumes a Hall CSV (R, φ, Z, B_R, B_φ, B_Z) with no pose metadata, plus a
# coil .dat file. Iteratively shifts and tilts the coil to minimise the
# field-difference residual computed on B_R and B_Z only (B_φ is ignored —
# for near-axisymmetric coils it is small enough that measurement noise
# would dominate that channel of the residual).

include(joinpath(@__DIR__, "synth_hall_cyl.jl"))

using Statistics
using Printf
using Random

"""
    load_hall_cyl_csv(path) -> NamedTuple

Read a `synth_hall_cyl`-style CSV. Skips any `#` comment header (and
therefore any pose information that might have been written by the
generator). Returns flat `Vector{Float64}` for `R, phi, Z, B_R, B_phi, B_Z`.
"""
function load_hall_cyl_csv(path::AbstractString)
    raw = readlines(path)
    # Strip comments
    body = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), raw)
    header = split(strip(body[1]), ",")
    col = Dict(strip(c) => j for (j, c) in enumerate(header))
    for needed in ("R", "phi", "Z", "B_R", "B_phi", "B_Z")
        haskey(col, needed) || error("CSV missing required column: $needed")
    end
    rows = body[2:end]
    N = length(rows)
    R   = zeros(N); phi = zeros(N); Z   = zeros(N)
    BR  = zeros(N); BP  = zeros(N); BZ  = zeros(N)
    for (i, line) in enumerate(rows)
        fields = split(strip(line), ",")
        R[i]   = parse(Float64, fields[col["R"]])
        phi[i] = parse(Float64, fields[col["phi"]])
        Z[i]   = parse(Float64, fields[col["Z"]])
        BR[i]  = parse(Float64, fields[col["B_R"]])
        BP[i]  = parse(Float64, fields[col["B_phi"]])
        BZ[i]  = parse(Float64, fields[col["B_Z"]])
    end
    return (R=R, phi=phi, Z=Z, B_R=BR, B_phi=BP, B_Z=BZ)
end

"""
    residual_BR_BZ(BR_pred, BZ_pred, BR_meas, BZ_meas) -> Float64

Sum over probes of `(B_R_pred − B_R_meas)² + (B_Z_pred − B_Z_meas)²`.
B_φ is intentionally not included.
"""
function residual_BR_BZ(BR_pred, BZ_pred, BR_meas, BZ_meas)
    s = 0.0
    @inbounds for i in eachindex(BR_pred)
        s += (BR_pred[i] - BR_meas[i])^2 + (BZ_pred[i] - BZ_meas[i])^2
    end
    return s
end

"""
    fit_pose_BR_BZ(coil_file, hall; init_pose=nothing, current_A=100.0,
                   step0_m=0.050, step0_deg=0.5, tol_m=1e-6, tol_deg=1e-4,
                   max_iter=200, verbose=true,
                   fwd_noise_floor_T=0.0, fwd_noise_rel_frac=0.0,
                   fwd_noise_seed=nothing)

5-DOF coordinate descent over `(x, y, z, tilt_x_deg, tilt_y_deg)` that
minimises the B_R + B_Z residual against `hall`. tilt_z is not searched
(B_R and B_Z don't constrain it for a near-axisymmetric coil).

`fwd_noise_*` is a Monte-Carlo objective hook: when non-zero, the
forward-modelled field has noise added before being compared to `hall`,
using a local `MersenneTwister(fwd_noise_seed)`. Default `0.0` ⇒ purely
deterministic forward model. The seed is stored in the returned result
regardless, so callers can document which seed accompanied the fit even
when no noise was injected.

Returns `(pose, history, n_eval, n_iter, converged, residual, fwd_noise)`
where `history` is a `Vector{NamedTuple}` of `(iter, n_eval, residual,
pose)` recording the best-so-far state at the end of every outer
iteration (entry 1 = initial pose).
"""
function fit_pose_BR_BZ(coil_file::AbstractString, hall;
        init_pose=nothing, current_A::Real=100.0,
        step0_m::Real=0.050, step0_deg::Real=0.5,
        tol_m::Real=1e-6, tol_deg::Real=1e-4,
        max_iter::Int=200, verbose::Bool=true,
        fwd_noise_floor_T::Real=0.0, fwd_noise_rel_frac::Real=0.0,
        fwd_noise_seed::Union{Integer,Nothing}=nothing)

    cs_template = read_coil_dat(coil_file)
    cs_template.currents .= current_A
    x_orig = copy(cs_template.x)
    y_orig = copy(cs_template.y)
    z_orig = copy(cs_template.z)
    coils  = [cs_template]

    # Pose components 1..3 are RELATIVE shifts (dx, dy, dz) in metres
    # — same convention as `place_coil!`. Default init: no shift, no tilt.
    pose = init_pose === nothing ? [0.0, 0.0, 0.0, 0.0, 0.0] :
                                   collect(Float64, init_pose)
    steps = [Float64(step0_m), Float64(step0_m), Float64(step0_m),
             Float64(step0_deg), Float64(step0_deg)]

    N  = length(hall.R)
    BR = zeros(N); BP = zeros(N); BZ = zeros(N)
    R  = collect(Float64, hall.R)
    P  = collect(Float64, hall.phi)
    Z  = collect(Float64, hall.Z)

    # Optional forward-model noise injection (default off)
    use_fwd_noise = fwd_noise_floor_T > 0 || fwd_noise_rel_frac > 0
    rng = fwd_noise_seed === nothing ? MersenneTwister() :
                                       MersenneTwister(fwd_noise_seed)
    floor2_fwd = Float64(fwd_noise_floor_T)^2
    rel_fwd    = Float64(fwd_noise_rel_frac)

    function eval_residual(p)
        cs_template.x .= x_orig
        cs_template.y .= y_orig
        cs_template.z .= z_orig
        place_coil!(coils, p[1], p[2], p[3]; tx_deg=p[4], ty_deg=p[5])
        fill!(BR, 0.0); fill!(BP, 0.0); fill!(BZ, 0.0)
        compute_biot_savart_boundary!(BR, BP, BZ, R, P, Z, coils)
        if use_fwd_noise
            @inbounds for i in 1:N
                Bmag = sqrt(BR[i]^2 + BP[i]^2 + BZ[i]^2)
                σ = sqrt(floor2_fwd + (rel_fwd * Bmag)^2)
                BR[i] += randn(rng) * σ
                BZ[i] += randn(rng) * σ
            end
        end
        return residual_BR_BZ(BR, BZ, hall.B_R, hall.B_Z)
    end

    best = eval_residual(pose)
    n_eval = 1
    iter = 0
    converged = false
    history = NamedTuple[]
    push!(history, (iter=0, n_eval=n_eval, residual=best, pose=copy(pose)))

    while iter < max_iter
        if all(@view(steps[1:3]) .< tol_m) && all(@view(steps[4:5]) .< tol_deg)
            converged = true
            break
        end
        improved = false
        for i in 1:5, sgn in (1.0, -1.0)
            trial = copy(pose); trial[i] += sgn * steps[i]
            val = eval_residual(trial)
            n_eval += 1
            if val < best
                pose, best = trial, val
                improved = true
            end
        end
        improved || (steps .*= 0.5)
        iter += 1
        push!(history, (iter=iter, n_eval=n_eval, residual=best, pose=copy(pose)))
    end

    if verbose
        println("\nfit_pose_BR_BZ result:")
        @printf("  iter=%d, eval=%d, converged=%s, residual=%.6e T²\n",
                iter, n_eval, converged, best)
        @printf("  pose: x=%.6f m, y=%.6f m, z=%.6f m, tx=%.6f deg, ty=%.6f deg\n",
                pose[1], pose[2], pose[3], pose[4], pose[5])
    end

    return (pose=pose, history=history, n_eval=n_eval, n_iter=iter,
            converged=converged, residual=best,
            fwd_noise=(floor_T=Float64(fwd_noise_floor_T),
                       rel_frac=Float64(fwd_noise_rel_frac),
                       seed=fwd_noise_seed))
end

# ───────────────────────────────────────────────────────────────────────────
# Demo
# ───────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    csv_path  = joinpath(@__DIR__, "csvs", "synth_hall_demo.csv")
    coil_file = joinpath(@__DIR__, "..", "examples", "Br_3D_example", "sparc_pf1u.dat")
    if !isfile(csv_path)
        error("$csv_path does not exist. Run synth_hall_cyl.jl first to " *
              "generate the demo input.")
    end
    hall = load_hall_cyl_csv(csv_path)
    @printf("Loaded %d probes from %s\n", length(hall.R), csv_path)
    @printf("  median |B_R|=%.3e T, |B_phi|=%.3e T, |B_Z|=%.3e T\n",
            median(abs.(hall.B_R)), median(abs.(hall.B_phi)),
            median(abs.(hall.B_Z)))
    result = fit_pose_BR_BZ(coil_file, hall; current_A=100.0,
                            max_iter=80, verbose=true)
    @printf("\nFinal residual after %d evals: %.6e T²\n",
            result.n_eval, result.residual)
end
