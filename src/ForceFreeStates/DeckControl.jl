const _ODE_KEYS = (:eulerlagrange_tolerance, :renorm_threshold, :numsteps_init, :numunorms_init, :save_interval, :nchunks)
const _GALERKIN_DECK_KEYS = Dict(String(f) => "gal_" * String(f) for f in fieldnames(Galerkin) if !(f in (:cut_solution, :basis_output)))
_GALERKIN_DECK_KEYS["frobenius_order_ceiling"] = "gal_sing_order_ceiling"

# Translate the current deck layout until the consolidated section parser replaces it.
function control_from_deck(section::AbstractDict, debug::AbstractDict=Dict{String,Any}())
    token = get(section, "integrator", "riccati")
    T = token == "forward" ? Forward : token == "riccati" ? Riccati : token == "galerkin" ? Galerkin :
        error("Unknown integrator: $token")
    numerical = Dict{Symbol,Any}()
    if T === Galerkin
        for (name, key) in _GALERKIN_DECK_KEYS
            haskey(section, key) && (numerical[Symbol(name)] = section[key])
        end
        numerical[:basis_output] = get(debug, "gal_basis_output", false)
    else
        for key in fieldnames(T)
            haskey(section, String(key)) && (numerical[key] = section[String(key)])
        end
    end
    numerical_keys = Set(vcat(["integrator", "extended_precision_bvp"], collect(String.(_ODE_KEYS)), collect(values(_GALERKIN_DECK_KEYS))))
    physical = Dict(Symbol(k) => v for (k, v) in section if !(k in numerical_keys))
    ctrl = ForceFreeStatesControl(; algorithm=T(; numerical...), physical...)
    validate_algorithm(ctrl)
    return ctrl
end

function validate_algorithm(ctrl::ForceFreeStatesControl)
    if ctrl.algorithm isa Galerkin
        for (name, default) in ((:frobenius_psi_max, 0.0), (:truncate_at_dW_peak, false), (:dW_edge_scan_start, 0.99))
            getfield(ctrl, name) == default || throw(ArgumentError("Galerkin does not support non-default `$name`."))
        end
    end
    return ctrl
end

function control_to_deck(ctrl::ForceFreeStatesControl)
    section = Dict{String,Any}(String(f) => getfield(ctrl, f) for f in fieldnames(typeof(ctrl)) if f !== :algorithm && getfield(ctrl, f) !== nothing)
    alg = ctrl.algorithm
    section["integrator"] = String(_integrator_symbol(alg))
    for f in fieldnames(typeof(alg))
        f in (:basis_output, :cut_solution) && continue
        key = alg isa Galerkin ? _GALERKIN_DECK_KEYS[String(f)] : String(f)
        section[key] = getfield(alg, f)
    end
    return section
end

problem_keywords(ctrl::ForceFreeStatesControl) = Dict(f => getfield(ctrl, f) for f in fieldnames(typeof(ctrl)) if !(f in (:algorithm, :nn_low, :nn_high)))
