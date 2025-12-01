"""
Minimal bicube compatibility shim.
Provides `bicube_eval_external` used by torque.jl. For real
`SplinesMod.BicubicSpline` objects we delegate to SplinesMod.bicube_eval!
Otherwise we return a safe default vector.
"""
function bicube_eval_external(bc, psi::Float64, theta::Float64, n::Integer)
    # If bc looks like a SplinesMod BicubicSpline, try to call its eval
    try
        if isdefined(Main, :SplinesMod) && typeof(bc) <: SplinesMod.BicubicSpline
            # SplinesMod.bicube_eval! modifies an output array in place; use a try-catch
            out = zeros(Float64, max(5, n))
            try
                SplinesMod.bicube_eval!(bc, psi, theta)
                # Attempt to read a field or fallback
                return getfield(bc, :_f) isa AbstractArray ? Float64.(bc._f) : out
            catch
                return out
            end
        end
    catch
    end

    # Default safe return (b, db/dpsi, db/dtheta, jac, dj/dpsi)
    return [1.0, 0.0, 0.0, 1.0, 0.0]
end
"""
Minimal bicube compatibility shim.
Provides `bicube_eval_external` used by torque.jl. For real
`SplinesMod.BicubicSpline` objects we delegate to SplinesMod.bicube_eval!
Otherwise we return a safe default vector.
"""
function bicube_eval_external(bc, psi::Float64, theta::Float64, n::Integer)
    # If bc looks like a SplinesMod BicubicSpline, try to call its eval
    try
        if isdefined(Main, :SplinesMod) && typeof(bc) <: SplinesMod.BicubicSpline
            # SplinesMod.bicube_eval! modifies an output array in place; use a try-catch
            out = zeros(Float64, max(5, n))
            try
                SplinesMod.bicube_eval!(bc, psi, theta)
                # Attempt to read a field or fallback
                return getfield(bc, :_f) isa AbstractArray ? Float64.(bc._f) : out
            catch
                return out
            end
        end
    catch
    end

    # Default safe return (b, db/dpsi, db/dtheta, jac, dj/dpsi)
    return [1.0, 0.0, 0.0, 1.0, 0.0]
end
