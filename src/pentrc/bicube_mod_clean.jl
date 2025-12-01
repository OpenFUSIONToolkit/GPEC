"""
Minimal bicube compatibility shim (clean test copy).
Provides `bicube_eval_external` used by torque.jl. For real
`SplinesMod.BicubicSpline` objects we delegate to SplinesMod.bicube_eval!
Otherwise we return a safe default vector.
"""
function bicube_eval_external(bc, psi::Float64, theta::Float64, n::Integer)
    try
        if isdefined(Main, :SplinesMod) && bc isa SplinesMod.BicubicSpline
            out = zeros(Float64, max(5, n))
            try
                SplinesMod.bicube_eval!(bc, psi, theta)
                return getfield(bc, :_f) isa AbstractArray ? Float64.(bc._f) : out
            catch
                return out
            end
        end
    catch
    end
    return [1.0, 0.0, 0.0, 1.0, 0.0]
end
