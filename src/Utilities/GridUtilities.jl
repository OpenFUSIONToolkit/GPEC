"""
    GridUtilities

Reusable grid generation functions for non-uniform spacing.
"""

"""
    powspace(x_min, x_max, npower, nx, spacing)

Power-law spaced grid with analytic derivative calculation.
Creates a grid with more points concentrated near boundaries.

# Arguments
- `x_min, x_max`: Grid boundaries
- `npower::Int`: Power law exponent (1=linear, higher=more concentration at edges)
- `nx::Int`: Number of grid points
- `spacing::String`: "lower" (concentrate at x_min), "upper" (at x_max), or "both"

# Returns
- `Matrix{Float64}`: (2, nx) where row 1 = positions, row 2 = dr/d(norm) derivatives
"""
function powspace(x_min::Float64, x_max::Float64, npower::Int, nx::Int,
                  spacing::String)::Matrix{Float64}

    result = zeros(Float64, 2, nx)

    y = if spacing == "lower"
        collect(range(-1.0, 0.0, length=nx))
    elseif spacing == "upper"
        collect(range(0.0, 1.0, length=nx))
    elseif spacing == "both"
        collect(range(-1.0, 1.0, length=nx))
    else
        error("Unknown spacing type: $spacing. Use \"lower\", \"upper\", or \"both\".")
    end

    F = zeros(Float64, nx)
    deriv = zeros(Float64, nx)

    for i in eachindex(y)
        deriv[i] = abs((y[i] - 1) * (y[i] + 1))^npower
        yi = y[i]

        # Analytic antiderivatives for powers 1-9
        F[i] = if npower == 1
            -yi + yi^3 / 3
        elseif npower == 2
            yi - 2 * yi^3 / 3 + yi^5 / 5
        elseif npower == 3
            -yi + yi^3 - 3 * yi^5 / 5 + yi^7 / 7
        elseif npower == 4
            yi - 4 * yi^3 / 3 + 6 * yi^5 / 5 - 4 * yi^7 / 7 + yi^9 / 9
        elseif npower == 5
            -yi + 5 * yi^3 / 3 - 2 * yi^5 + 10 * yi^7 / 7 - 5 * yi^9 / 9 + yi^11 / 11
        elseif npower == 6
            yi - 2 * yi^3 + 3 * yi^5 - 20 * yi^7 / 7 + 5 * yi^9 / 3 - 6 * yi^11 / 11 + yi^13 / 13
        elseif npower == 7
            -yi + 7 * yi^3 / 3 - 21 * yi^5 / 5 + 5 * yi^7 - 35 * yi^9 / 9 + 21 * yi^11 / 11 - 7 * yi^13 / 13 + yi^15 / 15
        elseif npower == 8
            yi - 8 * yi^3 / 3 + 28 * yi^5 / 5 - 8 * yi^7 + 70 * yi^9 / 9 - 56 * yi^11 / 11 + 28 * yi^13 / 13 - 8 * yi^15 / 15 + yi^17 / 17
        elseif npower == 9
            -yi + 3 * yi^3 - 36 * yi^5 / 5 + 12 * yi^7 - 14 * yi^9 + 126 * yi^11 / 11 - 84 * yi^13 / 13 + 12 * yi^15 / 5 - 9 * yi^17 / 17 + yi^19 / 19
        else
            @warn "Power $npower not in analytic database, using numeric integration" maxlog=1
            _powspace_numeric_integral(yi, npower)
        end
    end

    delta_y = F[nx] - F[1]
    delta_x = x_max - x_min

    result[2, :] = deriv .* (y[nx] - y[1])
    result[1, :] = (F .- F[1]) .* (delta_x / delta_y) .+ x_min

    return result
end

function _powspace_numeric_integral(y::Float64, npower::Int)::Float64
    y == 0.0 && return 0.0
    npts = 100
    y_int = range(0.0, y, length=npts)
    f_int = abs.((y_int .- 1) .* (y_int .+ 1)) .^ npower
    integral = (f_int[1] + f_int[npts]) / 2
    for i in 2:(npts - 1)
        integral += f_int[i]
    end
    return integral * (y / (npts - 1))
end
