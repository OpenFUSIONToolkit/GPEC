# PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE

"""
utilities module

DESCRIPTION:
    Collection of utility procedures for general Julia needs.

PUBLIC MEMBER FUNCTIONS:
    All functions are public

REVISION HISTORY:
    2014.03.06 -Logan- initial writing.
    2025.11.25 -Fairborn- Converted to Julia
    Converted using Claude AI for first pass

AUTHOR: Logan
EMAIL: nikolas.logan@columbia.edu
"""

# Wrap utilities in a module so it can be `using`-ed or included cleanly.
module Utilities

include(joinpath(@__DIR__, "params.jl"))  # For r4, r8, twopi
# NCDatasets is optional; make its import non-fatal so Utilities can be
# included even when NetCDF support is not installed in the environment.
try
    @eval using NCDatasets
catch err
    @warn "NCDatasets not available; NetCDF functionality disabled." exception=(err, catch_backtrace())
end # module Utilities

using Printf
using Statistics

"""
    get_free_file_unit(lu_max=-1)

Find a unit number that is not in use.
Note: In Julia, this is less relevant as file I/O uses file handles directly,
but included for compatibility with ported code.

# Arguments
- `lu_max::Int`: Maximum unit (default -1, which sets to 97)

# Returns
- `Int`: Free unit number
"""
function get_free_file_unit(lu_max=-1)
    m = lu_max < 1 ? 97 : lu_max
    
    # In Julia, we don't really need unit numbers
    # Return a random number that could be used as an identifier
    return rand(1:m)
end
end # module Utilities

"""
    to_upper(strIn)

Capitalize a string.

# Arguments
- `strIn::String`: Input string

# Returns
- `String`: Uppercase string
"""
function to_upper(strIn::String)
    return uppercase(strIn)
end

"""
    btoi(bool)

Convert logical variable to integer.

# Arguments
- `bool::Bool`: True or False

# Returns
- `Int`: 0 or 1
"""
function btoi(bool::Bool)
    return bool ? 1 : 0
end

"""
    itob(ibool)

Convert integer variable to logical.

# Arguments
- `ibool::Int`: Any integer variable

# Returns
- `Bool`: True if ibool > 0
"""
function itob(ibool::Int)
    return ibool > 0
end

"""
    progressbar(j, jstart, jstop; op_step=1, op_percent=10)

Print progress bar for do loop every specified percentage of iterations.

# Arguments
- `j::Int`: Current iteration variable
- `jstart::Int`: Start of loop
- `jstop::Int`: End of loop
- `op_step::Int`: Loop increment (default 1)
- `op_percent::Int`: Prints progress every percent of the iterations (default 10)
"""
function progressbar(j::Int, jstart::Int, jstop::Int; op_step::Int=1, op_percent::Int=10)
    iteration = div(j - jstart, op_step) + 1
    iterations = div(jstop - jstart, op_step) + 1
    bar = "  |----------| ???% iterations complete"
    
    crnt = floor(Int, iteration * 100.0 / iterations / op_percent)
    last = floor(Int, (iteration - 1) * 100.0 / iterations / op_percent)
    
    if iteration == 1 || crnt > last || iteration == iterations
        done = iteration == iterations ? 100 : crnt * op_percent
        bar_str = collect(bar)
    bar_str[16:18] .= collect(lpad(string(done), 3))
        
        d10 = div(done, 10)
        for k in 1:d10
            bar_str[3+k] = 'o'
        end
        
        println(String(bar_str))
    end
end

"""
    timer(; mode=0, unit=nothing)

Handles timing statistics.

# Arguments
- `mode::Int`: mode = 0 starts timer, mode < 0 writes time and restarts, mode > 0 writes time
- `unit::Union{IO,Nothing}`: Output stream (default stdout)
"""
const _timer_seconds = Ref(0.0)
function timer(; mode::Int=0, unit::Union{IO,Nothing}=nothing)
    time = Base.time()

    # Write split
    if mode != 0
        elapsed = time - _timer_seconds[]
        secs = Int(floor(elapsed))
        hrs = div(secs, 3600)
        mins = div(secs - hrs * 3600, 60)
        secs = (secs - hrs * 3600) - mins * 60

        if unit !== nothing
                println(unit, " Total cpu time = ", elapsed, " seconds")
        else
            if hrs > 0
                println("Total cpu time = $hrs hours, $mins minutes, $secs seconds")
            elseif mins > 0
                println("Total cpu time = $mins minutes, $secs seconds")
            else
                println("Total cpu time = $secs seconds")
            end
        end
    end

    # Start timer
    if mode <= 0
        _timer_seconds[] = time
    end
    return nothing
end

"""
    splitstring(str, sep; maxsplit=nothing, debug=false)

Split string into substrings using sep as delimiter.

# Arguments
- `str::String`: Original string
- `sep::String`: Delimiter separating words in original string
- `maxsplit::Union{Int,Nothing}`: Maximum number of splits (default: no limit)
- `debug::Bool`: Debug flag

# Returns
- `Vector{String}`: Array of substrings
"""
function splitstring(str::String, sep::String; maxsplit::Union{Int,Nothing}=nothing, debug::Bool=false)
    # Julia's built-in split is more powerful, but we replicate the Fortran behavior
    parts = split(str, sep; keepempty=false)
    
    if maxsplit !== nothing
        parts = parts[1:min(length(parts), maxsplit)]
    end
    
    if debug
        println("number of words is ", length(parts))
        for (i, word) in enumerate(parts)
            println(i, " ", word)
        end
    end
    
    return String.(parts)
end

"""
    replacestring!(str, subout, subin; debug=false)

Replace substring with an alternate substring of equal length.

# Arguments
- `str::String`: Original string (will be returned modified)
- `subout::String`: Substring to be replaced
- `subin::String`: Substring replacing subout
- `debug::Bool`: Debug flag

# Returns
- `String`: Modified string
"""
function replacestring(str::String, subout::String, subin::String; debug::Bool=false)
    if debug
        println("  Replacing $subout with $subin")
    end
    
    return replace(str, subout => subin)
end

"""
    readtable(file; verbose=false, debug=false)

Read ascii file containing a single table.
- Assumes data starts on the first line starting with a number
- Assumes column titles are on the line directly above the data
- Ignores all other lines

# Arguments
- `file::String`: File path
- `verbose::Bool`: Print status messages
- `debug::Bool`: Print debug messages

# Returns
- `table::Matrix{Float64}`: The data
- `titles::Vector{String}`: The column headers
"""
function readtable(file::String; verbose::Bool=false, debug::Bool=false)
    if verbose
        println("Reading table from file: ")
        println("  $file")
    end
    
    lines = readlines(file)
    numbers = "1234567890"
    signs = "+-"
    
    startline = 0
    endline = 0
    
    # Find data lines
    for (l, line) in enumerate(lines)
        line_stripped = strip(line)
        if isempty(line_stripped)
            continue
        end
        
        j = 1
        # Skip leading signs
        while j <= length(line_stripped) && line_stripped[j] in signs
            j += 1
        end
        
        # Check if line starts with number
        isnum = j <= length(line_stripped) && line_stripped[j] in numbers
        
        if isnum && startline == 0
            startline = l
        elseif !isnum && startline != 0 && endline == 0
            endline = l - 1
            break
        end
    end
    
    if endline == 0
        endline = length(lines)
    end
    
    if startline == 0
        error("ERROR: readtable - No data found")
    end
    
    if debug
        println("-> Table start,end lines = $startline, $endline")
    end
    
    # Parse data
    data_lines = lines[startline:endline]
    
    # Get titles from line before data
    titles = String[]
    if startline > 1
        title_line = replace(lines[startline-1], '\t' => ' ')
        titles = splitstring(title_line, " "; debug=debug)
    end
    
    # Parse table
    table = []
    for line in data_lines
        values = parse.(Float64, split(line))
        push!(table, values)
    end
    
    # Convert to matrix
    ncol = maximum(length.(table))
    nrow = length(table)
    table_matrix = zeros(nrow, ncol)
    
    for (i, row) in enumerate(table)
        table_matrix[i, 1:length(row)] .= row
    end
    
    if debug
        println("-> Found $nrow by $ncol table")
        println("Done")
    end
    
    return table_matrix, titles
end

"""
    nunique(array; op_sorted=false)

Find number of unique elements in an array.

# Arguments
- `array::Vector{Float64}`: Original array
- `op_sorted::Bool`: Common values or unique values are assumed to be grouped

# Returns
- `Int`: Number of unique elements
"""
function nunique(array::AbstractVector{<:Real}; op_sorted::Bool=false)
    if op_sorted
        # Optimized for sorted arrays
        count = 1
        for i in 2:length(array)
            if array[i] != array[i-1]
                count += 1
            end
        end
        return count
    else
        # General approach
        return length(unique(array))
    end
end

"""
    median(array)

Compute the median of a 1D array.

# Arguments
- `array::Vector{Float64}`: Input array

# Returns
- `Float64`: Median value
"""
function median_val(array::Vector{Float64})
    return median(array)  # Use Statistics.median
end

"""
    sorted(array)

Return a sorted copy of the array.

# Arguments
- `array::Vector{Float64}`: Input array

# Returns
- `Vector{Float64}`: Sorted array
"""
function sorted(array::Vector{Float64})
    return sort(array)
end

"""
    pnorm(array, p)

Return the p-norm of a 1D array.

# Arguments
- `array::Vector{Float64}`: Input array
- `p::Float64`: Must be greater than 1 (1=taxicab, 2=euclidean, Inf=maximum)

# Returns
- `Float64`: p-norm value
"""
function pnorm(array::Vector{Float64}, p::Float64)
    if p < 1
        error("ERROR: p-norm with p<1 is not a norm")
    end
    
    if isinf(p)
        return maximum(abs.(array))
    end
    
    return sum(abs.(array).^p)^(1/p)
end

"""
    cpnorm(array, p)

Return the p-norm of a 1D complex array.

# Arguments
- `array::Vector{ComplexF64}`: Input array
- `p::Float64`: Must be greater than 1

# Returns
- `Float64`: p-norm value
"""
function cpnorm(array::Vector{ComplexF64}, p::Float64)
    if p < 1
        error("ERROR: p-norm with p<1 is not a norm")
    end
    
    if isinf(p)
        return maximum(abs.(array))
    end
    
    return sum(abs.(array).^p)^(1/p)
end

"""
    ri(c)

Return the real and imaginary components of a complex value as a tuple.

# Arguments
- `c::ComplexF64`: Complex value

# Returns
- `Vector{Float64}`: [real, imag]
"""
function ri(c::ComplexF64)
    return [real(c), imag(c)]
end

"""
    append_1d!(list, element)

Append a 1D array (modifies in place via push!).

# Arguments
- `list::Vector{Float64}`: Array to be appended
- `element::Float64`: Number appended to end of list
"""
function append_1d!(list::Vector{Float64}, element::Float64)
    push!(list, element)
end

# Alternative that matches Fortran behavior more closely
function append_1d(list::Vector{Float64}, element::Float64)
    return vcat(list, element)
end

"""
    append_2d!(list, elements)

Append a 2D array along second dimension.

# Arguments
- `list::Matrix{Float64}`: 2D array to be appended
- `elements::Vector{Float64}`: 1D array appended to 2nd dimension of list

# Returns
- `Matrix{Float64}`: Appended matrix
"""
function append_2d(list::Matrix{Float64}, elements::Vector{Float64})
    if size(list, 1) != length(elements)
        error("Cannot append arrays without matched dimensions")
    end
    
    return hcat(list, elements)
end

"""
    iscdftf(m, func)

Compute 1D truncated-discrete Fourier forward transform.

# Arguments
- `m::Vector{Int}`: Fourier modes
- `func::Vector{ComplexF64}`: Function on a regular grid 0-1

# Returns
- `Vector{ComplexF64}`: Fourier components
"""
function iscdftf(m::Vector{Int}, func::Vector{ComplexF64})
    ms = length(m)
    fs = length(func)
    funcm = zeros(ComplexF64, ms)
    
    for i in 1:ms
        for j in 0:fs-1
            funcm[i] += (1.0/fs) * func[j+1] * exp(-2π*im*m[i]*j/fs)
        end
    end
    
    return funcm
end

"""
    iscdftb(m, funcm, fs)

Compute 1D truncated-discrete Fourier backward transform.

# Arguments
- `m::Vector{Int}`: Fourier modes
- `funcm::Vector{ComplexF64}`: Fourier components
- `fs::Int`: Number of function points

# Returns
- `Vector{ComplexF64}`: Function on regular grid
"""
function iscdftb(m::Vector{Int}, funcm::Vector{ComplexF64}, fs::Int)
    ms = length(m)
    func = zeros(ComplexF64, fs+1)
    
    for i in 0:fs-1
        for j in 1:ms
            func[i+1] += funcm[j] * exp(2π*im*m[j]*i/fs)
        end
    end
    func[fs+1] = func[1]
    
    return func
end

"""
    check(stat, msg="")

Check if a netcdf call was successful. If not, raise an error.

# Arguments
- `stat::Int`: Status code (0 = success in NetCDF)
- `msg::String`: Optional error message
"""
function check(stat::Int, msg::String="")
    if stat != 0
        if !isempty(msg)
            error("ERROR: $msg - NetCDF error code: $stat")
        else
            error("ERROR: failed to write/read netcdf file - error code: $stat")
        end
    end
end