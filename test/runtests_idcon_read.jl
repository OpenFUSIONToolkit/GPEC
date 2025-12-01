using Test

# Load the DCON interface module under test (path resolved relative to this test file)
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
using .DconInterface

# Ensure data directory exists
mkpath("test/data")
path = "test/data/idcon_test.bin"

# Write a small Fortran-unformatted-like binary with two records
open(path, "w") do io
    # Record 1: two Float64 values (mpert as 4.0, chi1 as a large value so
    # this record is NOT mistaken for a theta array which must lie in 0..2π)
    payload = IOBuffer()
    write(payload, Float64(4.0))
    write(payload, Float64(1e9))
    data = take!(payload)
    n = Int32(length(data))
    write(io, n)
    write(io, data)
    write(io, n)

    # Record 2: three Float64 theta samples (0, π/2, π)
    payload2 = IOBuffer()
    write(payload2, Float64(0.0))
    write(payload2, Float64(pi/2))
    write(payload2, Float64(pi))
    data2 = take!(payload2)
    n2 = Int32(length(data2))
    write(io, n2)
    write(io, data2)
    write(io, n2)
end

# Point the module to the test file and run the reader
DconInterface.idconfile = path
DconInterface.idcon_read(0)

@test DconInterface.mpert[] == 4
@test length(DconInterface.theta) >= 3

println("idcon_read heuristic test passed")
