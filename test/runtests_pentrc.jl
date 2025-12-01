using Test

# Include the pentrc interface under test
include(joinpath(@__DIR__, "..", "src", "pentrc", "pentrc_interface.jl"))

@testset "pentrc_interface basic behavior" begin
    # 1) read_pentrc_input should warn when no pentrc.in file exists
    # Temporarily move any existing file out of the way so the test is hermetic.
    pentrc_in = joinpath(pwd(), "pentrc.in")
    bak = pentrc_in * ".bak_for_test"
    moved = false
    if isfile(pentrc_in)
        mv(pentrc_in, bak; force=true)
        moved = true
    end
    try
        @test_warn "pentrc.in not found" read_pentrc_input()
    finally
        if moved
            mv(bak, pentrc_in; force=true)
        end
    end

    # 2) initialize_pentrc with all ops disabled should not call read_kin/read_equil
    # and should populate default psi_out when it is all -1
    global psi_out = fill(-1.0, npsi_out)
    initialize_pentrc(; op_kin=false, op_deq=false, op_peq=false)
    @test length(psi_out) == 30
    @test psi_out == collect(1:30) ./ 30.6

    # 3) openmp_threads deprecated warning is emitted when > 0
    global openmp_threads = 4
    @test_warn "openmp_threads is deprecated" initialize_pentrc(; op_kin=false, op_deq=false, op_peq=false)
    # restore
    global openmp_threads = 0
end

println("runtests_pentrc.jl: tests defined")
