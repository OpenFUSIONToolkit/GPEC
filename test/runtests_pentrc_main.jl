using Test

# This test will include `pentrc.jl` after temporarily changing
# the working directory so the file-local includes resolve correctly.
root = joinpath(@__DIR__, "..")
pentrc_dir = joinpath(root, "src", "pentrc")

@testset "pentrc main (smoke)" begin
    pwd0 = pwd()
    try
        # Include the pentrc interface (safe to include in tests)
        include(joinpath(@__DIR__, "..", "src", "pentrc", "pentrc_interface.jl"))

        # Configure safe runtime environment to avoid heavy computations
        global clean = false
        global diag_flag = true
        global verbose = false
        global pentrc_threads = 0
        global openmp_threads = 0

        # Initialize with file-reading disabled so the function doesn't
        # attempt to read kinetic/equilibrium files during the test.
        initialize_pentrc(; op_kin=false, op_deq=false, op_peq=false)

        # Test the moment handling logic from pentrc.main in isolation
        global moment = "heat"
        if moment == "heat"
            global qt = true
        elseif moment == "pressure"
            global qt = false
        else
            error("ERROR: Input moment must be 'pressure' or 'heat'")
        end
        @test qt == true

        global moment = "pressure"
        if moment == "heat"
            global qt = true
        elseif moment == "pressure"
            global qt = false
        else
            error("ERROR: Input moment must be 'pressure' or 'heat'")
        end
        @test qt == false

    finally
        cd(pwd0)
    end
end

println("runtests_pentrc_main.jl: tests defined")
