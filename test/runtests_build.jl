@testset "Make fortran files" begin
    try
        # Run the build function
        result = JPEC.build_fortran()
        @test true
        @info "Fortran builds completed successfully"
    catch e
        @test false
        @error "build_spline_fortran() failed: $e"
    end
end
