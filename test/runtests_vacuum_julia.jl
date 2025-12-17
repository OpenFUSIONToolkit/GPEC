@testset "Test Vacuum Julia" begin

    mth = 512
    grdgre = zeros(Float64, 2*nobs, 2*nsrc)
    gren = zeros(Float64, nobs, nsrc)

    try
        inputs = JPEC.VacuumMod.VacuumInput(
        mtheta=mth,
        n=1,
        )
        settings = JPEC.VacuumMod.WallGeometry()
        data_file = joinpath(@__DIR__, "test_data/vacuum_data.txt")
        lines = readlines(data_file)

        # Helper function to parse a line of comma-separated numbers
        function parse_number_line(line)
            tokens = split(line, ',')
            return [parse(Float64, strip(token)) for token in tokens if !isempty(strip(token))]
        end

        # Parse the five lines
        if length(lines) < 5
            error("Expected 5 lines in output_data.txt, but found only $(length(lines)) lines!")
        end

        xobs = parse_number_line(lines[1])
        zobs = parse_number_line(lines[2])
        xsce = parse_number_line(lines[3])
        zsce = parse_number_line(lines[4])
        params = parse_number_line(lines[5])

        xobs = xobs[1:mth]
        zobs = zobs[1:mth]
        xsce = xsce[1:mth]
        zsce = zsce[1:mth]

        # Extract kernel function parameters
        j1 = Int(params[1])
        j2 = Int(params[2])
        isgn = Int(params[3])
        iopw = Int(params[4])
        iops = Int(params[5])
        wall_flag = params[6] != 0 

        #---------------------------------------------------------------
        #  Set up Kernel Function Parameters
        #---------------------------------------------------------------
        try
            JPEC.VacuumMod.kernel!(
                grdgre, gren, 
                xobs[1:mth], zobs[1:mth], 
                xsce[1:mth], zsce[1:mth], 
                j1, j2, isgn, iopw, iops, wall_flag, inputs, settings
            )

            if any(isnan, grdgre) || any(isnan, gren)
                error("Input arrays contain NaN values. Calculation aborted.")
            end
            println("✓ Julia kernel! function executed successfully!")
        catch e
            println("✗ Error running Julia kernel! function.")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
            end
        end
        @info "mscvac OK!"
    catch e
        @test false
        @error "mscvac failed: $e"
    end ㅇ

end