using DelimitedFiles
using Printf
using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using JPEC
mth = 512  # Number of poloidal grid points
inputs = JPEC.VacuumMod.VacuumInput(
    mtheta=mth,
    n=1,
)
settings = JPEC.VacuumMod.WallGeometry()

#---------------------------------------------------------------
#  Read the output_data.txt file
#---------------------------------------------------------------
data_file = joinpath(@__DIR__, "output_data.txt")

println("Reading file: $data_file")
println("File exists: $(isfile(data_file))")
println()

# Read all lines
lines = readlines(data_file)
println("Total lines in file: $(length(lines))")

# Helper function to parse a line of comma-separated numbers
function parse_number_line(line)
    tokens = split(line, ',')
    return [parse(Float64, strip(token)) for token in tokens if !isempty(strip(token))]
end

# Parse the five lines
if length(lines) < 5
    error("Expected 5 lines in output_data.txt, but found only $(length(lines)) lines!")
end

println("\nParsing data from first 5 lines:")
xobs = parse_number_line(lines[1])
println("  Line 1 (xobs): $(length(xobs)) values")

zobs = parse_number_line(lines[2])
println("  Line 2 (zobs): $(length(zobs)) values")

xsce = parse_number_line(lines[3])
println("  Line 3 (xsce): $(length(xsce)) values")

zsce = parse_number_line(lines[4])
println("  Line 4 (zsce): $(length(zsce)) values")

params = parse_number_line(lines[5])
println("  Line 5 (params): $(length(params)) values")

xobs = xobs[1:mth]
zobs = zobs[1:mth]
xsce = xsce[1:mth]
zsce = zsce[1:mth]

println("\n✓ Successfully parsed all data!")
println("\nData summary:")
println("  xobs: $(length(xobs)) points, range [$(minimum(xobs)), $(maximum(xobs))]")
println("  zobs: $(length(zobs)) points, range [$(minimum(zobs)), $(maximum(zobs))]")
println("  xsce: $(length(xsce)) points, range [$(minimum(xsce)), $(maximum(xsce))]")
println("  zsce: $(length(zsce)) points, range [$(minimum(zsce)), $(maximum(zsce))]")
println("  params: $(params)")

# Extract kernel function parameters
j1_input = Int(params[1])
j2_input = Int(params[2])
isgn_input = Int(params[3])
iopw_input = Int(params[4])
iops_input = Int(params[5])
wall_flag_input = params[6] != 0  # Convert to boolean

println("\nKernel parameters:")
println("  j1 = $j1_input")
println("  j2 = $j2_input")
println("  isgn = $isgn_input")
println("  iopw = $iopw_input")
println("  iops = $iops_input")
println("  wall_flag = $wall_flag_input")

#---------------------------------------------------------------
#  Set up Kernel Function Parameters
#---------------------------------------------------------------

# Set up parameters for kernel function test
nobs = length(xobs)
nsrc = length(xsce)

# Validate data consistency
if length(xobs) != length(zobs)
    error("xobs and zobs have different lengths: $(length(xobs)) vs $(length(zobs))")
end
if length(xsce) != length(zsce)
    error("xsce and zsce have different lengths: $(length(xsce)) vs $(length(zsce))")
end

println("Data validation passed!")
println()

# Initialize output matrices
grdgre = zeros(Float64, 2*nobs, 2*nsrc)
gren = zeros(Float64, nobs, nsrc)

# Use parameters loaded from input file
j1 = j1_input
j2 = j2_input
isgn = isgn_input
iopw = iopw_input
iops = iops_input
wall_flag = wall_flag_input

println("Test configuration:")
println("  Observer points: $nobs")
println("  Source points: $nsrc")
println("  Observer type (j1): $j1 (1=plasma, 2=wall)")
println("  Source type (j2): $j2 (1=plasma, 2=wall)")
println("  Sign parameter (isgn): $isgn")
println("  Wall option (iopw): $iopw")
println("  Log singularity option (iops): $iops")
println("  Wall flag: $wall_flag")
println()
println("Output matrix dimensions:")
println("  grdgre: $(size(grdgre))")
println("  gren: $(size(gren))")

#---------------------------------------------------------------
#  Call the kernel function
#---------------------------------------------------------------

# Run the kernel function
# Note: This may fail if additional setup/initialization is required
try
    JPEC.VacuumMod.kernel!(
        grdgre, gren, 
        xobs[1:mth], zobs[1:mth], 
        xsce[1:mth], zsce[1:mth], 
        j1, j2, isgn, iopw, iops, wall_flag, inputs, settings
    )
    
    println("✓ Kernel function executed successfully!")
    # println()
    # println("Result statistics:")
    # println("  grdgre: min=$(minimum(grdgre)), max=$(maximum(grdgre)), mean=$(sum(grdgre)/length(grdgre))")
    # println("  gren: min=$(minimum(gren)), max=$(maximum(gren)), mean=$(sum(gren)/length(gren))")
    
catch e
    println("✗ Error running kernel function.")
    # println(e)
    println()
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    println()
end

#---------------------------------------------------------------
#  Print a small portion of the output matrices for inspection
#---------------------------------------------------------------

# Debug: Check what's in the result matrices
println("===== RESULT DIAGNOSTICS =====")
# println("grdgre exists: $(isdefined(Main, :grdgre))")
# println("gren exists: $(isdefined(Main, :gren))")
# println()

if isdefined(Main, :grdgre) && isdefined(Main, :gren)

    writedlm("gren_output.txt", gren)
    writedlm("grdgre_output.txt", grdgre)

    println("grdgre size: $(size(grdgre))")
    println("gren size: $(size(gren))")
    println()

    println("gren first 5x5:")
    # display(gren[1:min(5,size(gren,1)), 1:min(5,size(gren,2))])
    for i in 1:min(5,size(gren,1))
        for j in 1:min(5,size(gren,2))
            @printf("%12.9f  ", gren[i,j])
        end
        println()
    end
    println()
    println("grdgre first 5x5:")
    for i in 1:min(5,size(grdgre,1))
        for j in 1:min(5,size(grdgre,2))
            @printf("%12.9f  ", grdgre[i,j])
        end
        println()
    end
    
    # println("grdgre first 5x5:")
    # display(grdgre[1:min(5,size(grdgre,1)), 1:min(5,size(grdgre,2))])
    println()
    
else
    println("ERROR: Result matrices not defined!")
    println("Make sure to run the kernel execution cell first.")
end