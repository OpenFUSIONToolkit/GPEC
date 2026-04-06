"""
    Output

Output file management for PENTRC results.
Handles ASCII and NetCDF output formatting.
"""

using Printf

"""
    init_output(outp::PentrcControl, dir_path::String)

Initialize output directory and open file handles.

# Arguments
- `outp::PentrcControl`: Output configuration
- `dir_path::String`: Working directory path
"""
function init_output(ctrl::PentrcControl, dir_path::String)

    # Create output directory if it doesn't exist
    if !ispath(dir_path)
        mkpath(dir_path)
    end

    return dir_path
end


"""
    write_torque_ascii(outp::PentrcControl, nn::Int, method::String, 
                       psi::Vector{Float64}, torque::Vector{ComplexF64})

Write torque calculation results to ASCII file.

# Arguments
- `outp::PentrcControl`: Output configuration
- `nn::Int`: Toroidal mode number
- `method::String`: Calculation method name
- `psi::Vector{Float64}`: Poloidal flux values
- `torque::Vector{ComplexF64}`: Computed torque values
"""
function write_torque_ascii(ctrl::PentrcControl, intr::PentrcInternal, nn::Int, method::String,
                           psi::Vector{Float64}, torque::Vector{ComplexF64})

    if !ctrl.output_torque || !ctrl.output_ascii
        return
    end

    nstring = @sprintf("%4d", nn)
    fname = joinpath(intr.dir_path, "pentrc_torque_$(method)_n$(strip(nstring)).out")
    
    open(fname, "w") do f
        println(f, "PENTRC TORQUE CALCULATION")
        println(f, "Method: $method")
        println(f, "Toroidal mode number: $nn")
        println(f)
        @printf(f, "%16s %20s %20s\n", "psi", "Re(Torque)", "Im(Energy)")
        
        for i in eachindex(psi)
            @printf(f, "%16.8e %20.12e %20.12e\n", 
                    psi[i], real(torque[i]), imag(torque[i]))
        end
    end
end


"""
    write_orbit_ascii(outp::PentrcControl, nn::Int, psi::Float64, 
                      theta::Vector{Float64}, orbit::Vector{ComplexF64})

Write orbit/equilibrium data to ASCII file.

# Arguments
- `outp::PentrcControl`: Output configuration
- `nn::Int`: Toroidal mode number
- `psi::Float64`: Poloidal flux value
- `theta::Vector{Float64}`: Poloidal angle array
- `orbit::Vector{ComplexF64}`: Orbit data
"""
function write_orbit_ascii(ctrl::PentrcControl, intr::PentrcInternal, nn::Int, psi::Float64,
                          theta::Vector{Float64}, orbit::Vector{ComplexF64})

    if !ctrl.output_orbit || !ctrl.output_ascii
        return
    end

    nstring = @sprintf("%4d", nn)
    pstring = @sprintf("%.4f", psi)
    fname = joinpath(intr.dir_path, "pentrc_orbit_n$(strip(nstring))_psi$(strip(pstring)).out")
    
    open(fname, "w") do f
        println(f, "PENTRC ORBIT DATA")
        println(f, "Toroidal mode: $nn, psi = $psi")
        println(f)
        @printf(f, "%16s %20s %20s\n", "theta", "Re(Data)", "Im(Data)")
        
        for i in eachindex(theta)
            @printf(f, "%16.8e %20.12e %20.12e\n",
                    theta[i], real(orbit[i]), imag(orbit[i]))
        end
    end
end


"""
    write_energy_ascii(outp::PentrcControl, nn::Int, psi::Float64,
                       xlmda::Vector{Float64}, energy::Vector{Float64})

Write kinetic energy data to ASCII file.

# Arguments
- `outp::PentrcControl`: Output configuration
- `nn::Int`: Toroidal mode number
- `psi::Float64`: Poloidal flux value
- `xlmda::Vector{Float64}`: Pitch angle array
- `energy::Vector{Float64}`: Energy values
"""
function write_energy_ascii(ctrl::PentrcControl, intr::PentrcInternal, nn::Int, psi::Float64,
                           xlmda::Vector{Float64}, energy::Vector{Float64})

    if !ctrl.output_ascii
        return
    end

    nstring = @sprintf("%4d", nn)
    pstring = @sprintf("%.4f", psi)
    fname = joinpath(intr.dir_path, "pentrc_energy_n$(strip(nstring))_psi$(strip(pstring)).out")
    
    open(fname, "w") do f
        println(f, "PENTRC KINETIC ENERGY")
        println(f, "Toroidal mode: $nn, psi = $psi")
        println(f)
        @printf(f, "%16s %20s\n", "pitch angle", "energy")
        
        for i in eachindex(xlmda)
            @printf(f, "%16.8e %20.12e\n", xlmda[i], energy[i])
        end
    end
end


"""
    print_summary(intr::PentrcInternal, ctrl::PentrcControl)

Print summary information about computation to stdout.

# Arguments
- `intr::PentrcInternal`: Computation results
- `ctrl::PentrcControl`: Control parameters
"""
function print_summary(intr::PentrcInternal, ctrl::PentrcControl)
    
    if !ctrl.verbose
        return
    end
    
    println()
    println("=" * 60)
    println("PENTRC SUMMARY")
    println("=" * 60)
    
    println("Method: $(intr.method)")
    println("Toroidal mode: $(ctrl.nn)")
    println("Bounce harmonic range: $(ctrl.nl)")
    println()
    
    println("Results:")
    if ctrl.dynamic_grid
        @printf("  Total torque (dynamic): %12.4e\n", real(intr.tphi))
        @printf("  Total energy (dynamic): %12.4e\n", imag(intr.tphi)/(2*ctrl.nn))
    end
    
    if ctrl.equil_grid || ctrl.input_grid
        @printf("  Total torque (grid): %12.4e\n", real(intr.teq))
        @printf("  Total energy (grid): %12.4e\n", imag(intr.teq)/(2*ctrl.nn))
    end
    
    println()
    println("=" * 60)
    println()
end
