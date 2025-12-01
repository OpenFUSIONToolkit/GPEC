module Inputs

using LinearAlgebra
using DelimitedFiles
#Initial Conversion done with Github Copilot

# Local project modules - attempt to import, otherwise include local files.
function _ensure_local_module(modsym::Symbol, paths::Vector{String})
    # Prefer local files first (these are often plain scripts in this repo).
    for p in paths
        fp = joinpath(@__DIR__, p)
        if isfile(fp)
                # Some helper files (Utilities, Splines wrapper) are intended to
                # be top-level modules. Include them in `Main` so other local
                # includes that expect `Main.Utilities` or `Main.SplinesMod` work.
                if modsym in (:Utilities, :Splines)
                    Core.eval(Main, :(include($(fp))))
                else
                    include(fp)
                end
                return
            end
    end

    # Fall back to attempting to `using` the module as an installed package.
    try
        @eval using $(modsym)
        return
    catch err
        error("Module $(modsym) not found as a local file or installed package. Provide one of: ", join(paths, ", "))
    end
end

# Utilities: expected at src/pentrc/utilities.jl
_ensure_local_module(:Utilities, ["utilities.jl"])
# If `utilities.jl` is a script (not a `module Utilities`), including it will
# inject the functions into this module (`Inputs`). To preserve the original
# code's use of the `Utilities` namespace, create a local alias pointing to
# this module when a real `Utilities` module isn't defined.
if !isdefined(@__MODULE__, :Utilities)
    const Utilities = @__MODULE__
end
# Splines modules: expected at src/Splines
_ensure_local_module(:Splines, [joinpath("..","Splines","Splines.jl")])
# The Splines wrapper (`Splines.jl`) defines module `SplinesMod` and includes
# Helper/Cubic/Bicubic files. Include that once and create short aliases so
# the rest of this file can keep using `Splines` / `CubicSpline` / `BicubicSpline`
_ensure_local_module(:Splines, [joinpath("..","Splines","Splines.jl")])

# `include` of Splines.jl defines `SplinesMod` in this module's scope. Create
# convenient aliases so existing code (which expects `Splines` and
# `BicubicSpline` modules) continues to work.
const Splines = isdefined(@__MODULE__, :SplinesMod) ? SplinesMod : (isdefined(Main, :SplinesMod) ? Main.SplinesMod : error("Splines module not found"))
const BicubicSpline = Splines.BicubicSpline
const CubicSpline = Splines.CubicSpline
_ensure_local_module(:DconInterface, ["dcon_interface.jl"])

export newm, read_equil, read_kin

# Public placeholders for data structures used across the module
const kin = Ref{Any}(nothing)    # will hold a spline object
const dbob_m = Ref{Any}(nothing)
const divx_m = Ref{Any}(nothing)
const xs_m = Ref{Any}[Ref{Any}(nothing) for _ in 1:3]
const fnml = Ref{Any}(nothing)

# ======================================================================
# newm: convert a spectrum fm_i defined on modes m_i into a new mode set m_o
# (Direct port of the Fortran `newm` function)
# ======================================================================
function newm(m_i::Vector{Int}, fm_i::AbstractVector{<:Complex}, m_o::Vector{Int})::Vector{ComplexF64}
    # Accept complex inputs with any element type (Int/Float) and promote
    # to ComplexF64 for consistent numeric behavior.
    fm = ComplexF64.(fm_i)
    len_i = length(m_i)
    len_o = length(m_o)
    if length(fm) != len_i
        error("newm: length mismatch between m_i and fm_i")
    end

    # Compute indices according to Fortran logic
    i = max(abs(m_i[1]) - abs(m_o[1]) + 1, 1)
    ii = len_i - max(abs(m_i[end]) - abs(m_o[end]), 0)
    j = max(abs(m_o[1]) - abs(m_i[1]) + 1, 1)
    jj = len_o - max(abs(m_o[end]) - abs(m_i[end]), 0)

    # Check alignment
    if m_o[j] != m_i[i] || m_o[jj] != m_i[ii]
        throw(ErrorException("ERROR: Misalignment in m-spaces"))
    end

    newm = zeros(ComplexF64, len_o)
    newm[j:jj] .= fm[i:ii]
    return newm
end

# ======================================================================
# read_equil: port of Fortran read_equil(file,hlog)
# Calls into dcon_interface (stubbed) to load binary equilibrium data.
# ======================================================================
function read_equil(file::AbstractString; hlog::Union{Nothing,String}=nothing)
    # set idconfile in the dcon interface
    DconInterface.idconfile = file

    # prepare ideal solutions. (psixy=0)
    DconInterface.idcon_read(0)
    if hlog !== nothing
        DconInterface.idcon_harvest(hlog)
    end
    DconInterface.idcon_transform()
    # reconstruct metric tensors.
    DconInterface.idcon_metric()
    # form action matrices
    DconInterface.idcon_action_matrices()

    # set additional geometry spline
    DconInterface.set_geom()
    println("chi1 = ", DconInterface.chi1)
    return nothing
end

# ======================================================================
# read_kin: read ascii kinetic profiles and form kin spline
# Simplified port of the Fortran read_kin subroutine. This expects your
# Utilities.readtable and Splines.* functions to have equivalent Julia
# behavior. Where interfaces differ, adapt the calls accordingly.
# ======================================================================
function read_kin(file::AbstractString, zi::Integer, zimp::Integer, mi::Integer, mimp::Integer,
                  nfac::Float64, tfac::Float64, wefac::Float64, wpfac::Float64; write_log::Bool=false)

    # Use Utilities.readtable to obtain table and titles
    # Expected: table is a NxM Array{Float64,2} (or similar) and titles is a Vector{String}
    table, titles = Utilities.readtable(file, verbose=false, write_log=write_log)

    nrows, ncols = size(table)

    # temporary spline on experimental grid: build xs and fs arrays then
    # construct a CubicSpline using the Splines API.
    NKIN = 100
    xs_tmp = vec(table[1:end-1, 1])
    fs_tmp = Array{Float64}(undef, length(xs_tmp), 5)
    fs_tmp[:, :] .= table[1:end-1, 2:6]

    # create and fit cubic spline on the experimental grid
    tmp = Splines.CubicSpline(xs_tmp, fs_tmp; bctype="extrap")

    # Evaluate tmp on regular grid and build arrays for kin spline,
    # then construct the kin CubicSpline from the sampled data.
    xs_kin = collect(0.0:1.0/NKIN:1.0)
    fs_kin = zeros(Float64, length(xs_kin), 9)
    for (ii, psi) in enumerate(xs_kin)
        ftmp = Splines.spline_eval!(tmp, psi)
        fs_kin[ii, 1:5] = ftmp[1:5]
    end

    # convert temperatures to SI (eV->J)
    fs_kin[:, 3:4] .*= 1.602e-19

    kin_val = Splines.CubicSpline(xs_kin, fs_kin; bctype="extrap")
    # set a title field if present (some code expects kin_val.title)
    try
        setfield!(kin_val, :title, ["psi_n","n_i","n_e","t_i","t_e","omegae","loglam","nu_i","nu_e","zeff"])
    catch
        # ignore if the object does not have a title field
    end

    # collisionalities and other derived fields (direct translation)
    zeff = similar(kin_val.fs[:,1])
    zpitch = similar(zeff)
    welec = similar(zeff)

    # compute zeff and other derived arrays elementwise
    for i in 1:length(zeff)
        zeff[i] = zimp - (kin_val.fs[i,1]/kin_val.fs[i,2])*zi*(zimp-zi)
    end
    # zpitch used in later formulas
    for i in 1:length(zpitch)
        zpitch[i] = 1.0 + (1.0 + mimp)/(2.0*mimp)*zimp*(zeff[i]-1.0)/(zimp-zeff[i])
    end

    # Fill kin fields similar to Fortran; assumes appropriate indexing
    for i in 1:size(kin_val.fs,1)
        kin_val.fs[i,6] = 17.3 - 0.5*log(kin_val.fs[i,2]/1.0e20) + 1.5*log(kin_val.fs[i,4]/1.602e-16)
        kin_val.fs[i,7] = (zpitch[i]/3.5e17)*kin_val.fs[i,1]*kin_val.fs[i,6]/(sqrt(1.0*mi)*(kin_val.fs[i,3]/1.602e-16)^1.5)
        kin_val.fs[i,8] = (zpitch[i]/3.5e17)*kin_val.fs[i,2]*kin_val.fs[i,6]/(sqrt(Utilities.me/Utilities.mp)*(kin_val.fs[i,4]/1.602e-16)^1.5)
        kin_val.fs[i,9] = zeff[i]
    end

    Splines.spline_fit(kin_val, "extrap")

    # manipulate N and T profiles
    kin_val.fs[:,1:2] .*= nfac
    kin_val.fs[:,3:4] .*= tfac

    # rotation manipulation
    welec .= wefac .* kin_val.fs[:,5]
    for i in 1:length(welec)
        if welec[i] == 0.0
            welec[i] = 1e-9
        end
    end

    # compute wdian, wdiat, wpefac -- placeholders calling Utilities or other arrays
    # For now use simple translations; these reference kin%fs1 in Fortran which
    # may be first derivative; here we leave as TODO to compute accurate derivatives
    wdian = zeros(length(welec))
    wdiat = zeros(length(welec))
    wpefac = zeros(length(welec))

    # indirect manipulation
    # Avoid division by zero
    for i in 1:length(welec)
        if welec[i] != 0.0
            wpefac[i] = (wpfac*(welec[i]+wdian[i]+wdiat[i]) - (wdian[i]+wdiat[i]))/welec[i]
        else
            wpefac[i] = 1.0
        end
    end
    welec .= wpefac .* welec
    kin_val.fs[:,5] = welec

    Splines.spline_fit(kin_val, "extrap")

    if write_log
        out_unit = Utilities.get_free_file_unit(-1)
        println("Writing kinetic spline to pentrc_kinetics.out")
        Splines.spline_write1(kin_val, true, false, out_unit, 0, true)
    end

    # set module-level kin
    kin[] = kin_val
    return kin_val
end

## (Inputs module will remain open here; InputsExtras will be defined nested below)

# ---------------------------------------------------------------------
# Additional converted subroutines from inputs.f90
# These are higher-level ports: read_pmodb, read_peq, set_peq.
# They rely on Utilities, Splines, BicubicSplines, and DconInterface modules.
# Some detailed numerical transforms (DFT kernels, bicube_eval internals)
# are delegated back to those modules; TODO markers indicate places to
# refine when those modules' Julia APIs are available.
# ---------------------------------------------------------------------

module InputsExtras
# Refer to the top-level modules (they are included into `Main` by
# the loader logic in `Inputs`). Use `Main` to access them reliably.
# NOTE: This file intentionally leaves the full DCON implementation as a
# TODO. The code calls `DconInterface.*` functions in a few places to
# preserve the structure of the original Fortran port, but the
# DCON interface remains a stub in this branch (option B). When you are
# ready to expand DCON, replace the stubbed `src/pentrc/dcon_interface.jl`
# with the full port and remove/resolve the TODOs below.
using Main.Inputs: newm, kin
const Inputs = Main.Inputs
using Main.DconInterface
using Main.Utilities
using Main.SplinesMod
# Provide aliases expected by the rest of this file
const Splines = Main.SplinesMod
const BicubicSpline = Main.SplinesMod.BicubicSpline

export read_pmodb, read_peq, set_peq

function read_pmodb(file::AbstractString, jac_in::AbstractString="", jsurf_in::Int=1, tmag_in::Int=1, debug::Bool=false, op_powin::Union{Nothing,NTuple{4,Int}}=nothing)
    # Read psi,m matrix of displacements and form dbob_m and divx_m splines
    table, titles = Utilities.readtable(file, verbose=debug)
    # table expected shape: (npsi*nm, columns) with columns [psi, m, ...]
    nm = Utilities.nunique(view(table,:,2))
    nrows = size(table,1)
    npsi = Int(nrows / nm)
    if npsi * nm != nrows
        error("ERROR - inputs - size of table not equal to product of unique psi & m")
    end

    # determine ordering: firstnm = nunique(table[1:nm,2])
    firstnm = Utilities.nunique(table[1:nm,2])
    if firstnm == nm
        # psi outer loop
        ms = Int.(table[1:nm,2])
    psi = [ table[j,1] for j in 1:nm:nrows ]
        # reshape complex spectra columns
        # Data is arranged in psi-blocks of size `nm`, so reshape as
        # (nm, npsi) then transpose to get (npsi, nm) rows per psi.
        lagbmni = reshape(table[:,5] .+ im*table[:,6], nm, npsi)'
        divxmni = reshape(table[:,7] .+ im*table[:,8], nm, npsi)'
    else
        # m outer loop
    ms = [ Int(table[i,2]) for i in 1:npsi:nrows ]
        psi = table[1:npsi,1]
        lagbmni = reshape(table[:,5] .+ im*table[:,6], npsi, nm)
        divxmni = reshape(table[:,7] .+ im*table[:,8], npsi, nm)
    end

    # Decide coordinate jacobian powers
    if jac_in == "" || jac_in == "default"
        jac_in = DconInterface.jac_type
        if DconInterface.verbose
            println("  -> WARNING: Assuming DCON ", DconInterface.jac_type, " coordinates")
        end
    end

    # map jac_in to powers (powin)
    powin = (-1,-1,-1,-1)
    if jac_in == "hamada"
        powin = (0,0,0,0)
    elseif jac_in == "pest"
        powin = (0,0,2,0)
    elseif jac_in == "equal_arc"
        powin = (0,1,0,0)
    elseif jac_in == "boozer"
        powin = (2,0,0,0)
    elseif jac_in == "park"
        powin = (1,0,0,0)
    elseif jac_in == "polar"
        powin = (0,1,0,1)
    elseif jac_in == "other"
        if op_powin === nothing
            error("Missing op_powin for jac_in='other'")
        end
        powin = op_powin
    else
        error("ERROR: inputs - jac_in must be one of the allowed names")
    end

    # Convert spectra to working mpert size using newm
    mpert = DconInterface.mpert[]
    mfac = DconInterface.mfac
    lagbmns = similar(lagbmni, npsi, mpert)
    divxmns = similar(divxmni, npsi, mpert)
    for i in 1:npsi
        if mpert > nm
            lagbmns[i,:] = newm(ms, vec(lagbmni[i,:]), collect(1:mpert))
            divxmns[i,:] = newm(ms, vec(divxmni[i,:]), collect(1:mpert))
            # TODO (DCON): `idcon_coords` is required to map the m-space
            # coordinates into DCON's internal representation. The current
            # repository includes a lightweight DCON stub so the port can
            # load; implement the full DCON behavior later. For now we
            # call the function and expect the DCON stub to be present.
            DconInterface.idcon_coords(psi[i], lagbmns[i,:], mfac, mpert, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            DconInterface.idcon_coords(psi[i], divxmns[i,:], mfac, mpert, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
        else
            DconInterface.idcon_coords(psi[i], lagbmni[i,:], ms, nm, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            DconInterface.idcon_coords(psi[i], divxmni[i,:], ms, nm, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            lagbmns[i,:] = newm(ms, vec(lagbmni[i,:]), collect(1:mpert))
            divxmns[i,:] = newm(ms, vec(divxmni[i,:]), collect(1:mpert))
        end
        if debug && i % max(1,Int(round(npsi/10))) == 0
            println("progress: ", i, "/", npsi)
        end
    end

    # Build bicubic grids (xs = psi, ys = mode index grid) and fill the
    # function arrays by performing DFT/back-transforms per-psi row.
    xs_bi = psi
    # TODO: confirm the `ys_bi` grid is correct. The Fortran version uses a
    # particular mapping of mode indices (including possible negative modes
    # or a specific ordering). Currently we use a simple 1..mpert index:
    #    ys_bi = collect(1:mpert)
    # Verify this against the Fortran inputs.f90 DFT/mode conventions and
    # change to use actual mode numbers (ms or signed modes) if required.
    ys_bi = Float64.(collect(1:mpert))
    # fs has shape (npsi, mpert, nqty). We treat each field as scalar (nqty=1)
    fs_dbob = zeros(Float64, npsi, mpert, 1)
    fs_divx = zeros(Float64, npsi, mpert, 1)

    mvec = collect(1:mpert)
    for i in 1:npsi
    lagbfun = Utilities.iscdftb(mvec, lagbmns[i,:], DconInterface.mthsurf[])
    divxfun = Utilities.iscdftb(mvec, divxmns[i,:], DconInterface.mthsurf[])
    for j in 1:(DconInterface.mthsurf[]+1)
            # Evaluate eqfun at (psi,theta) to get normalization factor, if available
            if DconInterface.eqfun !== nothing
                try
                    Splines.bicube_eval!(DconInterface.eqfun, psi[i], DconInterface.theta[j])
                catch
                    # If eqfun is not a bicube object, skip eval but attempt to read a value
                    nothing
                end
                denom = try
                    getfield(DconInterface.eqfun, :_f)[1]
                catch
                    try
                        DconInterface.eqfun.f[1]
                    catch
                        1.0
                    end
                end
            else
                denom = 1.0
            end
            lagbfun[j] /= denom
            divxfun[j] /= denom
        end
    fs_dbob[i, :, 1] .= real.(Utilities.iscdftf(mvec, lagbfun))
    fs_divx[i, :, 1] .= real.(Utilities.iscdftf(mvec, divxfun))
    end

    dbob_m_local = BicubicSpline(xs_bi, ys_bi, fs_dbob; bctypex="extrap", bctypey="not-a-knot")
    divx_m_local = BicubicSpline(xs_bi, ys_bi, fs_divx; bctypex="extrap", bctypey="not-a-knot")

    # set global refs
    Inputs.dbob_m[] = dbob_m_local
    Inputs.divx_m[] = divx_m_local

    return (dbob_m_local, divx_m_local)
end


function read_peq(file::AbstractString, jac_in::AbstractString="", jsurf_in::Int=1, tmag_in::Int=1, force_xialpha::Bool=false, debug::Bool=false, op_powin::Union{Nothing,NTuple{4,Int}}=nothing)
    # Similar structure to read_pmodb but handles three displacement components
    table, titles = Utilities.readtable(file, verbose=debug)
    nm = Utilities.nunique(view(table,:,2))
    npsi = Int(size(table,1) / nm)
    if npsi * nm != size(table,1)
        error("ERROR - inputs - size of table not equal to product of unique psi & m")
    end

    # allocate arrays and reshape similar to Fortran
    firstnm = Utilities.nunique(table[1:nm,2])
    if firstnm == nm
        ms = Int.(table[1:nm,2])
    psi = [ table[j,1] for j in 1:nm:size(table,1) ]
        # reshape into (nm, npsi) and transpose so row i is psi i
        xmp1mni = reshape(table[:,3] .+ im*table[:,4], nm, npsi)'
        xspmni  = reshape(table[:,5] .+ im*table[:,6], nm, npsi)'
        xmsmni  = reshape(table[:,7] .+ im*table[:,8], nm, npsi)'
    else
    ms = [ Int(table[i,2]) for i in 1:npsi:size(table,1) ]
        psi = table[1:npsi,1]
        xmp1mni = reshape(table[:,3] .+ im*table[:,4], npsi, nm)
        xspmni  = reshape(table[:,5] .+ im*table[:,6], npsi, nm)
        xmsmni  = reshape(table[:,7] .+ im*table[:,8], npsi, nm)
    end

    # Map jac_in to powers (same as read_pmodb)
    if jac_in == "" || jac_in == "default"
        jac_in = DconInterface.jac_type
    end
    # compute powin as in read_pmodb
    powin = (-1,-1,-1,-1)
    if jac_in == "hamada"
        powin = (0,0,0,0)
    elseif jac_in == "pest"
        powin = (0,0,2,0)
    elseif jac_in == "equal_arc"
        powin = (0,1,0,0)
    elseif jac_in == "boozer"
        powin = (2,0,0,0)
    elseif jac_in == "park"
        powin = (1,0,0,0)
    elseif jac_in == "polar"
        powin = (0,1,0,1)
    elseif jac_in == "other"
        if op_powin === nothing
            error("Missing op_powin for jac_in='other'")
        end
        powin = op_powin
    else
        error("ERROR: inputs - jac_in must be one of the allowed names")
    end
    # Convert spectra using newm and idcon_coords calls similar to read_pmodb
    mpert = DconInterface.mpert[]
    mfac = DconInterface.mfac
    # allocate converted arrays: shape (npsi, mpert)
    xmp1mns = Array{ComplexF64}(undef, npsi, mpert)
    xspmns  = Array{ComplexF64}(undef, npsi, mpert)
    xmsmns  = Array{ComplexF64}(undef, npsi, mpert)

    for i in 1:npsi
        if mpert > nm
            xmp1mns[i, :] = newm(ms, vec(xmp1mni[i, :]), collect(1:mpert))
            xspmns[i, :]  = newm(ms, vec(xspmni[i, :]),  collect(1:mpert))
            xmsmns[i, :]  = newm(ms, vec(xmsmni[i, :]),  collect(1:mpert))
            # TODO (DCON): idcon_coords is a DCON-specific mapping routine.
            # The full DCON behavior and side-effects are intentionally not
            # implemented here; this is left for a later DCON port step.
            DconInterface.idcon_coords(psi[i], xmp1mns[i, :], mfac, mpert, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            DconInterface.idcon_coords(psi[i], xspmns[i, :], mfac, mpert, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            DconInterface.idcon_coords(psi[i], xmsmns[i, :], mfac, mpert, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
        else
            # inform dcon about the coordinates for the original ms ordering
            DconInterface.idcon_coords(psi[i], xmp1mni[i, :], ms, nm, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            DconInterface.idcon_coords(psi[i], xspmni[i, :],  ms, nm, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            DconInterface.idcon_coords(psi[i], xmsmni[i, :],  ms, nm, powin[3], powin[2], powin[1], powin[4], tmag_in, jsurf_in)
            # still create an mpert-sized working copy for later DFTs
            xmp1mns[i, :] = newm(ms, vec(xmp1mni[i, :]), collect(1:mpert))
            xspmns[i, :]  = newm(ms, vec(xspmni[i, :]),  collect(1:mpert))
            xmsmns[i, :]  = newm(ms, vec(xmsmni[i, :]),  collect(1:mpert))
        end
        if debug && i % max(1,Int(round(npsi/10))) == 0
            println("read_peq progress: ", i, "/", npsi)
        end
    end

    # If force_xialpha: build matrices and invert using LAPACK via LinearAlgebra
    if force_xialpha
        for i in 1:npsi
            # TODO (DCON): This block depends on DCON's matrix assembly
            # (idcon_matrix) and on the matrices amat/bmat/cmat that DCON
            # produces. The current repo intentionally leaves those
            # routines as stubs. When porting DCON, implement
            # idcon_matrix and ensure amat/bmat/cmat/chi1 are provided
            # with the correct shapes/semantics.
            DconInterface.idcon_matrix(psi[i])
            # build ainv from DconInterface.amat and invert
            A = copy(DconInterface.amat)
            Ainv = inv(A)  # uses BLAS/LAPACK under the hood
            xmsmni[i,:] = (1.0 / DconInterface.chi1) * ( - (Ainv * (DconInterface.bmat * xmp1mni[i,:] + DconInterface.cmat * xspmni[i,:])) )
        end
    end

    # set_peq on the valid psi range
    ipsilow = 1
    ipsihigh = npsi
    # determine range consistent with sq spline
    # Allow DconInterface.sq to be either an object with field `xs` or a raw vector
    sq_xs = try
        hasproperty(DconInterface.sq, :xs) ? getproperty(DconInterface.sq, :xs) : DconInterface.sq
    catch
        DconInterface.sq
    end

    for i in 1:npsi
        if psi[i] >= sq_xs[1]
            ipsilow = i
            break
        end
    end
    for i in npsi:-1:1
        if psi[i] <= sq_xs[end]
            ipsihigh = i
            break
        end
    end

    # Pass the mpert-sized working arrays (xmp1mns etc.) into set_peq so
    # DFTs and bicubic construction use the correct mode resolution.
    set_peq(psi[ipsilow:ipsihigh], ms, xmp1mns[ipsilow:ipsihigh,:], xspmns[ipsilow:ipsihigh,:], xmsmns[ipsilow:ipsihigh,:], true, debug)
    return nothing
end


function set_peq(psi::Vector{Float64}, ms::Vector{Int}, xmp1mns::Array{ComplexF64,2}, xspmns::Array{ComplexF64,2}, xmsmns::Array{ComplexF64,2}, op_set_dbdx::Bool=true, op_debug::Bool=false)
    # Set m-quantity csplines for perturbed fields. This mirrors the Fortran set_peq.
    npsi = length(psi)
    nm = length(ms)

    # Build bicubic splines for the three perturbed fields using the
    # converted mpert-space spectra. We'll perform DFT/back-transform per
    # psi row (using Utilities.iscdftb/iscdftf) and normalize by eqfun as in
    # read_pmodb. This produces three BicubicSpline objects stored in
    # Inputs.xs_m[1..3].

    ys_bi = Float64.(collect(1:DconInterface.mpert[]))
    # TODO: confirm ys_bi should be ms or signed mode numbers (see TODO in read_pmodb)

    fs_xmp1 = zeros(Float64, npsi, DconInterface.mpert[], 1)
    fs_xsp  = zeros(Float64, npsi, DconInterface.mpert[], 1)
    fs_xms  = zeros(Float64, npsi, DconInterface.mpert[], 1)

    mvec = collect(1:DconInterface.mpert[])
    for i in 1:npsi
    fun1 = Utilities.iscdftb(mvec, xmp1mns[i,:], DconInterface.mthsurf[])
    fun2 = Utilities.iscdftb(mvec, xspmns[i,:], DconInterface.mthsurf[])
    fun3 = Utilities.iscdftb(mvec, xmsmns[i,:], DconInterface.mthsurf[])

        # Normalize by eqfun at this psi/theta sampling (if available)
    for j in 1:(DconInterface.mthsurf[]+1)
            if DconInterface.eqfun !== nothing
                try
                    Splines.bicube_eval!(DconInterface.eqfun, psi[i], DconInterface.theta[j])
                catch
                    nothing
                end
                denom = try
                    getfield(DconInterface.eqfun, :_f)[1]
                catch
                    try
                        DconInterface.eqfun.f[1]
                    catch
                        1.0
                    end
                end
            else
                denom = 1.0
            end
            fun1[j] /= denom
            fun2[j] /= denom
            fun3[j] /= denom
        end

    fs_xmp1[i, :, 1] .= real.(Utilities.iscdftf(mvec, fun1))
    fs_xsp[i, :, 1]  .= real.(Utilities.iscdftf(mvec, fun2))
    fs_xms[i, :, 1]  .= real.(Utilities.iscdftf(mvec, fun3))
    end

    xs_m1 = BicubicSpline(psi, ys_bi, fs_xmp1; bctypex="extrap", bctypey="not-a-knot")
    xs_m2 = BicubicSpline(psi, ys_bi, fs_xsp;  bctypex="extrap", bctypey="not-a-knot")
    xs_m3 = BicubicSpline(psi, ys_bi, fs_xms;  bctypex="extrap", bctypey="not-a-knot")

    Inputs.xs_m[1][] = xs_m1
    Inputs.xs_m[2][] = xs_m2
    Inputs.xs_m[3][] = xs_m3

    if op_set_dbdx
        # compute dbob_m and divx_m similarly (delegating to Utilities/bicube routines)
        # TODO: implement full weighting by 1/B and any additional DFT normalization
        # required to exactly match the Fortran implementation.
        println("set_peq: op_set_dbdx true — dbob_m/divx_m computation TODO")
    end

    return nothing
end

end # module InputsExtras

end # module Inputs
