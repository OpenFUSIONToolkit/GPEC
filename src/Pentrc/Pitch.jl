"""
Pitch-angle integration helpers used by `tpsi!`.

This file ports the `pitch_integration` layer needed by CLAR/GAR-style kernels:
- `lintgrnd!`
- `lambdaintgrl_lsode`
- `kappaintgrl`
- `kappadjsum`
"""

using Printf

global lambdadebug = false
global lambdaatol = 1e-12
global lambdartol = 1e-9

const pitch_nfs = 14

const pitch_state = Ref(Dict{Symbol,Any}(
    :pitch_ell => 0,
    :pitch_n => 0,
    :pitch_psi => 0.0,
    :pitch_wn => 0.0,
    :pitch_wt => 0.0,
    :pitch_we => 0.0,
    :pitch_nuk => 0.0,
    :pitch_bobmax => 0.0,
    :pitch_epsr => 0.0,
    :pitch_q => 0.0,
    :pitch_rex => 1.0,
    :pitch_imx => 1.0,
    :pitch_method => "",
    :pitch_flambda => nothing,
    :pitch_turns => nothing,
))

get_pitch_var(key::Symbol) = pitch_state[][key]
set_pitch_var!(key::Symbol, val) = (pitch_state[][key] = val)

mutable struct PitchRecord
    is_recorded::Bool
    psi_index::Int
    ell_index::Int
    ell::Vector{Int}
    psi::Vector{Float64}
    fs::Array{Float64,4}
end

PitchRecord() = PitchRecord(false, 0, 0, Int[], Float64[], Array{Float64}(undef, 0, 0, 0, 0))

const pitch_record = [PitchRecord() for _ in 1:nmethods]

function lintgrnd!(ydot::AbstractVector{<:Real}, y, p, lmda::Float64)
    flambda = get_pitch_var(:pitch_flambda)
    flambda === nothing && error("lintgrnd! called without a pitch `flambda` spline")

    flambda_f = Spl.spline_eval!(flambda, lmda)
    wb = real(flambda_f[1])
    wd = real(flambda_f[2])

    pitch_wn = get_pitch_var(:pitch_wn)
    pitch_wt = get_pitch_var(:pitch_wt)
    pitch_we = get_pitch_var(:pitch_we)
    pitch_nuk = get_pitch_var(:pitch_nuk)
    pitch_bobmax = get_pitch_var(:pitch_bobmax)
    pitch_epsr = get_pitch_var(:pitch_epsr)
    pitch_q = get_pitch_var(:pitch_q)
    pitch_ell = get_pitch_var(:pitch_ell)
    pitch_n = get_pitch_var(:pitch_n)
    pitch_rex = get_pitch_var(:pitch_rex)
    pitch_imx = get_pitch_var(:pitch_imx)
    pitch_psi = get_pitch_var(:pitch_psi)
    pitch_method = get_pitch_var(:pitch_method)

    xint =
        if lmda <= pitch_bobmax
            nueff = pitch_nuk
            lnq = pitch_ell + pitch_n * pitch_q
            xint_pos = xintgrl_lsode(pitch_wn, pitch_wt, pitch_we, wd, wb, nueff,
                                     pitch_ell, lnq, pitch_n, pitch_psi, lmda, pitch_method;
                                     op_record=false)
            xint_neg = xintgrl_lsode(pitch_wn, pitch_wt, pitch_we, wd, -wb, nueff,
                                     pitch_ell, lnq, pitch_n, pitch_psi, lmda, pitch_method;
                                     op_record=false)
            xint_pos + xint_neg
        else
            nueff = pitch_nuk / (2 * pitch_epsr)
            lnq = Float64(pitch_ell)
            xintgrl_lsode(pitch_wn, pitch_wt, pitch_we, wd, wb, nueff,
                          pitch_ell, lnq, pitch_n, pitch_psi, lmda, pitch_method;
                          op_record=false)
        end

    fill!(ydot, 0.0)
    for i in 3:flambda.nqty
        fres = flambda_f[i] * (pitch_rex * real(xint) + 1im * pitch_imx * imag(xint))
        ydot[1 + 2 * (i - 3)] = real(fres)
        ydot[2 + 2 * (i - 3)] = imag(fres)
    end

    return nothing
end

function _turns_eval(turns, lmda::Float64)
    if turns === nothing
        return zeros(6)
    end
    vals = Spl.spline_eval!(turns, lmda)
    out = zeros(6)
    ncopy = min(length(vals), 6)
    out[1:ncopy] .= vals[1:ncopy]
    return out
end

function record_pitch_method!(method::String, psi::Float64, ell::Int, fs::Matrix{Float64})
    for m in 1:nmethods
        if method == methods[m]
            if !pitch_record[m].is_recorded
                pitch_record[m].is_recorded = true
                pitch_record[m].psi_index = 0
                pitch_record[m].ell_index = 0
                pitch_record[m].psi = zeros(npsi_out)
                pitch_record[m].ell = zeros(Int, nell_out)
                pitch_record[m].fs = zeros(pitch_nfs, maxstep, nell_out, npsi_out)
            end

            k = pitch_record[m].psi_index
            j = pitch_record[m].ell_index

            if k == 0 || psi != pitch_record[m].psi[k]
                k += 1
            end
            if j == 0 || ell != pitch_record[m].ell[j]
                j = mod(j, nell_out) + 1
            end

            k > npsi_out && error("Too many pitch records for method $method")

            pitch_record[m].fs[:, :, j, k] = fs
            pitch_record[m].ell[j] = ell
            pitch_record[m].psi[k] = psi
            pitch_record[m].psi_index = k
            pitch_record[m].ell_index = j
        end
    end
    return nothing
end

function reset_pitch_record!()
    for m in 1:nmethods
        if pitch_record[m].is_recorded
            pitch_record[m] = PitchRecord()
        end
    end
    return nothing
end

function lambdaintgrl_lsode(wn::Float64, wt::Float64, we::Float64, nuk::Float64,
                            bobmax::Float64, epsr::Float64, q::Float64,
                            flambda::Spl.CubicSpline{ComplexF64}, ell::Int, n::Int,
                            rex::Float64, imx::Float64, psi::Float64,
                            turns=nothing, method::String="rlar";
                            op_record::Bool=false)
    neq = 2 * (flambda.nqty - 2)
    y = zeros(Float64, neq)
    lmda0 = flambda.xs[1]
    lmda1 = flambda.xs[end]

    set_pitch_var!(:pitch_wn, wn)
    set_pitch_var!(:pitch_wt, wt)
    set_pitch_var!(:pitch_we, we)
    set_pitch_var!(:pitch_nuk, nuk)
    set_pitch_var!(:pitch_bobmax, bobmax)
    set_pitch_var!(:pitch_epsr, epsr)
    set_pitch_var!(:pitch_q, q)
    set_pitch_var!(:pitch_ell, ell)
    set_pitch_var!(:pitch_n, n)
    set_pitch_var!(:pitch_rex, rex)
    set_pitch_var!(:pitch_imx, imx)
    set_pitch_var!(:pitch_psi, psi)
    set_pitch_var!(:pitch_method, method)
    set_pitch_var!(:pitch_flambda, flambda)
    set_pitch_var!(:pitch_turns, turns)

    prob = ODEProblem(lintgrnd!, y, (lmda0, lmda1))
    sol = solve(prob, Tsit5();
                abstol=lambdaatol, reltol=lambdartol,
                maxiters=maxstep,
                save_everystep=op_record,
                save_start=op_record)

    if sol.retcode != ReturnCode.Success
        out_unit = "pentrc_lambdaintrgl_lsode.err"
        open(out_unit, "w") do io
            println(io, "psi = ", psi)
            println(io, "lambda t_phi 2ndeltaw int(t_phi) int(2ndeltaw)")
            prob_err = ODEProblem(lintgrnd!, zeros(Float64, neq), (lmda0, lmda1))
            sol_err = solve(prob_err, Tsit5();
                            abstol=lambdaatol, reltol=lambdartol,
                            saveat=range(lmda0, lmda1; length=min(maxstep, 256)),
                            maxiters=maxstep)
            for (idx, lmda) in enumerate(sol_err.t)
                dydt = zeros(Float64, neq)
                lintgrnd!(dydt, sol_err.u[idx], nothing, lmda)
                @printf(io, " %16.8e %16.8e %16.8e %16.8e %16.8e\n",
                        lmda, dydt[1], dydt[2], sol_err.u[idx][1], sol_err.u[idx][2])
            end
        end
        error("ERROR: lambdaintgrl_lsode - too many steps in lambda required.")
    end

    if op_record
        fs = zeros(Float64, pitch_nfs, maxstep)
        istep = 0
        for (idx, lmda) in enumerate(sol.t)
            istep >= maxstep && break
            istep += 1
            dydt = zeros(Float64, neq)
            lintgrnd!(dydt, sol.u[idx], nothing, lmda)
            flambda_f = Spl.spline_eval!(flambda, lmda)
            turns_f = _turns_eval(turns, lmda)
            fs[:, istep] .= (
                lmda,
                dydt[1],
                dydt[2],
                sol.u[idx][1],
                sol.u[idx][2],
                real(flambda_f[1]),
                real(flambda_f[2]),
                real(flambda_f[3]),
                turns_f[1],
                turns_f[2],
                turns_f[3],
                turns_f[4],
                turns_f[5],
                turns_f[6],
            )
        end
        if istep > 0
            for j in istep+1:maxstep
                fs[:, j] .= fs[:, istep]
            end
            record_pitch_method!(method, psi, ell, fs)
        end

        if nlambda_out > 1
            for ilambda in 0:nlambda_out-1
                lmda = lmda0 + (ilambda / (nlambda_out - 1.0)) * (lmda1 - lmda0)
                flambda_f = Spl.spline_eval!(flambda, lmda)
                wb = real(flambda_f[1])
                wd = real(flambda_f[2])
                if lmda > bobmax
                    nueff = nuk / (2 * epsr)
                    lnq = Float64(ell)
                else
                    nueff = nuk
                    lnq = ell + n * q
                end
                xintgrl_lsode(wn, wt, we, wd, wb, nueff, ell, lnq, n, psi, lmda, method; op_record=true)
            end
        else
            xintgrl_lsode(wn, wt, we, real(Spl.spline_eval!(flambda, lmda0)[2]),
                          real(Spl.spline_eval!(flambda, lmda0)[1]), nuk, ell, Float64(ell),
                          n, psi, lmda0, method; op_record=true)
        end
    end

    return ComplexF64.(sol.u[end][1:2:end], sol.u[end][2:2:end])
end

function output_pitch_netcdf(args...; kwargs...)
    @warn "output_pitch_netcdf is not implemented in the Julia port yet." maxlog=1
    return nothing
end

function output_pitch_ascii(args...; kwargs...)
    @warn "output_pitch_ascii is not implemented in the Julia port yet." maxlog=1
    return nothing
end

function tintgrl_grid_pitch(args...)
    return complex(0.0)
end
