# Run one GPEC case twice (cold + warm) with a timestamping logger so phase boundaries
# can be recovered from the existing @info markers without touching src/.
using Logging, Dates

struct TSLogger <: Logging.AbstractLogger
    inner::Logging.AbstractLogger
end
Logging.min_enabled_level(l::TSLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::TSLogger, level, args...) = Logging.shouldlog(l.inner, level, args...)
Logging.catch_exceptions(l::TSLogger) = true
function Logging.handle_message(l::TSLogger, level, message, _module, group, id, file, line; kwargs...)
    msg = string(message)
    lines = filter(!isempty, split(msg, '\n'))
    tag = isempty(lines) ? "" : String(lines[1])
    println(stdout, "TSMARK ", Dates.datetime2unix(Dates.now()), " | ", tag)
    flush(stdout)
    Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
end

global_logger(TSLogger(global_logger()))

using GeneralizedPerturbedEquilibrium

println("TSMARK ", Dates.datetime2unix(Dates.now()), " | RUN1_START")
GeneralizedPerturbedEquilibrium.main([ARGS[1]])
println("TSMARK ", Dates.datetime2unix(Dates.now()), " | RUN1_END")
println("TSMARK ", Dates.datetime2unix(Dates.now()), " | RUN2_START")
GeneralizedPerturbedEquilibrium.main([ARGS[1]])
println("TSMARK ", Dates.datetime2unix(Dates.now()), " | RUN2_END")
