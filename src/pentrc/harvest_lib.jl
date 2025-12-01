# Minimal harvest_lib stub to satisfy pentrc.jl includes during tests
# This provides no-op implementations of the minimal harvest API

module HarvestLibStub

export init_harvest, set_harvest_verbose, set_harvest_payload_str,
       set_harvest_payload_int, set_harvest_payload_bol, set_harvest_payload_dbl,
       idcon_harvest, harvest_send

function init_harvest(name::AbstractString, hlog::String)
    return 0
end
function set_harvest_verbose(v::Integer)
    return 0
end
function set_harvest_payload_str(hlog::String, key::AbstractString, val::AbstractString)
    return 0
end
function set_harvest_payload_int(hlog::String, key::AbstractString, val::Integer)
    return 0
end
function set_harvest_payload_bol(hlog::String, key::AbstractString, val::Bool)
    return 0
end
function set_harvest_payload_dbl(hlog::String, key::AbstractString, val::Float64)
    return 0
end
function idcon_harvest(hlog::String)
    return 0
end
function harvest_send(hlog::String)
    return 0
end

end # module

# Export functions into Main so pentrc.jl can call them as top-level functions
using .HarvestLibStub
const init_harvest = HarvestLibStub.init_harvest
const set_harvest_verbose = HarvestLibStub.set_harvest_verbose
const set_harvest_payload_str = HarvestLibStub.set_harvest_payload_str
const set_harvest_payload_int = HarvestLibStub.set_harvest_payload_int
const set_harvest_payload_bol = HarvestLibStub.set_harvest_payload_bol
const set_harvest_payload_dbl = HarvestLibStub.set_harvest_payload_dbl
const idcon_harvest = HarvestLibStub.idcon_harvest
const harvest_send = HarvestLibStub.harvest_send
