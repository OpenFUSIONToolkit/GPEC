const BCTYPE_MAP = Dict(
    "natural" => 1, # unstable
    "periodic" => 2,
    "extrap" => 3,
    "not-a-knot" => 4,
    "notaknot" => 4
)

"""
    parse_bctype(bctype)

Internal helper to parse a boundary condition into a validated integer code.

## Arguments:

  - `bctype`: The boundary condition as a `String` or `Int`. Valid options are:

      + `"natural"` or `1`
      + `"periodic"` or `2`
      + `"extrap"` or `3`
      + `"not-a-knot"` or `4`

## Returns:

  - A validated `Int` code (1-4).
"""
function parse_bctype(bctype::String)
    normalized_bctype = replace(lowercase(bctype), r"[-_\s]" => "")
    if haskey(BCTYPE_MAP, normalized_bctype)
        return BCTYPE_MAP[normalized_bctype]
    else
        error("Invalid string for `bctype`: '$bctype'. Valid options are: $(keys(BCTYPE_MAP))")
    end
end

function parse_bctype(bctype::Int)
    # If it's already an integer, validate it.
    if bctype in values(BCTYPE_MAP)
        return bctype
    else
        error("Invalid integer for `bctype`: $bctype. Valid options are: $(collect(values(BCTYPE_MAP)))")
    end
end

#==============================================================================#
#                            Read-Only Interface Utilities
#==============================================================================#

"""
    ReadOnlyArray{T,N,A}

A thin wrapper that provides a read-only view.
"""
struct ReadOnlyArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end

Base.size(A::ReadOnlyArray) = size(A.data)
Base.getindex(A::ReadOnlyArray, I...) = @inbounds A.data[I...]
Base.setindex!(::ReadOnlyArray, args...) =
    throw(ArgumentError("Cannot modify a read-only array."))


"""
@expose_fields TypeName field1 field2 ...

Create a *read-only public interface* for a struct while still allowing
internal code to mutate the underlying buffers.

For every symbol `field` you list (e.g. `fs`, `fs1`),

* an **internal** field named `:_field` (e.g. `:_fs`, `:_fs1`) is assumed to
exist in `TypeName`;
* the macro generates the following methods:
"""

macro expose_fields(typ, fields...)
    # 1) Create a list of internal field names (Symbol)
    internal_syms = map(f -> Symbol("_" * string(f)), fields)
    # 2) Convert to a tuple literal AST (planted here as a QuoteNode)
    internal_tuple = Expr(:tuple, map(x -> QuoteNode(x), internal_syms)...)

    quote
        ######## getproperty ########
        function Base.getproperty(s::$(esc(typ)), fld::Symbol)
            $(
                foldr(
                (f, rest) -> :(
                    if fld === $(QuoteNode(f))
                        return ReadOnlyArray(getfield(s, $(QuoteNode(Symbol("_", f)))))
                    else
                        $rest
                    end
                ),
                fields;
                init=:(return getfield(s, fld))
            )
            )
        end

        ######## setproperty! ########
        function Base.setproperty!(s::$(esc(typ)), fld::Symbol, val)
            # (where internal_tuple is of the form (: _fs1, :_fs2, ...))
            if fld in $(internal_tuple)
                return setfield!(s, fld, val)
            end
            throw(ArgumentError(
                "Cannot modify property ':$fld'. " *
                "$(string($(esc(typ)))) fields are read-only."
            ))
        end
    end
end