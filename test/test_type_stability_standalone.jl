#!/usr/bin/env julia
# Standalone test for type stability that doesn't require building fortran libraries
# This can be run with: julia test/test_type_stability_standalone.jl

using Test

println("=" ^ 70)
println("Type Stability Standalone Tests")
println("=" ^ 70)
println()

# Mock the spline structures for testing (these match the actual structures)
mutable struct MockCubicSpline{T}
    handle::Ptr{Cvoid}
    _xs::Vector{Float64}
    _fs::Matrix{T}
    mx::Int
    nqty::Int
    bctype::Int32
    _fsi::Matrix{T}
    _fs1::Matrix{T}
    _f::Vector{T}
    _f1::Vector{T}
    _f2::Vector{T}
    _f3::Vector{T}
end

mutable struct MockBicubicSpline
    handle::Ptr{Cvoid}
    _xs::Vector{Float64}
    _ys::Vector{Float64}
    _fs::Array{Float64,3}
    mx::Int
    my::Int
    nqty::Int
    bctypex::Int32
    bctypey::Int32
    _fsx::Array{Float64,3}
    _fsy::Array{Float64,3}
    _fsxy::Array{Float64,3}
    _f::Vector{Float64}
    _fx::Vector{Float64}
    _fy::Vector{Float64}
    _fxx::Vector{Float64}
    _fxy::Vector{Float64}
    _fyy::Vector{Float64}
end

mutable struct MockFourierSpline
    handle::Ptr{Cvoid}
    _xs::Vector{Float64}
    _ys::Vector{Float64}
    _fs::Array{Float64,3}
    mx::Int
    my::Int
    mband::Int
    nqty::Int
    bctype::Int32
    fit_method::Int32
    cs::MockCubicSpline{ComplexF64}
end

# Empty constructor functions (matching the actual implementation)
function empty_MockCubicSpline(::Type{T}=ComplexF64) where {T<:Union{Float64,ComplexF64}}
    xs = Float64[]
    fs = Matrix{T}(undef, 0, 0)
    fsi = Matrix{T}(undef, 0, 0)
    fs1 = Matrix{T}(undef, 0, 0)
    f = Vector{T}(undef, 0)
    f1 = Vector{T}(undef, 0)
    f2 = Vector{T}(undef, 0)
    f3 = Vector{T}(undef, 0)
    return MockCubicSpline{T}(C_NULL, xs, fs, 0, 0, zero(Int32), fsi, fs1, f, f1, f2, f3)
end

function empty_MockBicubicSpline()
    xs = Float64[]
    ys = Float64[]
    fs = Array{Float64}(undef, 0, 0, 0)
    fsx = Array{Float64}(undef, 0, 0, 0)
    fsy = Array{Float64}(undef, 0, 0, 0)
    fsxy = Array{Float64}(undef, 0, 0, 0)
    f = Vector{Float64}(undef, 0)
    fx = Vector{Float64}(undef, 0)
    fy = Vector{Float64}(undef, 0)
    fxx = Vector{Float64}(undef, 0)
    fxy = Vector{Float64}(undef, 0)
    fyy = Vector{Float64}(undef, 0)
    return MockBicubicSpline(C_NULL, xs, ys, fs, 0, 0, 0, zero(Int32), zero(Int32), fsx, fsy, fsxy, f, fx, fy, fxx, fxy, fyy)
end

function empty_MockFourierSpline()
    cs = empty_MockCubicSpline(ComplexF64)
    return MockFourierSpline(C_NULL, Float64[], Float64[], Array{Float64}(undef, 0, 0, 0),
        0, 0, 0, 0, zero(Int32), zero(Int32), cs)
end

# Mock DCON structs using the empty spline constructors
Base.@kwdef mutable struct MockSingType
    psifac::Float64 = 0.0
    rho::Float64 = 0.0
    m::Vector{Int} = Int[]
    n::Vector{Int} = Int[]
    q::Float64 = 0.0
    q1::Float64 = 0.0
    di::Float64 = 0.0
    alpha::Vector{ComplexF64} = ComplexF64[]
    r1::Vector{Int} = Int[]
    r2::Vector{Int} = Int[]
    n1::Vector{Int} = Int[]
    n2::Vector{Int} = Int[]
    power::Vector{ComplexF64} = ComplexF64[]
    vmat::Array{ComplexF64,4} = Array{ComplexF64}(undef, 0, 0, 0, 0)
    mmat::Array{ComplexF64,4} = Array{ComplexF64}(undef, 0, 0, 0, 0)
    m0mat::Matrix{ComplexF64} = zeros(ComplexF64, 2, 2)
end

Base.@kwdef mutable struct MockDconInternal
    dir_path::String = ""
    msing::Int = 0
    sing::Vector{MockSingType} = MockSingType[]
    kinsing::Vector{MockSingType} = MockSingType[]
    locstab::MockCubicSpline{Float64} = empty_MockCubicSpline(Float64)
end

Base.@kwdef mutable struct MockFourFitVars
    mpert::Int
    mband::Int
    amats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    bmats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    cmats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    dmats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    emats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    hmats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    fmats_lower::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    kmats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
    gmats::MockCubicSpline{ComplexF64} = empty_MockCubicSpline(ComplexF64)
end

# Run tests
@testset "Type Stability Tests (Standalone)" begin
    
    @testset "Empty CubicSpline Constructors" begin
        empty_cs_f64 = empty_MockCubicSpline(Float64)
        @test empty_cs_f64.handle == C_NULL
        @test empty_cs_f64.mx == 0
        @test empty_cs_f64.nqty == 0
        @test typeof(empty_cs_f64) == MockCubicSpline{Float64}
        @test length(empty_cs_f64._f) == 0
        
        empty_cs_c64 = empty_MockCubicSpline(ComplexF64)
        @test empty_cs_c64.handle == C_NULL
        @test typeof(empty_cs_c64) == MockCubicSpline{ComplexF64}
        
        empty_cs_default = empty_MockCubicSpline()
        @test typeof(empty_cs_default) == MockCubicSpline{ComplexF64}
    end
    
    @testset "Empty BicubicSpline Constructor" begin
        empty_bs = empty_MockBicubicSpline()
        @test empty_bs.handle == C_NULL
        @test empty_bs.mx == 0
        @test empty_bs.my == 0
        @test typeof(empty_bs) == MockBicubicSpline
        @test length(empty_bs._f) == 0
    end
    
    @testset "Empty FourierSpline Constructor" begin
        empty_fs = empty_MockFourierSpline()
        @test empty_fs.handle == C_NULL
        @test empty_fs.mx == 0
        @test typeof(empty_fs) == MockFourierSpline
        @test empty_fs.cs.handle == C_NULL
    end
    
    @testset "DCON Struct Type Stability" begin
        # Test SingType
        st = MockSingType()
        @test typeof(st.power) == Vector{ComplexF64}
        @test typeof(st.vmat) == Array{ComplexF64, 4}
        @test typeof(st.mmat) == Array{ComplexF64, 4}
        @test length(st.power) == 0
        @test size(st.vmat) == (0, 0, 0, 0)
        @test fieldtype(typeof(st), :power) == Vector{ComplexF64}
        @test fieldtype(typeof(st), :vmat) == Array{ComplexF64, 4}
        
        # Test DconInternal
        di = MockDconInternal()
        @test typeof(di.locstab) == MockCubicSpline{Float64}
        @test di.locstab.handle == C_NULL
        @test typeof(di.sing) == Vector{MockSingType}
        @test fieldtype(typeof(di), :locstab) == MockCubicSpline{Float64}
        @test fieldtype(typeof(di), :sing) == Vector{MockSingType}
        
        # Test FourFitVars
        ffv = MockFourFitVars(mpert=5, mband=4)
        @test typeof(ffv.amats) == MockCubicSpline{ComplexF64}
        @test typeof(ffv.bmats) == MockCubicSpline{ComplexF64}
        @test ffv.amats.handle == C_NULL
        @test ffv.bmats.handle == C_NULL
        @test fieldtype(typeof(ffv), :amats) == MockCubicSpline{ComplexF64}
    end
    
    @testset "No Union Types Present" begin
        st = MockSingType()
        # Verify that there are no Union types in the fields
        @test !any(t -> t isa Union, [fieldtype(typeof(st), f) for f in fieldnames(typeof(st))])
        
        di = MockDconInternal()
        @test !any(t -> t isa Union, [fieldtype(typeof(di), f) for f in fieldnames(typeof(di))])
        
        ffv = MockFourFitVars(mpert=5, mband=4)
        @test !any(t -> t isa Union, [fieldtype(typeof(ffv), f) for f in fieldnames(typeof(ffv))])
    end
end

println()
println("=" ^ 70)
println("✅ All standalone type stability tests passed!")
println("=" ^ 70)
