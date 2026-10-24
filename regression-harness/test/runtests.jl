using Test, HDF5, JSON, SHA
include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "extractor.jl"))

@testset "Regression JSON extraction round trips" begin
    spec(rule, path="data") = QuantitySpec(rule, path, "complex_vector", rule, rule, 1e-10, 1)
    decoded(rule, data) = JSON.parse(apply_extraction(spec(rule), data).value_text; allownan=true)
    finite = ComplexF64[1+2im, -3+4im]
    @test decoded("all_complex", finite) == [[1.0, 2.0], [-3.0, 4.0]]
    @test decoded("first_1_complex", finite) == [[1.0, 2.0]]
    @test decoded("all_real", [1.25, -2.5, -0.0]) == [1.25, -2.5, -0.0]
    @test decoded("first_2", [1.25, -2.5, 9.0]) == [1.25, -2.5]
    @test decoded("diagonal_complex", [1+2im 9+8im; 7+6im 3+4im]) == [[1.0, 2.0], [3.0, 4.0]]

    real_data = [1.25, NaN, Inf, -Inf, -0.0]
    @test isequal(decoded("all_real", real_data), real_data)
    @test isequal(decoded("first_4", real_data), real_data[1:4])
    complex_data = [complex(1.0, 2.0), complex(NaN, Inf), complex(-Inf, -0.0)]
    expected_pairs = [[1.0, 2.0], [NaN, Inf], [-Inf, -0.0]]
    @test isequal(decoded("all_complex", complex_data), expected_pairs)
    @test isequal(decoded("first_2_complex", complex_data), expected_pairs[1:2])
    @test isequal(decoded("diagonal_complex", [complex_data[2] 0; 0 complex_data[3]]), expected_pairs[2:3])

    mktempdir() do dir
        path = joinpath(dir, "synthetic.h5")
        h5open(path, "w") do f
            f["real"] = real_data
            f["complex"] = complex_data
        end
        quantities = extract_quantities(path, [spec("all_real", "real"), spec("all_complex", "complex")], 0.0)
        @test isequal(JSON.parse(quantities[1].value_text; allownan=true), real_data)
        @test isequal(JSON.parse(quantities[2].value_text; allownan=true), expected_pairs)
    end

    nan_value = apply_extraction(spec("all_real"), [NaN])
    finite_value = apply_extraction(spec("all_real"), [1.0])
    asrow(q) = (value_type=q.value_type, value_real=q.value_real, value_int=q.value_int, value_text=q.value_text, noise_threshold=q.noise_threshold)
    @test compare_values(asrow(nan_value), asrow(nan_value))[3] == "OK"
    @test compare_values(asrow(nan_value), asrow(finite_value))[3] == "CHANGED"
    @test compare_values(asrow(finite_value), asrow(nan_value))[3] == "CHANGED"
end
