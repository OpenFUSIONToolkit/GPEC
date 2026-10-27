using Dates, SHA, SQLite, Tables, Test, TOML

const HARNESS_SOURCE = joinpath(@__DIR__, "..", "src")
include(joinpath(HARNESS_SOURCE, "types.jl"))
include(joinpath(HARNESS_SOURCE, "env.jl"))
include(joinpath(HARNESS_SOURCE, "database.jl"))

@testset "Regression database writes do not depend on garbage collection" begin
    fingerprint = EnvFingerprint("synthetic", "synthetic", "synthetic", 1, 1, "jit", true)
    quantities = [
        ExtractedQuantity("finite", "finite", 2.5, nothing, nothing, "real", 1e-10),
        ExtractedQuantity("array", "array", nothing, nothing, "[NaN,Infinity,-Infinity,-0.0]", "json_array", 1e-10),
        ExtractedQuantity("complex", "complex", nothing, nothing, "[[1.0,2.0],[NaN,-Infinity]]", "json_array", 1e-10)
    ]
    mktempdir() do directory
        # Discarded row cursors must be closed without waiting for collection.
        gc_enabled = GC.enable(false)
        try
            for index in 1:2
                db = open_database(joinpath(directory, "case$(index).sqlite"))
                try
                    store_run(db, "local", "local", "synthetic", "synthetic", "case", 0.25,
                        quantities; fingerprint=fingerprint)
                    info = get_run_info(db, "local", "case")
                    @test info.success
                    @test info.runtime_s == 0.25
                    @test all(getfield(info.fingerprint, key) == getfield(fingerprint, key)
                              for key in fieldnames(EnvFingerprint))
                    stored = get_quantities(db, "local", "case")
                    @test stored["finite"].value_real == 2.5
                    @test stored["array"].value_text == quantities[2].value_text
                    @test stored["complex"].value_text == quantities[3].value_text
                    store_failed_run(db, "local", "local", "synthetic", "synthetic", "case", "native failure")
                    @test !get_run_info(db, "local", "case").success
                    @test get_run_info(db, "local", "case").error_msg == "native failure"
                    @test isempty(get_quantities(db, "local", "case"))
                finally
                    close_database(db)
                end
            end
        finally
            GC.enable(gc_enabled)
        end
    end
end
