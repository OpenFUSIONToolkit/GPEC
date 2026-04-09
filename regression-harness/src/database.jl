"""
SQLite database for caching regression results.
"""

const SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash TEXT NOT NULL,
    commit_short TEXT NOT NULL,
    commit_date TEXT,
    commit_msg  TEXT,
    case_name   TEXT NOT NULL,
    ran_at      TEXT NOT NULL,
    runtime_s   REAL,
    success     INTEGER NOT NULL DEFAULT 1,
    error_msg   TEXT,
    UNIQUE(commit_hash, case_name)
);

CREATE TABLE IF NOT EXISTS quantities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    qty_name    TEXT NOT NULL,
    qty_label   TEXT,
    value_real  REAL,
    value_int   INTEGER,
    value_text  TEXT,
    value_type  TEXT NOT NULL,
    noise_threshold REAL NOT NULL DEFAULT 0.0,
    UNIQUE(run_id, qty_name)
);

CREATE INDEX IF NOT EXISTS idx_runs_commit ON runs(commit_hash);
CREATE INDEX IF NOT EXISTS idx_runs_case ON runs(case_name);
CREATE INDEX IF NOT EXISTS idx_quantities_run ON quantities(run_id);
"""

"""Materialize SQLite query results as a Vector of NamedTuples."""
function query_rows(db::SQLite.DB, sql::String, params=())
    result = DBInterface.execute(db, sql, params)
    ct = Tables.columntable(result)
    nrows = length(first(ct))
    nrows == 0 && return NamedTuple[]
    return [NamedTuple{keys(ct)}(Tuple(col[i] for col in values(ct))) for i in 1:nrows]
end

function open_database(path::String)::SQLite.DB
    db = SQLite.DB(path)
    DBInterface.execute(db, "PRAGMA journal_mode=WAL")
    DBInterface.execute(db, "PRAGMA foreign_keys=ON")
    for stmt in split(SCHEMA_SQL, ";")
        s = strip(stmt)
        isempty(s) && continue
        DBInterface.execute(db, s)
    end
    return db
end

function close_database(db::SQLite.DB)
    SQLite.close(db)
end

function is_cached(db::SQLite.DB, commit_hash::String, case_name::String)::Bool
    rows = query_rows(db, "SELECT id FROM runs WHERE commit_hash = ? AND case_name = ? AND success = 1",
                      (commit_hash, case_name))
    return !isempty(rows)
end

function delete_cached(db::SQLite.DB, commit_hash::String, case_name::String)
    # ON DELETE CASCADE handles quantities cleanup automatically
    DBInterface.execute(db, "DELETE FROM runs WHERE commit_hash = ? AND case_name = ?",
                        (commit_hash, case_name))
end

function store_run(db::SQLite.DB, commit_hash::AbstractString, commit_short::AbstractString,
                   commit_date::AbstractString, commit_msg::AbstractString,
                   case_name::AbstractString, runtime_s::Float64,
                   extracted::Vector{ExtractedQuantity};
                   success::Bool=true, error_msg::AbstractString="")
    ran_at = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")

    SQLite.transaction(db) do
        delete_cached(db, String(commit_hash), String(case_name))

        DBInterface.execute(db,
            """INSERT INTO runs
               (commit_hash, commit_short, commit_date, commit_msg, case_name, ran_at, runtime_s, success, error_msg)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (String(commit_hash), String(commit_short), String(commit_date), String(commit_msg),
             String(case_name), ran_at, runtime_s, success ? 1 : 0, String(error_msg)))

        run_id = SQLite.last_insert_rowid(db)

        for eq in extracted
            DBInterface.execute(db,
                """INSERT INTO quantities
                   (run_id, qty_name, qty_label, value_real, value_int, value_text, value_type, noise_threshold)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, eq.name, eq.label, eq.value_real, eq.value_int, eq.value_text,
                 eq.value_type, eq.noise_threshold))
        end
    end
end

function store_failed_run(db::SQLite.DB, commit_hash::AbstractString, commit_short::AbstractString,
                          commit_date::AbstractString, commit_msg::AbstractString,
                          case_name::AbstractString, error_msg::AbstractString)
    ran_at = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    delete_cached(db, String(commit_hash), String(case_name))
    DBInterface.execute(db,
        """INSERT INTO runs
           (commit_hash, commit_short, commit_date, commit_msg, case_name, ran_at, runtime_s, success, error_msg)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (String(commit_hash), String(commit_short), String(commit_date), String(commit_msg),
         String(case_name), ran_at, 0.0, 0, String(error_msg)))
end

"""
Get all quantities for a (commit, case) pair. Returns Dict{qty_name => NamedTuple}.
"""
function get_quantities(db::SQLite.DB, commit_hash::String, case_name::String)
    results = Dict{String, NamedTuple}()
    rows = query_rows(db,
        """SELECT q.qty_name, q.qty_label, q.value_real, q.value_int, q.value_text,
                  q.value_type, q.noise_threshold
           FROM quantities q
           JOIN runs r ON q.run_id = r.id
           WHERE r.commit_hash = ? AND r.case_name = ?""",
        (commit_hash, case_name))

    for row in rows
        name = something(row.qty_name, "")
        results[name] = (
            label = something(row.qty_label, name),
            value_real = row.value_real,
            value_int = row.value_int,
            value_text = row.value_text,
            value_type = something(row.value_type, "missing"),
            noise_threshold = something(row.noise_threshold, 0.0),
        )
    end
    return results
end

"""
Get run info for a (commit, case) pair. Returns NamedTuple or nothing.
"""
function get_run_info(db::SQLite.DB, commit_hash::String, case_name::String)
    rows = query_rows(db,
        """SELECT commit_short, commit_date, commit_msg, runtime_s, success, error_msg
           FROM runs WHERE commit_hash = ? AND case_name = ?""",
        (commit_hash, case_name))
    isempty(rows) && return nothing
    row = first(rows)
    return (
        commit_short = something(row.commit_short, ""),
        commit_date = something(row.commit_date, ""),
        commit_msg = something(row.commit_msg, ""),
        runtime_s = something(row.runtime_s, 0.0),
        success = coalesce(row.success, 0) == 1,
        error_msg = something(row.error_msg, ""),
    )
end

"""
Get all cached runs for a case, ordered by commit date.
"""
function get_all_runs_for_case(db::SQLite.DB, case_name::String)
    return query_rows(db,
        """SELECT commit_hash, commit_short, commit_date, commit_msg, runtime_s, success
           FROM runs WHERE case_name = ? ORDER BY commit_date""",
        (case_name,))
end
