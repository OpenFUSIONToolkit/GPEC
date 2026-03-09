# How to Run JPEC FUSE Demonstrations

This guide shows you exactly how to run all the demonstrations yourself.

---

## 🖥️ STEP 1: Open Terminal

**On Mac:**
- Press `Cmd + Space`
- Type "Terminal"
- Press Enter

You should see something like:
```
sophiapuscalau@Sophias-MacBook ~ %
```

---

## 📂 STEP 2: Go to JPEC Directory

Copy and paste this command into the terminal, then press Enter:

```bash
cd /Users/sophiapuscalau/JPEC
```

Your prompt should now show:
```
sophiapuscalau@Sophias-MacBook JPEC %
```

---

## 🎯 DEMO 1: Single-Slice Run

**Command:**
```bash
julia --project=. examples/fuse_single_slice_demo.jl
```

**What it does:**
- Loads one plasma equilibrium (t = 5.0 s)
- Runs MHD stability analysis
- Shows if the plasma is stable or unstable

**Expected output:**
```
======================================================================
JPEC FUSE Single-Slice Demonstration
======================================================================

Step 1: Loading equilibrium data...
  ✓ Equilibrium loaded into dd.equilibrium.time_slice[1]
    - Time: 5.0 s

Step 2: Running JPEC stability analysis...
  [Progress messages...]
  → Most unstable δW = 0.6186716295387088  (0 unstable modes)

Step 3: Interpreting stability results...
  MOST UNSTABLE MODE:
    n_tor = 1
    δW    = 0.618672

  ✓  VERDICT: PLASMA IS STABLE
     → All modes have δW > 0
     → This equilibrium is MHD-stable
```

**Time:** ~30 seconds

---

## 📈 DEMO 2: Time-Series Run

**Command:**
```bash
julia --project=. examples/fuse_timeseries_demo.jl
```

**What it does:**
- Creates 5 equilibrium snapshots (t = 0s, 2s, 4s, 6s, 8s)
- Runs stability analysis on each one
- Shows how stability changes over time

**Expected output:**
```
======================================================================
JPEC FUSE Time-Series Demonstration
======================================================================

Step 1: Creating time-evolving equilibrium scenario...
  ✓ Created dd with 5 equilibrium time slices

Step 2: Running JPEC time-series stability analysis...
  [1/5] Running time slice 1 (t = 0.0 s)...
    → Most unstable δW = 0.6187  (0 unstable modes)
  [2/5] Running time slice 2 (t = 2.0 s)...
    → Most unstable δW = 0.6187  (0 unstable modes)
  [...]

Step 3: Extracting stability evolution...
  Time-series stability summary:
    t (s)  |  Most unstable δW  |  # unstable modes  |  Verdict
    0.0    |  0.6187            |  0                 |  stable ✓
    2.0    |  0.6187            |  0                 |  stable ✓
    4.0    |  0.6187            |  0                 |  stable ✓
    6.0    |  0.6187            |  0                 |  stable ✓
    8.0    |  0.6187            |  0                 |  stable ✓

  ✓ PLASMA REMAINS STABLE throughout the discharge
```

**Time:** ~40-50 seconds

---

## 🧪 RUN ALL TESTS

**Command:**
```bash
julia --project=. -e 'using Test; using JPEC; include("test/runtests_imas.jl")'
```

**What it does:**
- Runs all 27 automated tests
- Verifies that IMAS integration works correctly

**Expected output:**
```
Test Summary:                                 | Pass  Total   Time
IMAS equilibrium: read_imas matches read_efit |    6      6  16.1s
Test Summary:                                  | Pass  Total  Time
IMAS write: write_imas populates dd.mhd_linear |   21     21  0.7s
```

**Result:** All 27 tests should PASS ✅

---

## 📄 READ THE SUMMARY DOCUMENT

**Command:**
```bash
cat examples/FUSE_INTEGRATION_SUMMARY.md
```

**What it does:**
- Displays the complete summary document in the terminal

**Alternative (nicer viewing):**
- Open the file in VS Code or any text editor:
  ```bash
  code examples/FUSE_INTEGRATION_SUMMARY.md
  ```

---

## 🆘 Troubleshooting

### "command not found: julia"
- Julia is not installed or not in your PATH
- Solution: Make sure Julia is installed

### "cannot find file"
- You're not in the right directory
- Solution: Run `cd /Users/sophiapuscalau/JPEC` again

### "No such file or directory"
- The examples directory doesn't exist
- Solution: Check that you're in the JPEC folder with `pwd`

---

## ✅ Quick Verification Checklist

After running all commands, you should have seen:
- ✅ Single-slice demo: "PLASMA IS STABLE ✓"
- ✅ Time-series demo: 5 time slices analyzed
- ✅ Tests: "27 Pass"
- ✅ Summary document displayed

---

## 🎓 What Each Command Means

| Command part | Meaning |
|--------------|---------|
| `julia` | Run the Julia programming language |
| `--project=.` | Use the current directory's project (JPEC) |
| `examples/fuse_single_slice_demo.jl` | Path to the demo script |
| `-e '...'` | Execute the code inside the quotes |
| `cat` | Display file contents |

---

## 📁 File Locations

All the files we created are in:
- **Demos:** `/Users/sophiapuscalau/JPEC/examples/`
  - `fuse_single_slice_demo.jl`
  - `fuse_timeseries_demo.jl`
  - `FUSE_INTEGRATION_SUMMARY.md`

- **Tests:** `/Users/sophiapuscalau/JPEC/test/`
  - `runtests_imas.jl`
  - `runtests_fuse.jl`

- **Core code:** `/Users/sophiapuscalau/JPEC/src/`
  - `JPEC_FUSE.jl`
  - `DCON/WriteImas.jl`
  - `DCON/Main.jl`

---

*Need help? Just ask!* 🚀
