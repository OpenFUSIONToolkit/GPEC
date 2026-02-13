# Setting up JPEC

- [Setting up JPEC](#setting-up-jpec)
  - [On Windows via WSL (Ubuntu)](#on-windows-via-wsl-ubuntu)
  - [On macOS](#on-macos)
  - [Pre-commit Hooks (Optional Developer Tools)](#pre-commit-hooks-optional-developer-tools)

## On Windows via WSL (Ubuntu)
1. Install WSL and Ubuntu
   If you don't already have WSL and Ubuntu installed, set this up. [This page](https://learn.microsoft.com/en-us/windows/wsl/install) gives detailed instructions on how to complete the installation. In the Windows Powershell,

   1. Make sure WSL is installed:
        ```PowerShell
        wsl --install
        ```
    2. Set Ubuntu as your default WSL distro:
        ```PowerShell
        wsl --set-default Ubuntu
        ```
    3. Launch Ubuntu and update:
        ```PowerShell
        sudo apt update && sudo apt upgrade -y
        ```
2. Install build tools in WSL
    ```shell
    sudo apt install build-essential gfortran cmake -y
    ```

    `build-essential` → GCC, make

    `gfortran` → Fortran compiler

    `cmake` → sometimes needed by dependencies

3. Install Julia in WSL

    1. Download the latest Linux tarball from the official site [Julia downloads](https://julialang.org/downloads/). It will look like
        ```shell
        wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.3-linux-x86_64.tar.gz
        ```

        ☆ Replace the URL with the latest stable version.

    2. Extract and move it to /opt (or any path):

        ```shell
        tar -xvzf julia-1.11.3-linux-x86_64.tar.gz
        sudo mv julia-1.11.3 /opt/
        ```

        ☆ Ensure these commands match the tarball you installed. These commands match the above tarball and might need to be modified for you installation.

    3. Add Julia to PATH:

        ```shell
        echo 'export PATH=/opt/julia-1.11.3/bin:$PATH' >> ~/.bashrc
        source ~/.bashrc
        ```

    4. Test it is properly installed

        ```shell
        julia --version
        ```

4. Install Python/Jupyter in WSL

   This step is only really required if you want to run the `.ipynb` test notebooks. You do not necessarily need Python3 installed, but Jupyter runs on a Python server.
   If you do not want to install Python3 and Jupyter, you can install the "IJulia" package to your Julia environment instead and run the command 'notebook()' in the terminal.
   1. To install Python3 and Jupyter notebooks, use these commands
        ```shell
        sudo apt install python3-pip python3-venv -y
        python3 -m pip install --user jupyter jupyterlab notebook ipykernel
        ```
    ⚠ Important: Add local Python scripts to PATH:

        ```shell
        echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
        source ~/.bashrc
        ```
    2. Verify it is properly installed

        ```shell
        jupyter --version
        ```

5. Clone JPEC into your WSL home folder.
Clone it from GitHub directly to your virtual machine.

    ```shell
    git clone https://github.com/OpenFUSIONToolkit/JPEC.git
    cd JPEC
    ```

6. Build Fortran dependencies (libspline.so)
    1. Go to the spline source folder:
        ```shell
        cd ~/JPEC/src/Splines/fortran
        ```

    2. Clean previous builds using
        ```shell
        make clean
        ```

    3. Build
        ```shell
        make
        ```

    4. Verify the library exists using
        ```shell
        ls ../../../deps/libspline.so
        ```

    5. Export library path so Julia can find it
        ```shell
        export LD_LIBRARY_PATH=~/JPEC/deps:$LD_LIBRARY_PATH
        ```

        Optional: add to `~/.bashrc` for persistence.

7. Install the Julia packages for JPEC
    1. Launch Julia:
        ```shell
        julia
        ```
    2. In Julia REPL:
        ``` julia
        using Pkg
        Pkg.instantiate()       # install recorded dependencies
        Pkg.add("Preferences")  # install missing dependency if needed
        Pkg.build("IJulia")     # rebuild kernel
        Pkg.precompile()        # precompile all packages - probably unnecessary
        ```

8. At this point, you should be able to run the code, open a `.ipynb` notebook, or connect VS Code to your WSL session.
    1. To open a .ipynb notebook
        1. Launch Jupyter from WSL, make sure you have exited Julia using the `exit()` command and then type in the shell
        ```shell
        jupyter notebook --no-browser
        ```
        It will print a URL with a token.

        2. Copy the URL into your Windows browser **OR** open the notebook in VS Code using the **Remote - WSL** extension.

    2. (Optionally) Integrate WSL with VS Code
        1. Install **Remote - WSL** extension in VS Code.
        2. Open VS Code → Connect To → Connect to WSL.
        3. Click Open Folder and then navigate to the JPEC folder on your VM. Open your JPEC folder from WSL: ~/JPEC.

        If this is not working, you can launch vscode from the WSL shell you have using the command `code .`

        4. Open a terminal inside VS Code — it will automatically use WSL/Ubuntu.
        5. You can now run:
            ```shell
            make        # rebuild libspline.so if needed
            julia       # run scripts
            jupyter notebook --no-browser
            ```

        6. VS Code also lets you open `.ipynb` notebooks in the WSL environment using the Jupyter extension. Click the "Select Kernel" button in the top right hand of the `.ipynb` file and select the Julia kernel installed in WSL.All dependencies (libspline.so, Julia packages) are accessible.
    3.  Run JPEC
        1. Make sure you are in WSL terminal, with `LD_LIBRARY_PATH` set to include deps.
        2. Launch Julia and run your scripts as usual:
            ```shell
            include("path/to/jpec_script.jl")
            ```

## On macOS

### Prerequisites

Before starting, you'll need a Terminal app to enter commands. You can find it by:
- Press `Cmd + Space` to open Spotlight
- Type "Terminal" and press Enter

Keep this Terminal window open throughout the installation process.

### 1. Install Xcode Command Line Tools

These tools provide the compilers needed to build Fortran code.

1. Open Terminal and run:
   ```bash
   xcode-select --install
   ```

2. A dialog will appear asking you to install the tools. Click "Install" and wait for it to complete (this may take several minutes).

3. Verify installation:
   ```bash
   gcc --version
   ```
   You should see output showing the GCC version.

### 2. Install Homebrew (Package Manager)

Homebrew makes it easy to install software on macOS. If you already have Homebrew installed, skip to step 3.

1. Install Homebrew by running this command in Terminal:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Follow the on-screen instructions. You may need to enter your Mac password.

3. After installation completes, the installer will show you two commands to run to add Homebrew to your PATH. They will look something like:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```
   **Important:** Copy and run these exact commands from your Terminal output.

4. Verify Homebrew is installed:
   ```bash
   brew --version
   ```

### 3. Install Fortran Compiler

The Fortran compiler is needed to build JPEC's Fortran components.

1. Install GCC (which includes gfortran):
   ```bash
   brew install gcc
   ```
   This may take several minutes to complete.

2. Verify gfortran is installed:
   ```bash
   gfortran --version
   ```

### 4. Install Julia

Julia is the programming language JPEC is written in.

**Option A: Install via Homebrew (Recommended for beginners)**

1. Install Julia:
   ```bash
   brew install julia
   ```

2. Verify Julia is installed:
   ```bash
   julia --version
   ```
   You should see something like `julia version 1.11.x`.

**Option B: Install via Official Installer**

1. Go to [https://julialang.org/downloads/](https://julialang.org/downloads/)

2. Download the macOS installer (`.dmg` file) for the latest stable version (1.11 or higher)

3. Open the downloaded `.dmg` file and drag Julia to your Applications folder

4. Add Julia to your PATH by running in Terminal:
   ```bash
   sudo mkdir -p /usr/local/bin
   sudo ln -s /Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
   ```
   Replace `1.11` with your actual version if different.

5. Verify Julia is installed:
   ```bash
   julia --version
   ```

### 5. Install Python and Jupyter (For Running Notebooks)

Jupyter notebooks (`.ipynb` files) require Python. If you only want to run Julia scripts and not notebooks, you can skip this step.

1. Install Python via Homebrew:
   ```bash
   brew install python
   ```

2. Install Jupyter:
   ```bash
   pip3 install jupyter jupyterlab notebook ipykernel
   ```

3. Verify Jupyter is installed:
   ```bash
   jupyter --version
   ```

### 6. Clone the JPEC Repository

Now we'll download the JPEC code from GitHub.

1. Choose where you want to put JPEC. For example, your home directory:
   ```bash
   cd ~
   ```
   Or create a Code folder:
   ```bash
   mkdir -p ~/Code
   cd ~/Code
   ```

2. Clone JPEC from GitHub:
   ```bash
   git clone https://github.com/OpenFUSIONToolkit/JPEC.git
   ```
   If you don't have `git` installed, macOS will prompt you to install it.

3. Enter the JPEC directory:
   ```bash
   cd JPEC
   ```

### 7. Build Fortran Dependencies

JPEC includes Fortran code that needs to be compiled into libraries.

1. Navigate to the Splines Fortran source folder:
   ```bash
   cd ~/Code/JPEC/src/Splines/fortran
   ```
   **Note:** Adjust the path if you cloned JPEC to a different location (e.g., `~/JPEC` instead of `~/Code/JPEC`).

2. Clean any previous builds:
   ```bash
   make clean
   ```

3. Build the Fortran library:
   ```bash
   make
   ```
   You should see compilation messages and eventually "Build complete!"

4. Verify the spline library was created:
   ```bash
   ls -l ../../../deps/libspline.dylib
   ```
   You should see the file listed.

5. Build the Vacuum Fortran library:
   ```bash
   cd ~/Code/JPEC/src/Vacuum/fortran
   make clean
   make
   ```

6. Verify the vacuum library was created:
   ```bash
   ls -l ../../../deps/libvac.dylib
   ```

7. Return to the JPEC root directory:
   ```bash
   cd ~/Code/JPEC
   ```

### 8. Install Julia Packages

Now we'll install all the Julia packages that JPEC depends on.

1. Launch Julia from the JPEC directory:
   ```bash
   julia --project=.
   ```
   The `--project=.` flag tells Julia to use the JPEC project environment.

2. You should now see the Julia prompt: `julia>`

3. Install all dependencies by typing these commands in the Julia prompt:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```
   This will download and install all required packages. It may take several minutes the first time.

4. Build the Julia kernel for Jupyter (if you installed Jupyter):
   ```julia
   Pkg.add("IJulia")
   Pkg.build("IJulia")
   ```

5. Precompile all packages (optional, but speeds up first use):
   ```julia
   Pkg.precompile()
   ```

6. Test that JPEC loads correctly:
   ```julia
   using JPEC
   ```
   If you see no errors, everything is working!

7. Exit Julia:
   ```julia
   exit()
   ```

### 9. Run the Example Notebook

Now you're ready to run the example!

1. Make sure you're in the JPEC directory:
   ```bash
   cd ~/Code/JPEC
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
   This will open Jupyter in your web browser.

3. In the Jupyter interface, navigate to:
   ```
   examples/DIIID-like_ideal_example/run_and_analyze.ipynb
   ```
   Click on the notebook to open it.

4. Select the Julia kernel:
   - If prompted to select a kernel, choose "Julia 1.11" (or whatever version you installed)
   - If the kernel is already selected, you're ready to go!

5. Run the notebook:
   - Click "Cell" → "Run All" from the menu, or
   - Press `Shift + Enter` to run each cell one at a time

   The first time you run the notebook, it will take a few minutes to compile. Subsequent runs will be faster.

### 10. Troubleshooting

**If you get an error about missing libraries:**

Make sure the Fortran libraries are built. Run from the JPEC directory:
```bash
ls deps/
```
You should see `libspline.dylib` and `libvac.dylib`. If not, repeat step 7.

**If Julia can't find packages:**

Make sure you're running Julia with the project environment:
```bash
cd ~/Code/JPEC
julia --project=.
```

**If the Jupyter kernel isn't found:**

Rebuild IJulia:
```bash
julia --project=. -e 'using Pkg; Pkg.build("IJulia")'
```
Then restart Jupyter.

**If you get permission errors:**

Make sure you have write permissions in the JPEC directory. You may need to use `sudo` for some Homebrew commands, but avoid using `sudo` with Julia commands.

### Alternative: Using VS Code (Optional)

If you prefer using VS Code instead of Jupyter in the browser:

1. Install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)

2. Install the Julia extension:
   - Open VS Code
   - Click the Extensions icon (or press `Cmd + Shift + X`)
   - Search for "Julia" and install the official Julia extension
   - Search for "Jupyter" and install the Jupyter extension

3. Open the JPEC folder in VS Code:
   - Click File → Open Folder
   - Navigate to and select your JPEC directory

4. Open the notebook `examples/DIIID-like_ideal_example/run_and_analyze.ipynb`

5. Click "Select Kernel" in the top right and choose "Julia 1.11"

6. Run the cells using `Shift + Enter`

## Running JPEC

Once JPEC is installed and built, you can run it in two ways:

### As a Command-Line Script

JPEC includes an executable script (`jpec`) in the project root directory. To run JPEC on a directory containing a `jpec.toml` configuration file:

```bash
./jpec path/to/directory
```

**Example:**
```bash
# Run JPEC on one of the included examples
./jpec examples/DIIID-like_ideal_example

# Run in the current directory (must contain jpec.toml)
./jpec
```

The script will:
1. Read configuration from `jpec.toml` in the specified directory
2. Load or generate the equilibrium based on the `[Equilibrium]` section
3. Compute force-free states (stability analysis) based on the `[ForceFreeStates]` section
4. If a `[PerturbedEquilibrium]` section exists, compute the plasma response to external perturbations
5. Write output to HDF5 files as configured

**Early Termination:** You can stop execution early by setting:
- `force_termination = true` in `[Equilibrium]` to stop after equilibrium setup
- `force_termination = true` in `[ForceFreeStates]` to stop after stability analysis (before perturbed equilibrium)

### As a Julia Library

You can also use JPEC programmatically in your own Julia scripts or notebooks:

```julia
using JPEC

# Run the full JPEC analysis pipeline
JPEC.main(["path/to/directory"])

# Or access individual modules
using JPEC.Equilibrium
using JPEC.Vacuum
using JPEC.ForceFreeStates

# Set up equilibrium only
equil = Equilibrium.setup_equilibrium("path/to/jpec.toml")

# Access equilibrium data
println("q at axis: ", equil.params.q0)
println("Beta-N: ", equil.params.betan)
```

### Configuration Files

JPEC uses TOML configuration files (`jpec.toml`) with the following main sections:

- **`[Equilibrium]`**: Equilibrium solver settings (input file, grid resolution, coordinate system, etc.)
- **`[Wall]`**: Wall geometry for vacuum calculations (shape, size, position)
- **`[ForceFreeStates]`**: Stability analysis settings (mode numbers, tolerances, flags)
- **`[PerturbedEquilibrium]`**: Plasma response settings (forcing data, output options)

See the example directories for complete configuration file templates.

## Pre-commit Hooks (Optional Developer Tools)

The repository uses pre-commit hooks to maintain code quality and prevent noisy commits from Jupyter notebook metadata and outputs.

### Why Use Pre-commit Hooks?

Pre-commit hooks automatically:
- Strip Jupyter notebook outputs, execution counts, and metadata (prevents merge conflicts and noisy diffs)
- Format Julia code according to `.JuliaFormatter.toml` settings
- Remove trailing whitespace and fix line endings
- Validate YAML/TOML syntax
- Prevent accidentally committing large files (>5MB)

### Installation

```bash
# Install pre-commit (requires Python/pip)
pip install pre-commit

# Install JuliaFormatter globally (required for Julia code formatting hook)
julia -e 'using Pkg; Pkg.add("JuliaFormatter")'

# Install the git hooks in your local repository
cd /path/to/JPEC
pre-commit install
```

### Usage

Once installed, the hooks run automatically on `git commit`. You can also run them manually:

```bash
# Run on all files in the repository
pre-commit run --all-files

# Run only on currently staged files
pre-commit run
```

### Bypassing Hooks (Not Recommended)

In rare cases where you need to bypass the hooks:

```bash
git commit --no-verify
```

However, this is discouraged as it may introduce notebook clutter or formatting inconsistencies.

### Optional: Standalone nbstripout Filter

For additional protection beyond pre-commit, you can install nbstripout as a git filter:

```bash
# Install nbstripout globally
pip install nbstripout

# Install filter for this repository
cd /path/to/JPEC
nbstripout --install
```

This ensures notebooks are cleaned even if pre-commit is bypassed with `--no-verify`. However, this is **optional** and not required for normal development - the pre-commit hook is sufficient for most workflows.
