# Setting up GPEC

- [Setting up GPEC](#setting-up-gpec)
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
    sudo apt install build-essential cmake -y
    ```

    `build-essential` → GCC, make

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

5. Clone GPEC into your WSL home folder.
Clone it from GitHub directly to your virtual machine.

    ```shell
    git clone https://github.com/OpenFUSIONToolkit/GeneralizedPerturbedEquilibrium.git
    cd GeneralizedPerturbedEquilibrium
    ```

6. Install the Julia packages for GPEC
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

7. At this point, you should be able to run the code, open a `.ipynb` notebook, or connect VS Code to your WSL session.
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
        3. Click Open Folder and then navigate to the GPEC folder on your VM. Open your GPEC folder from WSL: ~/GeneralizedPerturbedEquilibrium.

        If this is not working, you can launch vscode from the WSL shell you have using the command `code .`

        4. Open a terminal inside VS Code — it will automatically use WSL/Ubuntu.
        5. You can now run:
            ```shell
            julia       # run scripts
            jupyter notebook --no-browser
            ```

        6. VS Code also lets you open `.ipynb` notebooks in the WSL environment using the Jupyter extension. Click the "Select Kernel" button in the top right hand of the `.ipynb` file and select the Julia kernel installed in WSL. All dependencies are accessible.
    3.  Run GPEC
        1. Launch Julia and run your scripts as usual:
            ```shell
            include("path/to/script.jl")
            ```

## On macOS

(To be completed)

## Running GPEC

Once GPEC is installed and built, you can run it in two ways:

### As a Command-Line Script

GPEC includes an executable script (`gpec`) in the project root directory. To run GPEC on a directory containing a `gpec.toml` configuration file:

```bash
./gpec path/to/directory
```

**Example:**
```bash
# Run GPEC on one of the included examples
./gpec examples/DIIID-like_ideal_example

# Run in the current directory (must contain gpec.toml)
./gpec
```

The script will:
1. Read configuration from `gpec.toml` in the specified directory
2. Load or generate the equilibrium based on the `[Equilibrium]` section
3. Compute force-free states (stability analysis) based on the `[ForceFreeStates]` section
4. If a `[PerturbedEquilibrium]` section exists, compute the plasma response to external perturbations
5. Write output to HDF5 files as configured

**Early Termination:** You can stop execution early by setting:
- `force_termination = true` in `[Equilibrium]` to stop after equilibrium setup
- `force_termination = true` in `[ForceFreeStates]` to stop after stability analysis (before perturbed equilibrium)

### As a Julia Library

You can also use GPEC programmatically in your own Julia scripts or notebooks:

```julia
using GeneralizedPerturbedEquilibrium

# Run the full GPEC analysis pipeline
GeneralizedPerturbedEquilibrium.main(["path/to/directory"])

# Or access individual modules
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Vacuum
using GeneralizedPerturbedEquilibrium.ForceFreeStates

# Set up equilibrium only
equil = Equilibrium.setup_equilibrium("path/to/gpec.toml")

# Access equilibrium data
println("q at axis: ", equil.params.q0)
println("Beta-N: ", equil.params.betan)
```

### Configuration Files

GPEC uses TOML configuration files (`gpec.toml`) with the following main sections:

- **`[Equilibrium]`**: Equilibrium solver settings (input file, grid resolution, coordinate system, etc.)
- **`[Wall]`**: Wall geometry for vacuum calculations (shape, size, position)
- **`[ForceFreeStates]`**: Stability analysis settings (mode numbers, tolerances, flags)
- **`[PerturbedEquilibrium]`**: Plasma response settings (forcing data, output options)

See the example directories for complete configuration file templates.

(Setup instructions to be added)

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
cd /path/to/GPEC
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
cd /path/to/GPEC
nbstripout --install
```

This ensures notebooks are cleaned even if pre-commit is bypassed with `--no-verify`. However, this is **optional** and not required for normal development - the pre-commit hook is sufficient for most workflows.
