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

4. Clone GPEC into your WSL home folder:
    ```shell
    git clone https://github.com/OpenFUSIONToolkit/GeneralizedPerturbedEquilibrium.git
    cd GeneralizedPerturbedEquilibrium
    ```

5. Install the Julia packages for GPEC
    1. Launch Julia:
        ```shell
        julia
        ```
    2. In Julia REPL:
        ```julia
        using Pkg
        Pkg.instantiate()       # install recorded dependencies
        Pkg.add("Preferences")  # install missing dependency if needed
        Pkg.precompile()        # precompile all packages - probably unnecessary
        ```

6. At this point, you should be able to run the code or connect VS Code to your WSL session.
    1. (Optionally) Integrate WSL with VS Code
        1. Install **Remote - WSL** extension in VS Code.
        2. Open VS Code → Connect To → Connect to WSL.
        3. Click Open Folder and then navigate to the GPEC folder on your VM. Open your GPEC folder from WSL: ~/GeneralizedPerturbedEquilibrium.

        If this is not working, you can launch vscode from the WSL shell you have using the command `code .`

        4. Open a terminal inside VS Code — it will automatically use WSL/Ubuntu.
        5. You can now run Julia scripts directly from the terminal.
    2. Run GPEC
        1. Launch Julia and run your scripts as usual:
            ```shell
            include("path/to/script.jl")
            ```

## On macOS

### Prerequisites

Before starting, you'll need a Terminal app to enter commands. You can find it by:
- Press `Cmd + Space` to open Spotlight
- Type "Terminal" and press Enter

Keep this Terminal window open throughout the installation process.

### 1. Install Xcode Command Line Tools

These tools provide the compilers and build utilities needed by GPEC's dependencies.

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

### 3. Install Julia

Julia is the programming language GPEC is written in.

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

### 4. Clone the GPEC Repository

Now we'll download the GPEC code from GitHub.

1. Choose where you want to put GPEC. For example, your home directory:
   ```bash
   cd ~
   ```
   Or create a Code folder:
   ```bash
   mkdir -p ~/Code
   cd ~/Code
   ```

2. Clone GPEC from GitHub:
   ```bash
   git clone https://github.com/OpenFUSIONToolkit/GeneralizedPerturbedEquilibrium.git
   ```
   If you don't have `git` installed, macOS will prompt you to install it.

3. Enter the GPEC directory:
   ```bash
   cd GeneralizedPerturbedEquilibrium
   ```

### 5. Install Julia Packages

Now we'll install all the Julia packages that GPEC depends on.

1. Launch Julia from the GPEC directory:
   ```bash
   julia --project=.
   ```
   The `--project=.` flag tells Julia to use the GPEC project environment.

2. You should now see the Julia prompt: `julia>`

3. Install all dependencies by typing these commands in the Julia prompt:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```
   This will download and install all required packages. It may take several minutes the first time.

4. Precompile all packages (optional, but speeds up first use):
   ```julia
   Pkg.precompile()
   ```

5. Test that GPEC loads correctly:
   ```julia
   using GeneralizedPerturbedEquilibrium
   ```
   If you see no errors, everything is working!

6. Exit Julia:
   ```julia
   exit()
   ```

### 6. Run an Example

Now you're ready to run GPEC! See the [Running GPEC](#running-gpec) section below for full details. As a quick test:

1. Make sure you're in the GPEC directory:
   ```bash
   cd ~/Code/GeneralizedPerturbedEquilibrium
   ```

2. Run GPEC on one of the included examples:
   ```bash
   ./gpec examples/DIIID-like_ideal_example
   ```

   The first time you run GPEC, it will take a few minutes to compile. Subsequent runs will be faster.

### 7. Troubleshooting

**If Julia can't find packages:**

Make sure you're running Julia with the project environment:
```bash
cd ~/Code/GeneralizedPerturbedEquilibrium
julia --project=.
```

**If you get permission errors:**

Make sure you have write permissions in the GPEC directory. You may need to use `sudo` for some Homebrew commands, but avoid using `sudo` with Julia commands.

### Alternative: Using VS Code (Optional)

VS Code provides a convenient IDE for developing with GPEC:

1. Install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)

2. Install the Julia extension:
   - Open VS Code
   - Click the Extensions icon (or press `Cmd + Shift + X`)
   - Search for "Julia" and install the official Julia extension

3. Open the GPEC folder in VS Code:
   - Click File → Open Folder
   - Navigate to and select your GPEC directory

4. Open a terminal inside VS Code (Terminal → New Terminal) and run scripts directly:
   ```bash
   julia --project=. path/to/script.jl
   ```

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

You can also use GPEC programmatically in your own Julia scripts:

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

## Pre-commit Hooks (Optional Developer Tools)

The repository uses pre-commit hooks to maintain code quality.

### Why Use Pre-commit Hooks?

Pre-commit hooks automatically:
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

However, this is discouraged as it may introduce formatting inconsistencies.

## Development Tips

### Revise.jl

When iterating on code, use [Revise.jl](https://timholy.github.io/Revise.jl/stable/) to avoid full recompilation on every change. It tracks source file modifications and recompiles only the affected code.

Install Revise in your **global** Julia environment (not in this project's `Project.toml`):

```julia
# In the Julia REPL (no --project flag)
using Pkg
Pkg.add("Revise")
```

Then load it before `GeneralizedPerturbedEquilibrium` in any session:

```julia
using Revise
using GeneralizedPerturbedEquilibrium
```

For it to load automatically, add `using Revise` to your [Julia startup file](https://timholy.github.io/Revise.jl/stable/config/#Using-Revise-by-default):

```julia
# ~/.julia/config/startup.jl
try
    using Revise
catch e
    @warn "Could not load Revise" exception=e
end
```
