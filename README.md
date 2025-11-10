# Perturbed Equilibrium

This is a work in progress juliafication of the Generalized Perturbed Equilibrium Code suite https://github.com/PrincetonUniversity/GPEC.

## Documentation Pages

The WIP documentation can be found [here](https://openfusiontoolkit.github.io/JPEC/dev/).


## Developer Notes

### GIT Workflow

All developers need to use Vincent Driessen's [GitFlow](http://nvie.com/posts/a-successful-git-branching-model) workflow when editing the GPEC package. PLEASE READ THE ENTIRE POST. It is short, very thorough, and good for both experienced and new git users.

The highlights are,
  - There are two permanent branches: main (the post uses "master") and develop
  - The main branch is only updated for at release ready stages
  - New features should be developed in short-lived (days) branches coming off of and merging back to the develop branch.

Specific instructions are given in the link above as to exactly how to branch and merge these various branches. For example, the --no-ff option should be used when merging in order to keep branch histories. Just follow the examples and you wont go wrong!

#### Using github

Please see [this link](https://docs.google.com/document/d/1XAOTz1IV8ErZAAk-iSuEuddNOLB5XcoVZsAbPKRUUuA/edit?usp=sharing) for a guide on how to discuss your code development using github.

#### Commit messages

To assist with the process of compiling release notes, please confirm to the commit message format:
```
CODE - TAG - Detailed message
```
where CODE is EQUIL, DCON, VAC, etc. and TAGs are short descriptors of the type of commit. Example tags might be WIP (work in progress), MINOR, IMPROVEMENT, BUG FIX, NEW FEATURE, etc. Look through old commits for examples of what tags are commonly used.

Again, this is currently used for the by-hand compilation of release notes. The tags thus need to be human readable but are not strictly inforced to be within some limited set. The objective is to allow a lead developer to skim through commits and pick out only the key new features / bug fixes / improvements to note in the release while not having to read all the work-in-progress or minor changes.

### Julia Tips

#### Revise

When developing and recompiling code often, use Revise.jl to speed up the compile times by only recompiling what is impacted by the changed code. Integrate Revise.jl into your default Julia environment for use across projects, do not directly add it to the Project.toml of this particular project. To install Revise.jl in your global environment, open the Julia REPL using the `julia` command, enter the package manager by pressing `]`, and then `add Revise`:

Code

    $ julia
    julia> ]
    pkg> add Revise

Now, in the top of each Jupyter notebook, you can call `using Revise` at the top of any Jupyter notebook to speed up compile times as you develop and test. Better yet, set up your environement to [use Revise by default](https://timholy.github.io/Revise.jl/stable/config/#Using-Revise-by-default)