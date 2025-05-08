# Reinforcement Learning core

![testing workflow](https://github.com/Eva-ortiz/RL_core/actions/workflows/pytest.yml/badge.svg)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Python library in Reinforcement Learning algorithm implementation for Gymnasium environments.
Based on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and my own thesis work.

Some related publications:

- Ortiz-Mansilla, E. & García-Esteban, J.J. & Bravo-Abad, J. & Cuevas, J.C. (2024). Deep reinforcement learning for radiative heat transfer optimization problems. Physical Review Applied. 22. [10.1103/PhysRevApplied.22.054071](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.22.054071).

## Structure

The repository is structured into the following directories:

- `/data`: data folder.
- `/img`: images folder.
- `/RLcore`: Python source code of the repository.
- `/tests`: Python code for testing via pytest.
- `/script`: `.sh` scripts.

Set of workflows via Github Actions already installed:

- `pre-commit`: run pre-commit hooks
- `pytest`: automatically discover and runs tests in `tests/`

Tools:

- [uv](https://docs.astral.sh/uv/): manage dependencies, Python versions and virtual environments
- [ruff](https://docs.astral.sh/ruff/): lint and format Python code
- [mypy](https://mypy.readthedocs.io/): check types
- [pytest](https://docs.pytest.org/en/): run unit tests
- [pre-commit](https://pre-commit.com/): manage pre-commit hooks
- [prettier](https://prettier.io/): format YAML and Markdown
- [codespell](https://github.com/codespell-project/codespell): check spelling in source code

## Installation

### Application

Install package and pinned dependencies with the [`uv`](https://docs.astral.sh/uv/) package manager:

1. Install `uv`. See instructions for Windows, Linux or MacOS [here](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone repository

3. Install package and dependencies in a virtual environment:

   ```{bash}
   uv sync
   ```

4. Run any command or Python script with `uv run`, for instance:

   ```{bash}
   uv run RLcore/main.py
   ```

   Alternatively, you can also activate the virtual env and run the scripts normally:

   ```{bash}
   source .venv/bin/activate
   ```

In order to remove any virtual env, just make sure is into the working directory (`ls -lha .`) and remove it:

```{bash}
rm -rf .venv
```

### Library

Install a specific version of the package with `pip` or `uv pip`:

```{bash}
pip install git+ssh://git@github.com:Eva-ortiz/RL_core.git@0.0.0
```

## Setup development environment (Unix)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and pre-commit hooks:

```{bash}
make install
```

`uv` will automatically create a virtual environment with the specified Python version in `.python-version` and install the dependencies from `uv.lock` (both standard and dev dependencies). It will also install the package in editable mode.

### Adding new dependencies

Add dependencies with:

```{bash}
uv add <PACKAGE>
```

Add dev dependencies with:

```{bash}
uv add --dev <PACKAGE>
```

Remove dependency with:

```{bash}
uv remove <PACKAGE>
```

Check dependencies with:

```{bash}
uv pip list
```

In all cases `uv` will automatically update the `uv.lock` file and sync the virtual environment. This can also be done manually with:

```{bash}
uv sync
```

For more info, please refer to [uv commands documentation](https://docs.astral.sh/uv/reference/cli/).

### Tools

#### Run pre-commit hooks

Hooks are run on modified files before any commit. To run them manually on all files use:

```{bash}
make hooks
```

#### Run linter and formatter

```{bash}
make ruff
```

#### Run tests

```{bash}
make test
```

#### Run type checker

```{bash}
make mypy
```

### Fonts

If employed fonts are not available in your PC, you can get them through conda-forge:

```{bash}
conda activate env_name_py3.9
conda install -c conda-forge -y mscorefonts
```

Or, alternatively, [you can install them in your PC](https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/49884009#49884009):

```{bash}
sudo apt install msttcorefonts -qq
rm ~/.cache/matplotlib -rf           # remove cache
```

This last method will install `msttcorefonts` in path `/usr/share/fonts/truetype/msttcorefonts`.

## Acknowledgements

### My thesis director

[Juan Carlos Cuevas Rodríguez](http://webs.ftmc.uam.es/juancarlos.cuevas/)

### Collaborating company

[Komorebi's team](https://komorebi.ai/es/)

## Additional info

### Personal info

If you find some problems or you have anything to discuss about the code, feel free to start a discussion or please reach me through my e-mail: [eva.ortizm@estudiante.uam.es](mailto:eva.ortizm@estudiante.uam.es)

### Template

[Template](https://github.com/Komorebi-AI/python-template) for Python libraries by Komorebi-AI. The associated development guide can be found [here](https://github.com/Komorebi-AI/docs/blob/main/python_dev.md).
