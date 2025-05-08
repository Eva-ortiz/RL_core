.ONESHELL:
.PHONY: install hooks hooks-update ruff test mypy build run debug push

SHELL=/bin/bash
GH_USER=GITHUB_USERNAME
GH_TOKEN_FILE=GITHUB_TOKEN_PATH

# Install uv, pre-commit hooks and dependencies
# Note that `uv run` has an implicit `uv sync`, since it will (if necessary):
# - Download an install Python
# - Create a virtual environment
# - Update `uv.lock`
# - Sync the virtual env, installing and removing dependencies as required
install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv run pre-commit install

hooks:
	uv run pre-commit run --all-files

hooks-update:
	uv run pre-commit autoupdate

ruff:
	uv run ruff format .
	uv run ruff check --fix --show-fixes .

test:
	uv run pytest

mypy:
	uv run mypy --install-types --non-interactive
