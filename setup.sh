#!/bin/bash

# This script sets up the local development environment.
# It should be run from the root of the repository.

set -e

echo "Starting Development Environment Setup"

echo "Installing dependencies with uv..."
uv sync
echo "[DONE] Dependencies installed."

echo "Installing pre-commit hooks..."
uv run pre-commit install --install-hooks
echo "[DONE] Pre-commit hooks installed."

echo "[SUCCESS] Your local development environment is ready."
