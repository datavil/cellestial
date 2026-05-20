#!/usr/bin/env bash
# Run the release verification checklist. Fails fast on the first failing check.
# Usage: ./scripts/release-check.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> poetry check"
poetry check

echo "==> ruff check"
poetry run ruff check cellestial

echo "==> ruff format --check"
poetry run ruff format --check cellestial

echo "==> pytest"
poetry run pytest

echo "==> build"
poetry build

echo "==> wheel install/import smoke test"
wheel="$(ls -t dist/*.whl | head -n 1)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
python -m venv "$tmpdir/venv"
"$tmpdir/venv/bin/pip" install --quiet "$wheel"
"$tmpdir/venv/bin/python" -c "import cellestial; print('cellestial', cellestial.__version__)"

echo "==> all release checks passed"
