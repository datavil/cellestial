# Releasing

Run every check the same way before cutting a release:

```bash
./scripts/release-check.sh
```

The script fails fast on the first problem and runs, in order:

1. `poetry check` — validate `pyproject.toml`
2. `poetry run ruff check cellestial` — lint
3. `poetry run ruff format --check cellestial` — format check
4. `poetry run pytest` — tests
5. `poetry build` — build package dist

When it prints `all release checks passed`, the release is verified.
