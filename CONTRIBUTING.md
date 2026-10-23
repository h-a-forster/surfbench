# Contributing

Bug reports, new models, diagnostics and reference fixtures are welcome.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --group dev
uv run surfbench --version
```

## Checks

```sh
uv run pytest              # fast suite; add --runslow for long checks
uv run ruff check .
uv run ruff format --check .
```

CI runs these on pull requests (Ubuntu with Python 3.11 and 3.13, Windows with Python 3.13).

Test rules:

- No network. Tests use synthetic data or committed fixtures, never market data.
- Use `tmp_path` for every file.
- Tests must pass on Windows: use `pathlib` and do not assume a POSIX shell.
- Fix every random seed. Tests must be deterministic.
- Mark anything over a few seconds with `@pytest.mark.slow`.

## Adding a model

1. Put the parametrisation, its constraints, the calibration method and a reference in
   `docs/DESIGN.md` first.
2. Implement the `SurfaceModel` protocol (`surfbench.models.base`) with the common
   objective; assert the constraints on every accepted fit.
3. Add an oracle test (closed form, published value or independent implementation) and a
   synthetic recovery test.

## Data

Never commit market data, extracted snapshots or fitted parameters (see `docs/DESIGN.md`
section 3). Results under `results/` hold aggregate statistics only.

## Code

- Python 3.11+, typed. Runtime dependencies: numpy and scipy. Open an issue before adding
  one.
- Match the surrounding style; ruff enforces the rest (line length 100).
- Long computations must go through `surfbench.checkpoint.UnitStore`, so they can resume
  after a crash.
