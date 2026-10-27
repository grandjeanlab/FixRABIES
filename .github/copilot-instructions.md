# FixRABIES Copilot Instructions

## Environment and validation

- Use `uv` for all project commands. The supported Linux x86_64 environment
  and Python version are declared in `pyproject.toml`.
- Install development dependencies with:

  ```sh
  uv sync --dev --all-extras
  ```

- Run all tests:

  ```sh
  uv run pytest -q
  ```

- Run one test or test function:

  ```sh
  uv run pytest -q tests/test_validate_inputs.py
  uv run pytest -q tests/test_validate_inputs.py::test_validate_training_inputs_and_run_training
  ```

- Lint and format-check Python files:

  ```sh
  uv run ruff check src tests
  uv run black --check src tests
  ```

- Inspect the command-line interface with `uv run fixrabies -h` or
  `uv run fixrabies <command> -h`.

## Architecture

- `fixrabies.cli` is the package entry point. It defines shared `-fix_dir`,
  `-nipype`, and logging options, configures logs in `<fix_dir>/logs`, then
  lazily dispatches each subcommand to its `run_*` function.
- The working directory is the cross-command contract:
  - `prepare` reads RABIES outputs and writes one ICA-FIX-compatible scan
    directory per scan under `<fix_dir>/data`.
  - `label` validates those scan directories, selects training scans, and
    records their paths in `<fix_dir>/model/training_set.txt`.
  - `train` validates that list and creates
    `<fix_dir>/model/<fix_model>.pyfix_model`.
  - `classify` validates the prepared scans and local model, then iterates the
    scan directories through `fix -c`.
- Command modules (`prepare.py`, `label.py`, `train.py`) perform input
  validation and invoke workflows or interfaces. Keep filesystem validation in
  `interfaces/validate_inputs.py` so it runs before Nipype.
- Graph-based work belongs in `workflows/`. `prepare_wf.py` uses Nipype nodes
  and iterables to build scan-specific FIX data, run MELODIC, and extract
  features. `label_wf.py` handles random scan selection and label metadata.
  `classify_wf.py` uses an `IdentityInterface` iterable to run classification
  for every prepared scan.
  Preserve the CLI pattern of copying `args.nipype`, popping `plugin`, and
  passing the remainder as `plugin_args`.
- `nodes/` contains functions wrapped by Nipype `Function` nodes. Functions
  executed in these nodes must import dependencies locally because Nipype
  serializes their source for execution.
- `interfaces/pyfix.py` is the local Nipype interface for the current PyFIX
  training CLI. Do not use `nipype.interfaces.fsl.fix.Training`: it targets
  the legacy `.RData` model workflow. The local interface runs
  `fix -t -s <species> <model-stem> <scan>...`.

## Project conventions

- Treat scan lists as absolute paths. Training entries must resolve to direct
  child directories of `<fix_dir>/data`; validate their `fix/features.csv` and
  `hand_labels_noise.txt` before invoking PyFIX.
- A selected-label percentage of `0` still means one scan. For any other
  percentage, derive the expected training count with `ceil(scan_count *
  percentage / 100)`. Reuse `training_set.txt` only when its non-empty line
  count matches that expected count.
- Model names are names, not paths. Training writes the model stem beneath
  `<fix_dir>/model`, rejects names already shipped in PyFIX resources, and
  verifies `<fix_model>.pyfix_model` was produced after training.
- Use `importlib.resources` for package templates and PyFIX resource models;
  do not rely on source-tree or virtual-environment paths.
- `tests/test_validate_inputs.py` intentionally exercises the existing
  `~/test` fixture, including its prepared data and training set. Keep those
  tests side-effect-free: mock external training and remove any temporary
  model file created for output-validation checks.
