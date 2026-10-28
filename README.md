# FixRABIES

FixRABIES prepares rodent fMRI data for ICA-FIX, supports interactive
component labelling and model training, applies FIX cleanup, and provides
dual-regression and seed-based connectivity analyses. It currently accepts
RABIES preprocessing outputs and supports the `mouseDSURQE`, `mouseAIBS`, and
`ratSIGMA` template spaces.

## Installation

For development or a local checkout, use `uv`:

```bash
git clone https://github.com/grandjeanlab/FixRABIES.git
cd FixRABIES
uv sync --dev --all-extras
uv run fixrabies -h
```

Install the optional patch-denoising dependency with:

```bash
pip install ".[patch]"
```

For a regular installation:

```bash
pip install .
```

Use the command help to see all available options and defaults:

```bash
fixrabies -h
fixrabies cleanup -h
fixrabies analyze -h
```

## Workflow

Each command uses the same working directory, passed with `-fix_dir`. Run the
commands in this order:

1. `prepare` creates one ICA-FIX-compatible scan directory per preprocessed
   scan.
2. `label` selects scans for manual component labelling and writes the
   training list.
3. `train` creates a `.pyfix_model` from the labelled scans.
4. `classify` applies that model to every prepared scan.
5. `review` opens the classification review interface.
6. `cleanup` applies the reviewed/classified component list and optionally
   smooths and censors the denoised data.
7. `analyze` runs optional dual-regression and/or seed-based analyses on the
   cleaned, smoothed data.

All commands accept `-nipype` to configure execution. For example:

```bash
fixrabies prepare \
  -preprocess_dir /path/to/rabies \
  -fix_dir /path/to/fixrabies \
  -species mouseDSURQE \
  -nipype '{"plugin": "MultiProc", "n_procs": 4}'
```

## Command guide

### Prepare scans

```bash
fixrabies prepare \
  -preprocess_dir /path/to/rabies \
  -fix_dir /path/to/fixrabies \
  -species mouseDSURQE
```

`prepare` reads RABIES outputs, creates `data/<scan>/`, runs MELODIC, and
writes the files needed by FIX. Use `-patch_denoise '{"apply": true, ...}'`
when the optional patch-denoising dependency is installed.

### Label training scans

```bash
fixrabies label \
  -fix_dir /path/to/fixrabies \
  -select_label 20
```

`-select_label` is the percentage of prepared scans to label. A value of `0`
selects one scan; `100` selects all scans. The GUI writes
`hand_labels_noise.txt` in each selected scan and records absolute scan paths
in `model/training_set.txt`. If that file already has the expected number of
non-empty entries, it is reused.

### Train a model

```bash
fixrabies train \
  -fix_dir /path/to/fixrabies \
  -fix_model mouse_awake_v1 \
  -species mouseDSURQE
```

Training requires a non-empty `model/training_set.txt`; every listed scan must
be a direct child of `data/` and contain `fix/features.csv` and
`hand_labels_noise.txt`. The trained model is written as
`model/mouse_awake_v1.pyfix_model`.

### Classify and review components

```bash
fixrabies classify \
  -fix_dir /path/to/fixrabies \
  -fix_model mouse_awake_v1 \
  -threshold 20

fixrabies review \
  -fix_dir /path/to/fixrabies \
  -fix_model mouse_awake_v1 \
  -threshold 20
```

Classification writes
`data/<scan>/fix4melview_mouse_awake_v1_thr20.txt`. Use the same model name
and threshold for `review` and `cleanup`.

### Apply FIX cleanup

```bash
fixrabies cleanup \
  -fix_dir /path/to/fixrabies \
  -fix_model mouse_awake_v1 \
  -threshold 20 \
  -smooth 0.3 \
  -FD_threshold 0.075 \
  -DVARS_threshold 1.5
```

`cleanup` applies FIX to every classified scan, then writes
`filtered_func_data_clean_smoothed.nii.gz`. It computes DVARS, reads the
`MeanFD` column from `fd.csv`, and writes `exclude.csv` with the metrics and
their selections. Set either threshold to `0` to disable that censoring
criterion. Optional FIX flags are:

```bash
-motionhp              # add motion-parameter regression
-aggressive            # use full-variance cleanup
-highpass SECONDS      # add FIX high-pass filtering
```

### Run connectivity analysis

Run dual regression with the per-scan ICA maps:

```bash
fixrabies analyze \
  -fix_dir /path/to/fixrabies \
  --dice_threshold 90 \
  --extract_threshold 15z
```

Provide `-DR_template` to resample and use a custom ICA atlas:

```bash
fixrabies analyze \
  -fix_dir /path/to/fixrabies \
  -DR_template /path/to/atlas_4d.nii.gz \
  --norm
```

`--extract_threshold` accepts a percentile such as `90p` (0--100) or a
z-score such as `7.5z`. Existing dual-regression stage-2 maps are reused, but
the outcome report is still generated.

Add seed-based analyses with a JSON list of seed masks:

```bash
fixrabies analyze \
  -fix_dir /path/to/fixrabies \
  -seed '{
    "apply": true,
    "seeds": [
      "/path/to/acc.nii.gz",
      "/path/to/thalamus.nii.gz"
    ]
  }'
```

## Output structure

The working directory is the contract between commands:

```text
<fix_dir>/
├── data/
│   └── <scan>/
│       ├── filtered_func_data.nii.gz
│       ├── filtered_func_data.ica/
│       ├── mask.nii.gz
│       ├── fd.csv
│       ├── fix/features.csv
│       ├── hand_labels_noise.txt
│       ├── fix4melview_<model>_thr<threshold>.txt
│       ├── filtered_func_data_clean.nii.gz
│       ├── filtered_func_data_clean_smoothed.nii.gz
│       └── exclude.csv
├── model/
│   ├── training_set.txt
│   └── <model>.pyfix_model
├── results/
│   └── <scan>/
│       ├── dr/
│       │   ├── dr_stage2_subject00000.nii.gz
│       │   ├── dr_stage2_subject00000_Z.nii.gz
│       │   └── outcomes.csv
│       └── seed/
│           ├── <seed>.txt
│           ├── <seed>.nii.gz
│           ├── <seed>_Z.nii.gz
│           └── corr.txt
└── logs/
    ├── fixrabies.log
    └── <command>.log
```

`results/<scan>/dr/outcomes.csv` has one row per ICA component and contains
the masked Dice overlap (`beta_dice`, `z_dice`) and extracted values
(`beta_value`, `z_value`) for dual-regression beta and Z maps.
`results/<scan>/seed/corr.txt` is a tab-separated correlation matrix whose
row and column names are the seed basenames.
