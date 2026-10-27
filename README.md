
FixRABIES is an alternative confound correction model implementing ICA-FIX workflow for rodent fMRI datasets preprocessed with RABIES (and potentially other preprocessing pipelines). 

This is an opinonated pipeline, loosely based on [Zerbi et al. 2025](https://pubmed.ncbi.nlm.nih.gov/26296501/). 

## Package roadmap. 
- Dataset preparation for Fix (done)
- Manual ICA labeling with GUI (work in progress)
- FIX classifier training (done)
- FIX classification
- classification review
- FIX denosing
- Dual-regression analysis (optional, else can use RABIES anlysis workflow)
- Seed-based analysis (optional, else can use RABIES analysis workflow)

## Planned supported spaces:
- DSURQE (mouse)
- AIBS (mouse)
- SIGMA (rat)

## Installation 
```{bash}
git clone https://github.com/grandjeanlab/FixRABIES.git
cd FixRABIES
pip intall .
```

### Patch denoissing support (for "prepare")
```{bash}
pip install .[patch]
```

## Development
### Nix development shell
```{bash}
nix develop . -c nu
```
This provides Python 3.14, uv, and GCC for Nix-based systems.

### uv development
```{bash}
#create a virtual environement
uv venv --python 3.14 .venv

#get dependencies
uv sync --dev --all-extras

#run the cli
uv run fixrabies -h 
```

## Output directory structure
FixRABIES directory sturcture
```{text}
├── data
│   ├── sub-...
│   └── sub-...
├── models
│   ├── custom_model.pyfix_model
│   └── label_select.csv
├── confound_correction
│   ├── sub-...
│   └── sub-...
├── analysis
│   ├── sub-...
│   └── sub-...
├── workflows
│   ├── prepare
│   └── ...
└── log
    ├── prepare.log
    └── ...
```

Content of a FIX directory for a single subject 
```{text}
├── bg.nii.gz 
├── design.fsf
├── filtered_func_data_clean.nii.gz
├── filtered_func_data_clean_smoothed.nii.gz
├── filtered_func_data.ica
│   ├── ...
│   └── report
│       ├── f1.png
│       ├── t1.png
│       ├── IC_1_thresh.png
│       ├── ... 
├── filtered_func_data.nii.gz
├── fix
│   ├── ...
├── fix4melview_{custom_model}_thr{thr}.txt
├── hand_labels_noise.txt
├── mc
│   └── prefiltered_func_data_mcf.par
├── reg
│   ├── highres2example_func.mat
│   ├── highres.nii.gz
│   └── highres_pveseg.nii.gz
└── ...
```

## Workflows

### prepare
```{bash}
fixrabies prepare \
  -preprocess_dir /path/to/preprocess/dir \
  -fix_dir /path/to/fix/dir \
  -species mouseDSURQE
```

FIX preparation using RABIES preporcessing as inputs. 

```{text}
filtered_func_data.nii.gz <- {preprocess_dir}/bold_datasink/commonspace_old/_scan_info.../_run_.../sub-...nii.gz
OR 
filtered_func_data.nii.gz <- patch-denoise outputs

bg.nii.gz <- {preprocess_dir}/bold_datasink/commonspace_resampled_template/commonspace_template.nii.gz

mask.nii.gz <- {preprocess_dir}/bold_datasink/commonspace_mask/commonspace_template_brain_mask_resampled.nii.gz

design.fsf <- generated

fd.csv <- {preprocess_dir}/motion_datasink/FD_csv/_scan_info.../_run_.../sub-...csv

mc/prefiltered_func_data_mcf.par <- {preprocess_dir}/motion_datasink/motion_param_csv/_scan_info.../_run_.../sub-...csv

reg/highres.nii.gz <- {preprocess_dir}/bold_datasink/commonspace_resampled_template/commonspace_template.nii.gz

reg/highres_pveseg.nii.gz <- {FixRABIES}/template/{species}/highres_pvseg.nii.gz

reg/highres2example_func.mat <- generated 

```

Tasks:
- Collect preprocessing outputs
- Extract motion parameters
- Run optional patch-denoising
- Build FIX directory structure
- Run melodic

### label
```{bash}
fixrabies label \
  -fix_dir /path/to/fix/dir \
  -select_label 20
```

Calls a PySide graphical interface to label noise component for training. 

Graphical concept.   
```{text}
+------------------------------------------------+   
| Subject - Component 17                         |   
+----------------------+-------------------------+   
| Spatial Map          | Power Spectrum          |   
|                      |                         |   
|                      |                         |   
+----------------------+-------------------------+   
| Time                                           |   
+------------------------------------------------+   
| [Artifact] [Signal] [Unsure]                   |   
+------------------------------------------------+   
| Prev | Next | Save                             |   
+------------------------------------------------+   
```

### train
```{bash}
fixrabies train \
  -fix_dir /path/to/fix/dir \
  -fix_model my_model
```

This will create /path/to/fix/dir/models/my_model.pyfix_model

### classify
```{bash}
fixrabies classify \
  -fix_dir /path/to/fix/dir \
  -fix_model my_model
```

I plan to implement re-usable models in pyfix. currently only supports models trained in the ***train*** workflow

### review
```{bash}
fixrabies review \
  -fix_dir /path/to/fix/dir
```
will open a GUI with all classified components and the possibility to manually correct them. same panel as in the train workflow 

### cleanup
```{bash}
fixrabies cleanup \
  -fix_dir /path/to/fix/dir
```

will apply FIX denoising and other options and outputs files into a "RABIES-like" directory to be compatible with the RABIES analysis workflow

### analyze
```{bash}
fixrabies analyze \
  -fix_dir /path/to/fix/dir \
  -DR_template /path/to/atlas.nii.gz
```


