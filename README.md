
FixRABIES is an alternative confound correction model implementing ICA-FIX workflor for rodent fMRI datasets preprocessed with RABIES (and potentially other workflows). 

This is an opinonated pipeline, loosely based on [Zerbi et al. 2025](https://pubmed.ncbi.nlm.nih.gov/26296501/). 

## Package roadmap. 
- Dataset preparation for Fix
- Manual ICA labeling with GUI
- FIX classifier training 
- FIX classification
- classification review
- FIX denosing
- Dual-regression analysis (optional, else can use RABIES anlysis workflow)
- Seed-based analysis (optional, else can use RABIES analysis workflow)

## Supported spaces:
- DSURQE (mouse)
- AIBS (mouse)
- SIGMA (rat)

## Installation 
```{bash}
pip intall .
```

### GUI support (for "label" and "report" viewing)
```{bash}
pip install .[gui]
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
This provides Python 3.14, uv, and GCC.

### uv development
```{bash}
#create a virtual environement
uv venv --python 3.14 .venv

#get dependencies
uv sync --dev

#run the cli
uv run fixrabies -h 
```

## Workflows

### prepare
```{bash}
fixrabies prepare \
  -preprocess_dir /path/to/preprocess/dir \
  -fix_dir /path/to/fix/dir \
  -species mouseDSURQE
```

Tasks:
- Collect preprocessing outputs
- Extract motion parameters
- Run optional patch-denoising
- Build FIX directory structure

### label
```{bash}
fixrabies label \
  -fix_dir /path/to/fix/dir \
  -select_label 20
```

requires 
```{bash}
pip install FixRABIES[gui]
```

Calls a PySide graphical interface to label noise component for training. 

Graphical concept.   

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

### train
```{bash}
fixrabies train \
  -fix_dir /path/to/fix/dir \
  -fix_model my_model
```

This will create /path/to/fix/dir/my_model

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
will open a GUI with all classified components and the possibility to manually fix it. 

### apply
```{bash}
fixrabies apply \
  -fix_dir /path/to/fix/dir
```

will apply FIX denoising and other options and outputs files into a "RABIES-like" directory to be compatible with the RABIES analysis workflow

### analyze
```{bash}
fixrabies analyze \
  -fix_dir /path/to/fix/dir \
  -DR_template /path/to/atlas.nii.gz
```


