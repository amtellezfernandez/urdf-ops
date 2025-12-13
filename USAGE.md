# LeRobot Dataset Tool 🤖

**Simple, professional CLI for managing LeRobot datasets**

## Installation

### Option 1: Using an existing virtual environment

If you already have a virtual environment activated (e.g., conda, venv):

```bash
# Install lerobot in the current environment
uv pip install lerobot

# Or install from requirements.txt
uv pip install -r requirements.txt
```

### Option 2: Create a new virtual environment with uv

```bash
# Create a virtual environment
uv venv

# Activate it (Linux/Mac)
source .venv/bin/activate

# Or on Windows
.venv\Scripts\activate

# Install dependencies
uv pip install lerobot
```

### Option 3: Using conda/mamba

```bash
# Create and activate a conda environment
conda create -n lerobot python=3.10
conda activate lerobot

# Install with uv
uv pip install lerobot
```

## Quick Start

```bash
# Get info about any dataset
python src/cli.py info --dataset lerobot/svla_so100_stacking

# Validate dataset quality
python src/cli.py validate --dataset lerobot/svla_so100_stacking

# Split dataset
python src/cli.py split --dataset lerobot/svla_so100_stacking --ratio 0.8 0.2 --names train val --output my_split

# Delete episodes
python src/cli.py delete --dataset lerobot/svla_so100_stacking --episodes 0,2,5 --output filtered

# Add custom feature
python src/cli.py add-feature --dataset lerobot/svla_so100_stacking --name reward --output with_reward

# Merge datasets (with automatic validation!)
python src/cli.py merge --datasets my_split/train,my_split/val --output merged

# Get episode details
python src/cli.py episode --dataset lerobot/svla_so100_stacking --episode 0

# Visualize dataset with Rerun
python src/cli.py visualize --dataset lerobot/svla_so100_stacking
```

## 📋 Commands

### ✅ `validate` - Validate Dataset Quality

Comprehensive validation checks for LeRobot datasets to ensure data quality and catch common issues.

```bash
# Validate a dataset
python src/cli.py validate --dataset lerobot/svla_so100_stacking

# Validate with detailed output
python src/cli.py validate --dataset lerobot/svla_so100_stacking --verbose

# Skip specific validation checks
python src/cli.py validate --dataset lerobot/svla_so100_stacking --skip-action-norm
```

**Validation Checks:**

1. **Semantic Validation**
   - Verifies required features exist (observations, actions)
   - Checks feature naming conventions
   - Validates data types and metadata

2. **Action Normalization**
   - Checks action bounds and ranges
   - Detects NaN or infinite values
   - Validates proper normalization ([-1, 1] or [0, 1])
   - Identifies low-variance dimensions

3. **Observation Modality Consistency**
   - Ensures consistent camera/image shapes
   - Validates state observation dimensions
   - Checks image value ranges
   - Detects missing or corrupted data

4. **Frame-Level Alignment**
   - Verifies episode boundaries are consistent
   - Checks frame counts match metadata
   - Validates timestamp monotonicity (if available)
   - Detects gaps or overlaps between episodes

5. **Off-by-One Error Detection**
   - Checks action-observation temporal alignment
   - Validates episode lengths
   - Detects boundary issues
   - Verifies next.observation consistency

**Options:**
- `--dataset` Dataset repo ID or local path
- `--stream` Stream dataset from Hub (use with caution for large datasets)
- `--verbose, -v` Show detailed validation results with all checks
- `--skip-semantic` Skip semantic validation
- `--skip-action-norm` Skip action normalization checks
- `--skip-obs-consistency` Skip observation consistency checks
- `--skip-frame-alignment` Skip frame alignment checks
- `--skip-off-by-one` Skip off-by-one error detection

**Output:**
```
Validating: lerobot/svla_so100_stacking

Dataset Info:
  Episodes: 206
  Frames:   25650

Validation Results:
============================================================
Total Checks:  12
Passed:        10
Errors:        0
Warnings:      2

WARNINGS:
  ⚠ Actions may not be properly normalized
    min: [-2.1, -1.8]
    max: [2.3, 2.0]
    expected_range: [-1, 1] or [0, 1]

✓ Validation passed with warnings
```

**When to Use:**
- Before training models on a new dataset
- After merging or modifying datasets
- When debugging data pipeline issues
- To verify dataset quality for publication

---

### 🔍 `info` - Get Dataset Information

```bash
# Get info about a dataset (downloads if needed)
python src/cli.py info --dataset lerobot/svla_so100_stacking

# Stream dataset directly from Hub (no local download)
python src/cli.py info --dataset lerobot/svla_so100_stacking --stream
```

**Options:**
- `--dataset` Dataset repo ID or local path
- `--stream` Stream dataset directly from Hub without local copies (useful for large datasets)

**Output:**
```
Dataset: lerobot/svla_so100_stacking

ℹ Loading from: HuggingFace Hub

Episodes:            206
Frames:              25650
FPS:                 10
Robot:               unknown

Features:            11
  • observation.image              (96, 96, 3)     video
  • observation.state              (2,)            float32
  • action                         (2,)            float32
  ...

✓ Info retrieved successfully
```

**With streaming:**
```
Dataset: lerobot/svla_so100_stacking

ℹ Streaming from: HuggingFace Hub (no local download)

Episodes:            100
Frames:              5000
...
```

---

### `split` - Split Dataset

```bash
python src/cli.py split \
    --dataset lerobot/svla_so100_stacking \
    --ratio 0.8 0.2 \
    --names train val \
    --output my_split
```

**Options:**
- `--dataset` Dataset to split
- `--ratio` Space-separated ratios (must sum to 1.0)
- `--names` Space-separated names for each split
- `--output` Output folder name (default: `split`)

**Output:**
```
Splitting: lerobot/svla_so100_stacking
ℹ Split ratios: {'train': 0.8, 'val': 0.2}
ℹ Output: datasets/my_split/
ℹ Processing... (this may take a moment)

✓ train: 164 episodes → datasets/my_split/train/
✓ val: 42 episodes → datasets/my_split/val/
```

---

### `delete` - Delete Episodes

```bash
python src/cli.py delete \
    --dataset lerobot/svla_so100_stacking \
    --episodes 0,2,5 \
    --output filtered
```

**Options:**
- `--dataset` Source dataset
- `--episodes` Comma-separated episode indices (e.g., `0,2,5`)
- `--output` Output folder name (default: `filtered`)

**Output:**
```
Deleting episodes: lerobot/svla_so100_stacking
ℹ Deleting 3 episodes: [0, 2, 5]
ℹ Output: datasets/filtered/
ℹ Processing... (this may take a moment)

✓ Deleted 3 episodes, 203 remaining
✓ Saved to: datasets/filtered/
```

---

### `add-feature` - Add Feature

```bash
python src/cli.py add-feature \
    --dataset lerobot/svla_so100_stacking \
    --name reward \
    --type reward \
    --output with_reward
```

**Options:**
- `--dataset` Source dataset
- `--name` Feature name
- `--type` Feature type: `reward`, `success`, or `custom` (default: `custom`)
- `--output` Output folder name (default: `with_feature`)

**Output:**
```
Adding feature: lerobot/svla_so100_stacking
ℹ Feature name: reward
ℹ Feature type: reward
ℹ Output: datasets/with_reward/
ℹ Processing... (this may take a moment)

✓ Added feature 'reward'
✓ Total features: 12
✓ Saved to: datasets/with_reward/
```

---

### `merge` - Merge Datasets (with Validation!)

**Merge remote datasets:**
```bash
python src/cli.py merge \
    --datasets lerobot/svla_so100_stacking_train,lerobot/svla_so100_stacking_val \
    --output merged
```

**Merge local datasets:**
```bash
python src/cli.py merge \
    --datasets my_split/train,my_split/val \
    --output merged
```

**Options:**
- `--datasets` Comma-separated dataset repo IDs (remote or local)
- `--output` Output folder name (default: `merged`)
- `--force` Skip confirmation prompt

**Output:**
```
Merging Datasets
ℹ Datasets to merge: 2
  1. lerobot/svla_so100_stacking_train
  2. lerobot/svla_so100_stacking_val

ℹ Comparing datasets...
✓ All datasets compatible!

Features:            11
Total episodes:      206
Total frames:        25650

Metadata (may differ):
  • lerobot/svla_so100_stacking_train
    FPS: 10, Robot: unknown, Episodes: 164
  • lerobot/svla_so100_stacking_val
    FPS: 10, Robot: unknown, Episodes: 42

⚠  Continue with merge? [y/N]: y
ℹ Merging... (this may take a while)

✓ Merged 2 datasets
✓ Result: 206 episodes, 25650 frames
✓ Saved to: datasets/merged/
```

**If datasets are incompatible:**
```
✗ Datasets have incompatible features:
  ✗ dataset2 missing: reward
  ! dataset2 has extra: custom_feature
  ✗ dataset2.action shape mismatch: (2,) vs (3,)

✗ Cannot merge: Features must be identical
ℹ Tip: Use 'add-feature' or 'remove-feature' to align datasets first
```

---

### `episode` - Get Episode Info

```bash
# Get episode info (downloads if needed)
python src/cli.py episode --dataset lerobot/svla_so100_stacking --episode 0

# Stream episode info directly from Hub
python src/cli.py episode --dataset lerobot/svla_so100_stacking --episode 0 --stream
```

**Options:**
- `--dataset` Dataset repo ID or local path
- `--episode` Episode index (0-based)
- `--stream` Stream dataset directly from Hub without local copies

**Output:**
```
Episode 0: lerobot/svla_so100_stacking

Episode Index:       0
Length:              161 frames
Frame Range:         0 → 161
Tasks:               ['Push the T-shaped block onto the T-shaped target.']

✓ Episode info retrieved
```

---

### 🎨 `visualize` - Visualize Dataset with Rerun

Visualize LeRobot datasets using [Rerun](https://rerun.io), a powerful visualization tool for robotics data.

**Prerequisites:**
```bash
pip install rerun-sdk
```

**Usage:**

The command automatically uses lerobot's built-in visualization script if available, otherwise falls back to Rerun CLI.

```bash
# Visualize a dataset from HuggingFace (will download if needed)
python src/cli.py visualize --dataset lerobot/svla_so100_stacking

# Visualize a specific episode
python src/cli.py visualize --dataset lerobot/svla_so100_stacking --episode-index 0

# Visualize a local dataset (compatible with lerobot API)
python src/cli.py visualize --repo-id lerobot/svla_so100_stacking --root /path/to/dataset --mode local

# Visualize using a local path
python src/cli.py visualize --dataset /path/to/dataset

# Visualize dataset in datasets/ folder
python src/cli.py visualize --dataset datasets/my_dataset
```

**Options:**
- `--dataset` or `--repo-id` Dataset repo ID, local path, or path in `datasets/` folder
- `--root` Root directory for local datasets (compatible with lerobot's API)
- `--mode` Set to `"local"` for local datasets (compatible with lerobot's API)
- `--episode-index` Episode index to visualize (0-based, compatible with lerobot's API)
- `--stream` Stream dataset (note: visualization requires local files, so this will fail)

**Compatibility:**

This command is compatible with lerobot's visualization API:
```bash
# These commands are equivalent:
python -m lerobot.scripts.visualize_dataset --repo-id lerobot/svla_so100_stacking --episode-index 0
python src/cli.py visualize --dataset lerobot/svla_so100_stacking --episode-index 0

# For local datasets:
python -m lerobot.scripts.visualize_dataset --repo-id lerobot/svla_so100_stacking --root /path/to/dataset --mode local --episode-index 0
python src/cli.py visualize --repo-id lerobot/svla_so100_stacking --root /path/to/dataset --mode local --episode-index 0
```

**Output:**
```
Visualizing: lerobot/svla_so100_stacking

ℹ Dataset not found locally. Downloading from HuggingFace Hub...
  Repository: lerobot/svla_so100_stacking
  Downloading to: datasets/svla_so100_stacking

✓ Dataset downloaded to: datasets/svla_so100_stacking

✓ Opening Rerun viewer for: datasets/svla_so100_stacking
ℹ Close the viewer window or press Ctrl+C to exit
```

**Notes:**
- Rerun will open a viewer window showing the dataset
- The dataset must be downloaded locally (streaming not supported for visualization)
- If the dataset is not found locally, it will be automatically downloaded to `datasets/` folder
- For more information about Rerun, visit: https://rerun.io/examples/integrations/lerobot_loader

---

## Examples

### Complete Workflow

```bash
# 1. Check original dataset
python src/cli.py info --dataset lerobot/svla_so100_stacking

# 2. Validate dataset quality
python src/cli.py validate --dataset lerobot/svla_so100_stacking --verbose

# 3. Split into train/val/test (70/20/10)
python src/cli.py split \
    --dataset lerobot/svla_so100_stacking \
    --ratio 0.7 0.2 0.1 \
    --names train val test \
    --output production

# 4. Validate the split datasets
python src/cli.py validate --dataset production/train
python src/cli.py validate --dataset production/val

# 5. Check the split results (local datasets)
python src/cli.py info --dataset production/train
python src/cli.py info --dataset production/val

# 6. Add reward to training set
python src/cli.py add-feature \
    --dataset production/train \
    --name reward \
    --type reward \
    --output train_with_reward

# 7. Validate after adding features
python src/cli.py validate --dataset train_with_reward

# 8. Merge train and val back together (local datasets)
python src/cli.py merge \
    --datasets production/train,production/val \
    --output train_val_merged \
    --force
```

### Data Quality Workflow

```bash
# 1. Download and validate a dataset
python src/cli.py info --dataset lerobot/pusht
python src/cli.py validate --dataset lerobot/pusht --verbose

# 2. Check for specific issues
python src/cli.py validate --dataset lerobot/pusht \
    --skip-semantic \
    --skip-obs-consistency

# 3. Validate before training
python src/cli.py validate --dataset datasets/my_dataset
```

### Switch Datasets Easily

```bash
# Work with any LeRobot dataset
python src/cli.py info --dataset lerobot/svla_so100_stacking

python src/cli.py split \
    --dataset lerobot/svla_so100_stacking \
    --ratio 0.8 0.2 \
    --names train val \
    --output aloha_split
```

---

## Output Structure

All results saved to `datasets/` folder:

```
datasets/
├── my_split/
│   ├── train/          ← 80% of data
│   │   ├── data/
│   │   ├── meta/
│   │   └── videos/
│   └── val/            ← 20% of data
│       ├── data/
│       ├── meta/
│       └── videos/
├── filtered/           ← After deleting episodes
│   ├── data/
│   ├── meta/
│   └── videos/
└── merged/             ← After merging datasets
    ├── data/
    ├── meta/
    └── videos/
```

---

## Python API (Advanced)

For programmatic access:

```python
from src import load_dataset

# Load dataset
dataset = load_dataset("lerobot/svla_so100_stacking")

# Get info
info = dataset.info()
print(f"Episodes: {info['total_episodes']}")

# Validate dataset
report = dataset.validate()
print(report.summary())

# Check validation results
if report.passed:
    print("Dataset is valid!")
else:
    print(f"Validation failed with {len(report.errors)} errors")
    for error in report.errors:
        print(f"  - {error.message}")

# Split
splits = dataset.split({"train": 0.8, "val": 0.2})

# Add features
import numpy as np
rewards = np.random.randn(info["total_frames"]).astype(np.float32)
dataset_with_reward = dataset.add_features({
    "reward": (rewards, {"dtype": "float32", "shape": (1,), "names": None})
})
```

For more Python API examples, see the function docstrings in [src/dataset_wrapper.py](src/dataset_wrapper.py).

---

## Tips

1. **Dataset Names**: Use full repo IDs (e.g., `lerobot/svla_so100_stacking`) or local names (e.g., `my_split/train`)
2. **Local Datasets**: All commands automatically detect datasets in `datasets/` folder
3. **Episode Lists**: No spaces in comma-separated lists (`0,2,5` not `0, 2, 5`)
4. **Ratios**: Must sum to 1.0 (e.g., `0.7 0.2 0.1`)
5. **Outputs**: Automatically saved to `datasets/` folder
6. **Merge Validation**: Tool automatically checks feature compatibility
7. **Metadata**: FPS/robot type can differ when merging (features must match)
8. **Progress Bars**: Install `tqdm` for visual progress: `uv pip install tqdm` (optional)
9. **Data Quality**: Run `validate` before training to catch data issues early
10. **Validation Checks**: Use `--verbose` flag to see detailed validation results and statistics
11. **Custom Validation**: Skip specific checks with `--skip-*` flags if you know your data is valid

---

## Troubleshooting

### ModuleNotFoundError: No module named 'lerobot'

If you get this error even after running `uv pip install lerobot`, it usually means:

1. **Stale installation pointing to deleted vendor directory**: If you previously had lerobot installed from the vendor directory, `pip` may still have a record pointing to the now-deleted location.

   **Solution**: Uninstall and reinstall:
   ```bash
   # Uninstall the old installation
   python -m pip uninstall lerobot -y
   
   # Reinstall properly
   uv pip install lerobot
   
   # Verify it works
   python -c "import lerobot; print(lerobot.__file__)"
   ```

2. **Different Python interpreters**: The Python running your script is different from where `uv` installed lerobot.

   **Solution**: Check which Python you're using:
   ```bash
   # Check which Python is being used
   python --version
   which python  # or: where python (Windows)
   
   # Check if lerobot is installed for this Python
   python -c "import lerobot; print(lerobot.__file__)"
   
   # If it fails, install lerobot for this specific Python
   python -m pip install lerobot
   # Or with uv:
   uv pip install lerobot
   ```

3. **Virtual environment not activated**: You installed lerobot in one environment but are running the script in another.

   **Solution**: Make sure your virtual environment is activated:
   ```bash
   # For conda
   conda activate candy  # or your environment name
   
   # For venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

4. **Run the diagnostic script**:
   ```bash
   python check_env.py
   ```
   This will show you which Python is being used and whether lerobot is installed.

### Dataset Already Exists

```bash
# Clean up first
rm -rf datasets/my_split
python src/cli.py split --dataset lerobot/svla_so100_stacking --ratio 0.8 0.2 --names train val --output my_split
```

### Dataset Not Found

If you get "Repository Not Found" for a local dataset, check the path includes the parent folder:

```bash
# Wrong - just the split name
python src/cli.py merge --datasets train,val --output merged

# Right - include parent folder
python src/cli.py merge --datasets my_split/train,my_split/val --output merged

# Verify datasets exist
ls datasets/
```

### Merge Fails - Incompatible Features

Use the tool's suggestions:

```bash
# Add missing features
python src/cli.py add-feature --dataset dataset2 --name reward --type reward --output dataset2_fixed

# Or remove extra features
python src/cli.py remove-feature --dataset dataset2 --features extra_feature --output dataset2_fixed

# Then merge
python src/cli.py merge --datasets dataset1,dataset2_fixed --output merged
```

---

## Icons Reference

- ✓ Success
- ℹ Information
- ✗ Error
- ⚠ Warning
- ! Notice (feature difference)

---

## Command Summary

| Command | What It Does | Example |
|---------|--------------|---------|
| `info` | Show dataset details | `info --dataset lerobot/svla_so100_stacking` |
| `validate` | Validate dataset quality | `validate --dataset lerobot/svla_so100_stacking --verbose` |
| `split` | Split into train/val/test | `split --dataset lerobot/svla_so100_stacking --ratio 0.8 0.2 --names train val` |
| `delete` | Remove episodes | `delete --dataset lerobot/svla_so100_stacking --episodes 0,2,5` |
| `add-feature` | Add new feature | `add-feature --dataset lerobot/svla_so100_stacking --name reward` |
| `merge` | Combine datasets | `merge --datasets my_split/train,my_split/val --output merged` |
| `episode` | Episode details | `episode --dataset lerobot/svla_so100_stacking --episode 0` |
| `visualize` | Visualize with Rerun | `visualize --dataset lerobot/svla_so100_stacking` |

---

