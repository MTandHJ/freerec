r"""Freerec Skills Module.

Provides skill-based tutorials and information for AI assistants
to understand freerec's internal mechanisms.
"""

SKILLS = {
    "make": {
        "name": "make",
        "description": "Tutorial for creating freerec-compatible datasets",
        "content": r"""## MAKE SKILL - Dataset Processing Tutorial

### PURPOSE
Converts raw interaction data into freerec's standard format for training recommendation models.

### DATA FORMAT REQUIREMENTS

#### Input Structure (RecBole-style atomic files):
```
data/
└── DatasetName/
    ├── DatasetName.inter  # Required: User-Item interactions
    ├── DatasetName.user   # Optional: User features
    └── DatasetName.item   # Optional: Item features
```

#### Interaction File (.inter) Format:
```
USER	ITEM	RATING	TIMESTAMP
1	1193	5	978300760
1	661	3	978302109
```

Column names are configurable via CLI arguments.

### COMMAND USAGE

```bash
freerec make [DatasetName] [OPTIONS]
```

### KEY OPTIONS

| Option | Default | Description |
|--------|---------|-------------|
| `--root` | `.` | Data root directory |
| `--filedir` | `[DatasetName]` | Subdirectory storing raw files |
| `--splitting` | `ROU` | Splitting strategy |
| `--kcore4user` | `10` | Min interactions per user |
| `--kcore4item` | `10` | Min interactions per item |
| `--star4pos` | `0` | Min rating for positive signal |
| `--ratios` | `8,1,1` | Train/Valid/Test ratios |
| `--days` | `7` | Days for time-based splitting |
| `--userColname` | `USER` | User ID column name |
| `--itemColname` | `ITEM` | Item ID column name |
| `--ratingColname` | `RATING` | Rating column name |
| `--timestampColname` | `TIMESTAMP` | Timestamp column name |

### SPLITTING STRATEGIES

| Method | Description | Use Case |
|--------|-------------|----------|
| `ROU` | Ratio On User - splits each user's history by ratio | General purpose |
| `RAU` | Ratio + At least one - ensures non-empty test per user | When test coverage matters |
| `ROD` | Ratio On Dataset - global temporal split | Temporal evaluation |
| `LOU` | Leave-one-out - last item for test, penultimate for valid | Sequential recommendation |
| `DOU` | Day On User - users split by their last interaction date | Temporal user split |
| `DOD` | Day On Dataset - interactions split by date | Temporal evaluation |

### OUTPUT STRUCTURE

```
data/
└── Processed/
    └── [DatasetName]_[star4pos][kcore4user][kcore4item]_[splitting]/
        ├── train.txt      # Training interactions
        ├── valid.txt      # Validation interactions
        ├── test.txt       # Test interactions
        ├── user.txt       # User features (if exists)
        ├── item.txt       # Item features (if exists)
        ├── config.yaml    # Field configuration
        ├── schema.pkl     # Field schema with statistics
        └── chunks/        # Pickled data chunks for streaming
            ├── train/
            ├── valid/
            └── test/
```

### OUTPUT FILE FORMAT

All `.txt` files are tab-separated with standardized column names:
```
USER	ITEM	RATING	TIMESTAMP
0	2143	4.0	978300019.0
```

### PROCESSING PIPELINE

1. **Load** - Read `.inter`, `.user`, `.item` files
2. **Filter** - Apply rating filter (`>= star4pos`)
3. **Core Filter** - Apply k-core filtering iteratively
4. **Tokenize** - Map user/item IDs to consecutive integers
5. **Sort** - Sort by timestamp within each user
6. **Split** - Apply splitting strategy
7. **Save** - Write output files and build chunk cache

### EXAMPLE

```bash
# Process MovieLens1M with LOU splitting
freerec make MovieLens1M \
    --root ./data \
    --kcore4user 5 \
    --kcore4item 5 \
    --star4pos 0 \
    --splitting LOU

# Output: data/Processed/MovieLens1M_550_LOU/
```

### CODE REFERENCE

- Entry point: `freerec/__main__.py::make()`
- Core class: `freerec/data/preprocessing/base.py::AtomicConverter`
- Key method: `AtomicConverter.make_dataset()`

### DATASET CLASSES

Freerec provides built-in dataset classes in `freerec/data/datasets/`:

| Class | Task | Description |
|-------|------|-------------|
| `RecDataSet` | Base | Generic recommendation dataset |
| `MatchingRecDataSet` | MATCHING | User-Item matching task |
| `NextItemRecDataSet` | NEXTITEM | Sequential next-item prediction |
| `PredictionRecDataSet` | PREDICTION | CTR/CVR prediction task |

### NOTES

- Datasets with `STREAMING=False` (e.g., PredictionRecDataSet) shuffle chunks during training
- Sequential datasets use `NUM_PADS=1` for padding token reservation
- Field tags (USER, ITEM, ID, etc.) control data flow through the pipeline
"""
    },

    "tune": {
        "name": "tune",
        "description": "Tutorial for hyperparameter tuning with freerec",
        "content": r"""## TUNE SKILL - Hyperparameter Tuning Tutorial

### PURPOSE
Performs grid search over hyperparameters with parallel execution across multiple devices.

### COMMAND USAGE

```bash
freerec tune [ExperimentName] [ConfigFile.yaml] [OPTIONS]
```

### CONFIG FILE TEMPLATE

```yaml
command: python main.py
envs:
  root: ../../data
  dataset: Amazon2014Beauty_550_LOU
  device: '0,1,2,3'
  num_workers: 4
params:
  seed: [0, 1, 2, 3, 4]
  lr: [1.e-4, 5.e-4, 1.e-3]
  batch_size: [256, 512]
defaults:
  optimizer: adam
  weight_decay: 0
  epochs: 100
```

### CONFIG SECTIONS

#### 1. `command` (Required)
The base command to execute for each run.

#### 2. `envs` (Optional but recommended)
Environment-level settings:
- `root`: Data root path
- `dataset`: Dataset name
- `device`: Comma-separated GPU IDs (e.g., `'0,1,2,3'`)
- `num_workers`: DataLoader workers per run

#### 3. `params` (Optional)
Hyperparameters to search. Each key can be:
- Single value: Treated as list with one element
- List: Grid search over all values

#### 4. `defaults` (Optional but recommended)
Default values for parameters not in `params`.
Ensures consistent hyperparameter values across all runs for fair comparison.

### CLI OPTIONS

| Option | Description |
|--------|-------------|
| `--root` | Override config's root |
| `--dataset` | Override config's dataset |
| `--device` | Override config's device |
| `--num-workers` | Override config's num_workers |
| `--resume` | Resume from last checkpoint |

### PARALLEL EXECUTION

- Automatically distributes runs across specified devices
- Monitors process status and queues new runs as devices free up
- Handles process termination gracefully with SIGINT

### CHECKPOINT & RESUME

- Checkpoints saved to: `./infos/[ExperimentName]/core/checkpoint.tar`
- Contains: Remaining parameter combinations to search
- Resume with: `freerec tune ... --resume`

### LOG STRUCTURE

```
logs/
└── [description]/              # Experiment name from --description
    ├── core/                   # Core tuning logs (for hyperparameter tuning)
    │   ├── README.md           # Config summary
    │   └── log.txt             # Tuning process log
    └── [dataset]/              # Dataset name
        └── [timestamp-id]/     # Individual run logs (MMDDHHMMSS)
            ├── log.txt         # Training log
            ├── summary/
            │   ├── SUMMARY.md
            │   └── *.png       # Metric curves
            └── data/
                ├── monitors.pkl
                └── best.pkl
```

**Note:** For `freerec tune`, each hyperparameter combination creates a new run with a unique timestamp-id under the same dataset folder.

### BEST RESULTS FORMAT

`best.pkl` structure (key file for AI to read tuning results):
```python
{
    'train': {'LOSS': 0.1234, 'NDCG@10': 0.456},   # Train metrics at last epoch
    'valid': {'LOSS': 0.2345, 'NDCG@10': 0.389},   # Valid metrics at last epoch
    'test': {'LOSS': 0.2456, 'NDCG@10': 0.378},    # Test metrics at last epoch
    'best': {'LOSS': 0.2356, 'NDCG@10': 0.382},    # Test metrics at best validation epoch
}
```

**Key distinction:**
- `'test'`: Metrics on test set using the **last checkpoint** (final epoch)
- `'best'`: Metrics on test set using the **best validation checkpoint** (selected by `which4best`)

### READING RESULTS PROGRAMMATICALLY

```python
import pickle
from freerec.utils import import_pickle

# Load best results for a single run
best = import_pickle('logs/expName/dataset/id/data/best.pkl')
best_test_ndcg = best['best']['NDCG@10']  # Best test performance
last_test_ndcg = best['test']['NDCG@10']  # Last epoch test performance

# Compare across multiple runs
import glob
results = []
for run_dir in glob.glob('logs/expName/core/*/'):
    best = import_pickle(run_dir + 'data/best.pkl')
    results.append(best['best'])
```

### CODE REFERENCE

- Entry point: `freerec/__main__.py::tune()`
- Core class: `freerec/launcher.py::Adapter`
- Key methods:
  - `Adapter.compile()` - Parse config
  - `Adapter.fit()` - Run grid search
  - `Adapter.poll()` - Monitor subprocesses

### EXAMPLE WORKFLOW

1. **Create config** (`tune_config.yaml`):
```yaml
command: python main.py --config=configs/BERT4Rec.yaml
envs:
  root: ../../data
  dataset: Amazon2014Beauty_550_LOU
  device: '0,1,2,3'
params:
  lr: [5.e-4, 1.e-3, 5.e-3]
  embedding_dim: [64, 128]
  seed: [0, 1, 2, 3, 4]
defaults:
  batch_size: 512
  epochs: 100
  maxlen: 50
```

2. **Run tuning**:
```bash
freerec tune BERT4Rec_tune tune_config.yaml
```

3. **Read results**:
```python
# Parse best.pkl files to compare hyperparameter combinations
```

### NOTES

- Each subprocess gets unique ID based on timestamp
- Device assignment is round-robin based on availability
- Results are stored in pickle files for programmatic access
"""
    },

    "log": {
        "name": "log",
        "description": "Guide for reading and understanding freerec training logs",
        "content": r"""## LOG SKILL - Training Log Analysis Guide

### PURPOSE
Explains how to read and extract information from freerec's training logs programmatically.

### LOG DIRECTORY STRUCTURE

```
logs/
└── [description]/
    └── [dataset]/
        └── [timestamp-id]/    # Timestamp-based ID (MMDDHHMMSS)
            ├── log.txt        # Main training log
            ├── model.pt       # Final model weights
            ├── best.pt        # Best model weights
            ├── summary/
            │   ├── SUMMARY.md # Training summary
            │   └── *.png      # Metric curve plots
            └── data/
                ├── monitors.pkl    # Full metric history
                └── best.pkl        # Best results per mode
```

**Note:** The `id` is generated using `time.strftime("%m%d%H%M%S")` at the start of each run.

### LOG FILE: log.txt

#### Format
```
[timestamp]:[message]
```

#### Key Log Entries

**Training Start:**
```
[Coach] >>> Set best meter: NDCG@10
[Seed] >>> Set seed: 1
[Benchmark] >>> cudnn.benchmark == False | cudnn.deterministic == True
```

**Per-Epoch Training:**
```
[Coach] >>> TRAIN @Epoch: 1 >>> LOSS Avg: 0.52341
[Coach] >>> VALID @Epoch: 5 >>> LOSS Avg: 0.34521 || NDCG@10 Avg: 0.38900
[Coach] >>> Better ***NDCG@10*** of ***0.3890***
```

**Early Stopping:**
```
[Coach] >>> Early Stop @Epoch: 45
```

**Summary:**
```
[LoG_PaTH] >>> logs/RecSys/Amazon2014Beauty_550_LOU/0322143052
```

### SUMMARY FILE: SUMMARY.md

#### Format
```markdown
| Mode | Metric | Best | @Step | Img |
|------|--------|------|-------|-----|
| train | LOSS | 0.1234 | 50 | ![](trainLOSS.png) |
| valid | NDCG@10 | 0.4567 | 45 | ![](validNDCG@10.png) |
| test | HITRATE@10 | 0.3456 | 50 | ![](testHITRATE@10.png) |
```

#### Contents
- Best metric values per mode (train/valid/test)
- Epoch/step at which best occurred
- Links to curve plots (PNG files)

### METRIC FILES

#### monitors.pkl (Full History)

Structure: `mode → metric_family (e.g. NDCG) → specific_metric (e.g. NDCG@10) → [history]`

```python
{
    'train': {
        'LOSS': {
            'LOSS': [0.5, 0.4, 0.35, ...],
        },
        'NDCG': {
            'NDCG@5': [],
            'NDCG@10': [],
        },
        'HITRATE': {
            'HITRATE@1': [],
            'HITRATE@5': [],
            'HITRATE@10': [],
        },
    },
    'valid': {
        'LOSS': {'LOSS': []},
        'NDCG': {
            'NDCG@5': [0.0253, 0.0289, ...],
            'NDCG@10': [0.0332, 0.0371, ...],
        },
        'HITRATE': {
            'HITRATE@1': [0.0099, ...],
            'HITRATE@5': [0.0407, ...],
            'HITRATE@10': [0.0654, ...],
        },
    },
    'test': {...},
}
```

#### best.pkl (Best Results)
```python
{
    'train': {'LOSS': 0.1234, 'NDCG@10': 0.5678},   # Train metrics at last epoch
    'valid': {'LOSS': 0.2345, 'NDCG@10': 0.4567},   # Valid metrics at last epoch
    'test': {'LOSS': 0.2456, 'NDCG@10': 0.4321},    # Test metrics at last epoch
    'best': {'LOSS': 0.2356, 'NDCG@10': 0.4437},    # Test metrics at best validation epoch
}
```

**Key distinction:**
- `'test'`: Metrics on test set using the **last checkpoint** (final epoch)
- `'best'`: Metrics on test set using the **best validation checkpoint** (selected by `which4best`)

### AVAILABLE METRICS

#### Loss Metrics
| Metric | Better | Description |
|--------|--------|-------------|
| `LOSS` | min | Training loss |
| `MSE` | min | Mean squared error |
| `MAE` | min | Mean absolute error |
| `RMSE` | min | Root mean squared error |
| `LOGLOSS` | min | Logarithmic loss |

#### Ranking Metrics
| Metric | Better | Description |
|--------|--------|-------------|
| `NDCG` | max | Normalized discounted cumulative gain |
| `MRR` | max | Mean reciprocal rank |
| `MAP` | max | Mean average precision |
| `HITRATE` | max | Hit rate (recall@k) |
| `PRECISION` | max | Precision@k |
| `RECALL` | max | Recall@k |
| `F1` | max | F1 score |

#### Classification Metrics
| Metric | Better | Description |
|--------|--------|-------------|
| `AUC` | max | Area under ROC curve |
| `GAUC` | max | Grouped AUC |

### METRIC VISUALIZATION

Curve plots saved as `PNG` in `summary/`:
- Filename format: `[mode][metric].png`
- Examples: `validNDCG@10.png`, `trainLOSS.png`
- X-axis: Epoch/Step (based on eval_freq)
- Y-axis: Metric value

### READING LOGS PROGRAMMATICALLY

```python
import pickle
from freerec.utils import import_pickle

# Load full history
monitors = import_pickle('logs/exp/dataset/id/data/monitors.pkl')
train_loss_history = monitors['train']['LOSS']['LOSS']
valid_ndcg_history = monitors['valid']['NDCG']['NDCG@10']

# Load best results
best = import_pickle('logs/exp/dataset/id/data/best.pkl')
best_test_ndcg = best['valid']['NDCG@10']    # Best validation checkpoint's validation performance
best_test_ndcg = best['best']['NDCG@10']    # Best validation checkpoint's test performance
last_test_ndcg = best['test']['NDCG@10']    # Last epoch's test performance
```

### PARSING LOG.TXT

```python
import re

def parse_log(log_path):
    with open(log_path) as f:
        content = f.read()

    # Extract epoch results
    pattern = r'\[Coach\] >>> (\w+) @Epoch: (\d+).*?(\w+) Avg: ([\d.]+)'
    matches = re.findall(pattern, content)

    # Extract best model info
    best_pattern = r'Better \*\*\*(\w+)\*\*\* of \*\*\*([\d.]+)\*\*\*'
    best_matches = re.findall(best_pattern, content)

    return matches, best_matches
```

### TUNING LOGS (freerec tune)

For hyperparameter tuning experiments:

```
logs/
└── [ExperimentName]/
    ├── core/
    │   ├── README.md           # Config summary
    │   └── log.txt             # Tuning coordinator log
    └── [timestamp]/            # Individual runs
        └── ...
```

To compare results across runs, read each run's `best.pkl`:
```python
import glob
for run_dir in glob.glob('logs/expName/*'):
    best = import_pickle(run_dir + '/data/best.pkl')
    print(f"Run {run_dir}: best_test_ndcg = {best['best']['NDCG@10']}")
```

### CODE REFERENCE

- Log setup: `freerec/parser.py::Parser.compile()`
- Logging: `freerec/utils.py::infoLogger()`, `LOGGER`
- Summary: `freerec/launcher.py::Coach.summary()`
- Monitors: `freerec/utils.py::Monitor`, `AverageMeter`

### QUICK REFERENCE

| File | Purpose | Format | How to Read |
|------|---------|--------|-------------|
| `log.txt` | Real-time training output | Plain text | Parse with regex |
| `SUMMARY.md` | Best results summary | Markdown table | Parse as markdown |
| `monitors.pkl` | Full metric history | Pickle dict | `import_pickle()` |
| `best.pkl` | Best results | Pickle dict | `import_pickle()` |
| `*.png` | Metric curves | PNG images | Load with matplotlib |
"""
    },

    "workflow": {
        "name": "workflow",
        "description": "Overview of freerec's training pipeline and architecture",
        "content": r"""## WORKFLOW SKILL - Training Pipeline Overview

### PURPOSE
Explains the complete training workflow from data loading to model evaluation.

### ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER ENTRY POINT                         │
│  freerec make  │  freerec tune  │  python main.py --config      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION LAYER                         │
│  Parser (parser.py)  │  CoreParser  │  Config (dict2obj.py)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  RecDataSet → DataPipe → DataLoader2 (torchdata)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                                │
│  RecSysArch → GenRecArch/SeqRecArch/PredRecArch                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                              │
│  Coach (launcher.py) → train/valid/test loops                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LOGGING LAYER                              │
│  monitors.pkl  │  best.pkl  │  SUMMARY.md                       │
└─────────────────────────────────────────────────────────────────┘
```

### COMPONENT DETAILS

#### 1. CONFIGURATION LAYER

**Parser** (`freerec/parser.py::Parser`)
- Parses CLI arguments and YAML config
- Sets up device, seed, logging
- Creates directory structure

**Key Settings:**
```yaml
root: ../../data
dataset: Amazon2014Beauty_550_LOU
device: '0'
batch_size: 512
lr: 5.e-4
epochs: 100
eval_freq: 5
monitors: [LOSS, NDCG@10, HITRATE@10]
which4best: NDCG@10
```

#### 2. DATA LAYER

**Dataset Class Hierarchy:**
```
BaseSet (IterDataPipe)
└── RecDataSet
    ├── MatchingRecDataSet    # User-Item matching
    ├── NextItemRecDataSet    # Sequential recommendation
    └── PredictionRecDataSet  # CTR/CVR prediction
```

**Data Flow:**
```
RecDataSet
    │
    ├── Source (User IDs / Pairs / Sequences)
    │
    ├── Transform (Sampling / Padding / etc.)
    │
    ├── Batch (batch_size)
    │
    └── Tensor (convert to torch.Tensor)
```

**Task-Specific DataPipes:**

| Task | Train Source | Valid/Test Source |
|------|--------------|-------------------|
| MATCHING | `shuffled_pairs_source()` | `ordered_user_ids_source()` |
| NEXTITEM | `shuffled_seqs_source()` | `ordered_user_ids_source()` |
| PREDICTION | `shuffled_inter_source()` | `ordered_inter_source()` |

#### 3. MODEL LAYER

**Base Architecture:** `RecSysArch` (`freerec/models/base.py`)

**Key Methods:**
```python
class RecSysArch(nn.Module):
    def fit(self, data) -> Loss         # Training forward
    def forward(self, data, ranking)    # Inference
    def recommend_from_full(self, data) # Full ranking
    def recommend_from_pool(self, data) # Sampled ranking
    def reset_ranking_buffers(self)     # Pre-eval cleanup
```

**Architecture Types:**

| Class | Task | Key Features |
|-------|------|--------------|
| `GenRecArch` | MATCHING | User/Item ID-based |
| `SeqRecArch` | NEXTITEM | Sequence modeling, padding |
| `PredRecArch` | PREDICTION | Point-wise prediction |

#### 4. TRAINING LAYER

**Coach Class** (`freerec/launcher.py::Coach`)

**Training Loop:**
```python
for epoch in range(start_epoch, epochs):
    if epoch % CHECKPOINT_FREQ == 0:
        save_checkpoint(epoch)
    if epoch % eval_freq == 0:
        valid(epoch)    # Evaluate on validation set
        test(epoch)     # Evaluate on test set (optional)
    train(epoch + 1)    # Train for one epoch

save()                  # Save final model
eval_at_best()          # Evaluate best model
summary()               # Generate summary report
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `ChiefCoach` | Base class with dataset/model/optimizer setup |
| `Coach` | Implements training/evaluation loops |
| `Adapter` | Hyperparameter tuning coordinator |

**Monitor System:**
```python
# Register metrics
self.set_monitors(['LOSS', 'NDCG@10', 'HITRATE@10'])

# During training
self.monitor(scores, targets, n=batch_size, mode='train')

# Logging
self.step(epoch)  # Prints: [Coach] >>> TRAIN @Epoch: 1 >>> LOSS Avg: 0.52341
```

**Early Stopping:**
- Triggered when `which4best` metric doesn't improve
- Controlled by `early_stop_patience`
- Saves best model automatically

#### 5. LOGGING LAYER

**Outputs:**

| File | Content |
|------|---------|
| `log.txt` | Real-time training logs |
| `summary/SUMMARY.md` | Best results table |
| `summary/*.png` | Metric curves |
| `data/monitors.pkl` | Full history |
| `data/best.pkl` | Best results |
| `model.pt` | Final weights |
| `best.pt` | Best weights |

### EXECUTION FLOW

```
┌──────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                             │
│    - Parse config                                              │
│    - Set device, seed                                          │
│    - Create directories                                        │
│    - Setup logging                                             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. DATA PREPARATION                                           │
│    - Load RecDataSet                                           │
│    - Build fields from schema                                  │
│    - Create DataPipes (train/valid/test)                       │
│    - Setup DataLoader2 with workers                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. MODEL SETUP                                                │
│    - Instantiate RecSysArch                                    │
│    - Setup optimizer (Adam/SGD/AdamW)                          │
│    - Setup LR scheduler (if any)                               │
│    - Move to device                                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. TRAINING LOOP                                              │
│    For each epoch:                                             │
│    - train(): Iterate trainloader, compute loss, backward     │
│    - valid(): Evaluate on validation set                       │
│    - check_best(): Update best model if improved               │
│    - Early stopping check                                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. FINALIZATION                                               │
│    - save(): Save final model                                  │
│    - load_best(): Load best checkpoint                         │
│    - eval_at_best(): Final evaluation                          │
│    - summary(): Generate SUMMARY.md                            │
└──────────────────────────────────────────────────────────────┘
```

### DISTRIBUTED TRAINING

**TorchRun Support:**
```bash
torchrun --nproc_per_node=4 main.py --config=config.yaml
```

**DDP Setup:**
- Handled by `ChiefCoach.set_model()`
- Uses `DistributedDataParallel` wrapper
- Gradient aggregation automatic

### KEY FILE PATHS

```python
# From parser.py
CHECKPOINT_PATH = "./infos/{description}/{dataset}/{device}"
LOG_PATH = "./logs/{description}/{dataset}/{id}"
CORE_CHECKPOINT_PATH = "./infos/{description}/core"
CORE_LOG_PATH = "./logs/{description}/core"
```

### CODE REFERENCE

| Component | File | Class/Function |
|-----------|------|----------------|
| Config | `parser.py` | `Parser`, `CoreParser` |
| Dataset | `data/datasets/base.py` | `RecDataSet` |
| DataPipe | `data/postprocessing/` | `PostProcessor` |
| Model | `models/base.py` | `RecSysArch` |
| Coach | `launcher.py` | `Coach`, `ChiefCoach` |
| Tuner | `launcher.py` | `Adapter` |
| Metrics | `utils.py` | `AverageMeter`, `Monitor` |
| Logging | `utils.py` | `infoLogger`, `LOGGER` |
"""
    }
}


def get_skill(skill_name: str) -> dict:
    r"""Retrieve a skill dictionary by name.

    Parameters
    ----------
    skill_name : str
        Name of the skill to retrieve.

    Returns
    -------
    dict or None
        The skill dictionary containing ``name``, ``description``, and
        ``content`` keys, or ``None`` if not found.
    """
    return SKILLS.get(skill_name)


def list_skills() -> list:
    r"""Return the names of all registered skills.

    Returns
    -------
    list of str
        Skill names.
    """
    return list(SKILLS.keys())


def print_skill(skill_name: str):
    r"""Print a skill's tutorial content to stdout.

    Parameters
    ----------
    skill_name : str
        Name of the skill to print.
    """
    skill = get_skill(skill_name)
    if skill:
        print(skill["content"])
    else:
        print(f"Error: Skill '{skill_name}' not found.")
        print(f"Available skills: {list_skills()}")
