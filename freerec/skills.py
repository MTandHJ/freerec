r"""Freerec Skills Module.

Provides concise reference guides for AI assistants to understand
freerec's internal mechanisms. Each skill can be accessed via
``freerec skill --<name>``.
"""

SKILLS = {
    "make": {
        "name": "make",
        "description": "Convert raw interaction data into freerec's standard format",
        "content": r"""## MAKE — Dataset Processing

Convert RecBole-style atomic files (.inter, .user, .item) into freerec format
with k-core filtering, tokenization, and train/valid/test splitting.

### Usage
    freerec make [DatasetName] [OPTIONS]

### Input
    data/[DatasetName]/
    ├── [DatasetName].inter    # TSV: USER, ITEM, RATING, TIMESTAMP
    ├── [DatasetName].user     # Optional: user features
    └── [DatasetName].item     # Optional: item features

    Column names are configurable via --userColname, --itemColname,
    --ratingColname, --timestampColname.

### Key Options
    --splitting     ROU (default) | RAU | ROD | LOU | DOU | DOD
    --kcore4user    10     Min interactions per user
    --kcore4item    10     Min interactions per item
    --star4pos      0      Min rating threshold for positive signal
    --ratios        8,1,1  Train/Valid/Test split ratios
    --days          7      Days for time-based splitting (DOU/DOD)
    --root          .      Data root directory
    --filedir       None   Subdirectory storing raw files (defaults to dataset name)

### Splitting Strategies
    ROU  Ratio On User — splits each user's history by ratio.
    RAU  Ratio + At least one — ensures ≥1 test item per user.
    ROD  Ratio On Dataset — global chronological split by ratio.
    LOU  Leave-One-Out — per-user: last item → test, penultimate → valid.
    DOU  Day On User — split users by their last interaction date.
    DOD  Day On Dataset — split interactions by date.

### Output
    data/Processed/[Dataset]_[star4pos][kcore4user][kcore4item]_[splitting]/
    ├── train.txt, valid.txt, test.txt  # TSV: USER, ITEM, RATING, TIMESTAMP
    ├── user.txt, item.txt              # If source had features
    ├── config.yaml, schema.pkl
    └── chunks/{train,valid,test}/      # Pickled data for streaming

### Pipeline
    Load → Rating filter (≥star4pos) → k-core filter → Tokenize →
    Sort (by timestamp) → Split → Save

### Dataset Classes (freerec/data/datasets/)
    RecDataSet              Base class (IterDataPipe)
    MatchingRecDataSet      User-Item matching (e.g. BPR, LightGCN)
    NextItemRecDataSet      Sequential next-item prediction (e.g. SASRec, BERT4Rec)
    PredictionRecDataSet    CTR/CVR prediction (e.g. DeepFM, DCN)

    PredictionRecDataSet sets STREAMING=False (shuffles chunks each epoch).
    NextItemRecDataSet uses NUM_PADS=1 for padding token reservation.

### Source Code
    Entry:  freerec/__main__.py::make()
    Core:   freerec/data/preprocessing/base.py::AtomicConverter.make_dataset()
""",
    },
    "tune": {
        "name": "tune",
        "description": "Grid-search hyperparameter tuning with parallel GPU execution",
        "content": r"""## TUNE — Hyperparameter Tuning

Grid search over hyperparameters with parallel execution across GPUs.
Each combination runs as an independent process; device assignment is round-robin.

### Usage
    freerec tune [ExperimentName] [ConfigFile.yaml] [OPTIONS]

### Tuning Workflow
    1. Write or edit ConfigFile.yaml — define params grid and defaults.
    2. freerec tune [ExpName] [ConfigFile.yaml] — run grid search.
    3. Read logs/[ExpName]/core/results.json — compare valid metrics.
    4. Pick the best combination, set it as defaults in ConfigFile.yaml.
    5. Narrow the params grid around the best values, or explore new dims.
    6. Repeat from step 2 until satisfied.

### Config Format
    ```yaml
    command: python main.py
    envs:
      root: ../../data
      dataset: Amazon2014Beauty_550_LOU
      device: '0,0,1,1,2,2,3,3'   #  one entry = one concurrent task
    params:                          # Cartesian product of all values
      lr: [1.e-4, 5.e-4, 1.e-3]
      batch_size: [256, 512]
    defaults:
      config: configs/Amazon2014Beauty_550_LOU.yaml
      optimizer: adam
      epochs: 100
    ```

    Note: Device assignment is round-robin across the comma-separated list —
    each entry is one concurrent task, NOT a multi-GPU training config.

### CLI Options
    --root PATH       Override config's root
    --dataset NAME    Override config's dataset
    --device STR      Override config's device assignment
    --num-workers N   Override config's num_workers
    --resume          Resume from last checkpoint

### Output
    logs/[ExperimentName]/
    ├── core/
    │   ├── log.txt              # Coordinator log
    │   ├── README.md            # Config summary
    │   └── results.json         # Aggregated results (appended per-run)
    └── [dataset]/[id]/          # id = MMDDHHMMSS, one per hyperparam combo
        ├── log.txt, model.pt, best.pt
        ├── summary/{SUMMARY.md,*.png}
        └── data/{monitors.pkl, best.pkl}

### results.json
    {"description": "...", "dataset": "...", "timestamp": "...",
     "runs": [{"id": "MMDDHHMMSS", "params": {...},
               "metrics": {"train": {}, "valid": {}, "test": {}, "best": {}}}]}

    Multiple tune sessions to the same description append to the same file.

### Reading Results
    ```python
    import json
    from freerec.utils import import_pickle

    # Aggregated results
    results = json.load(open('logs/exp/core/results.json'))
    for run in results['runs']:
        print(run['params'], run['metrics']['valid']['NDCG@10'])

    # Single run
    best = import_pickle('logs/exp/dataset/id/data/best.pkl')
    best['valid']['NDCG@10']   # Valid metrics at best validation epoch
    best['best']['NDCG@10']    # Test metrics at best validation epoch
    best['test']['NDCG@10']    # Test metrics at last epoch
    ```

### Source Code
    Entry:  freerec/__main__.py::tune()
    Core:   freerec/launcher.py::Adapter
    Config: freerec/parser.py::CoreParser
""",
    },
    "log": {
        "name": "log",
        "description": "Read and interpret freerec training logs and metric files",
        "content": r"""## LOG — Training Log Analysis

### Directory Structure
    logs/[description]/[dataset]/[id]/    # id = MMDDHHMMSS
    ├── log.txt             # Real-time training output
    ├── model.pt, best.pt   # Model weights (final, best)
    ├── summary/
    │   ├── SUMMARY.md      # Best metrics table
    │   └── *.png           # Metric curves (e.g. validNDCG@10.png)
    └── data/
        ├── monitors.pkl    # Full metric history (per epoch/step)
        └── best.pkl        # Best results snapshot

### log.txt Format
    ```text
    [Coach] >>> Set best meter: NDCG@10
    [Coach] >>> TRAIN @Epoch: 1 >>> LOSS Avg: 0.52341
    [Coach] >>> VALID @Epoch: 5 >>> LOSS Avg: 0.345 || NDCG@10 Avg: 0.389
    [Coach] >>> Better ***NDCG@10*** of ***0.3890***
    [Coach] >>> Early Stop @Epoch: 45
    ```

### best.pkl
    WARNING — 'test' ≠ 'best':
      'test' comes from the final-epoch checkpoint.
      'best' comes from the checkpoint that maximized the validation metric.
      Always report 'best' for paper results; use 'test' only for debugging.

    {'train': {'LOSS': 0.12},          # Last-epoch train metrics
     'valid': {'NDCG@10': 0.39},       # Last-epoch valid metrics
     'test':  {'NDCG@10': 0.38},       # Last-epoch test metrics
     'best':  {'NDCG@10': 0.44}}       # Test metrics at best-valid epoch

### monitors.pkl
    mode → metric_family → metric_name → [history_per_step]

    {'train': {'LOSS': {'LOSS': [0.5, 0.4, ...]}},
     'valid': {'NDCG': {'NDCG@10': [0.03, 0.04, ...]},
               'HITRATE': {'HITRATE@10': [0.06, 0.07, ...]}}}

### Available Metrics
    Loss:      LOSS, MSE, MAE, RMSE, LOGLOSS
    Ranking:   NDCG@K, HITRATE@K, PRECISION@K, RECALL@K, MRR@K, MAP@K, F1@K
    Classif.:  AUC, GAUC

### Reading Programmatically
    from freerec.utils import import_pickle

    monitors = import_pickle('logs/.../data/monitors.pkl')
    history = monitors['valid']['NDCG']['NDCG@10']

    best = import_pickle('logs/.../data/best.pkl')
    best_test = best['best']['NDCG@10']    # Use this for reporting

### Source Code
    Logging:  freerec/utils.py::{LOGGER, infoLogger, Monitor, AverageMeter}
    Summary:  freerec/launcher.py::Coach.summary()
    Paths:    freerec/parser.py::Parser.compile()
""",
    },
    "workflow": {
        "name": "workflow",
        "description": "Overview of freerec's training pipeline and architecture",
        "content": r"""## WORKFLOW — Training Pipeline Overview

### Lifecycle
    CLI / YAML config
        ↓
    Parser (freerec/parser.py)
        ↓
    RecDataSet / DataPipe (freerec/data/)
        ↓
    RecSysArch model (freerec/models/)
        ↓
    Coach training loop (freerec/launcher.py)
        ↓
    Metrics & Logging (freerec/utils.py)

### Config (freerec/parser.py::Parser)
    Parses CLI args + YAML config → dict2obj namespace.
    Key fields: root, dataset, device, batch_size, lr, epochs, eval_freq,
                monitors, which4best, early_stop_patience.

### Data (freerec/data/)
    RecDataSet (IterDataPipe) — three task-specific subclasses:

    Task          Dataset               Train Source                Valid/Test Source
    MATCHING      MatchingRecDataSet    shuffled_pairs_source()     ordered_user_ids_source()
    NEXTITEM      NextItemRecDataSet    shuffled_seqs_source()      ordered_user_ids_source()
    PREDICTION    PredictionRecDataSet  shuffled_inter_source()     ordered_inter_source()

    Pipeline: Source → Transform → Batch → Tensor (via torchdata DataLoader2)

    When to use:
      MATCHING    — User-Item recommendation (BPR, LightGCN, etc.)
      NEXTITEM    — Sequential next-item prediction (SASRec, BERT4Rec, etc.)
      PREDICTION  — CTR / CVR prediction with rich features (DeepFM, DCN, etc.)

### Model (freerec/models/base.py::RecSysArch)
    Subclass: GenRecArch (MATCHING), SeqRecArch (NEXTITEM), PredRecArch (PREDICTION)

    Key methods:
      fit(data) → Loss           Training forward pass
      forward(data, ranking)     Inference
      recommend_from_full(data)  Full ranking over all items
      recommend_from_pool(data)  Sampled ranking over candidate pool

### Training (freerec/launcher.py)
    ChiefCoach  Base: dataset, model, optimizer, DDP setup.
    Coach       Training/eval loops, early stopping, checkpointing, summary.
    Adapter     Hyperparameter tuning coordinator (freerec tune).

    Loop:
      for epoch in range(epochs):
          if epoch % eval_freq == 0: valid(epoch); test(epoch)
          train(epoch + 1)
      save(); eval_at_best(); summary()

    Early stopping: triggers when which4best doesn't improve for
    early_stop_patience consecutive evaluations.

### Logging
    See: freerec skill --log  and  freerec skill --tune

    Key outputs:
      log.txt          Real-time training log
      monitors.pkl     Full metric history (per epoch/step)
      best.pkl         Best results snapshot
      SUMMARY.md       Best metrics summary table
      *.png            Metric curves

### Distributed Training (DDP)
    torchrun --nproc_per_node=N main.py --config=config.yaml
    Handled by ChiefCoach.set_model() via DistributedDataParallel.

### Path Templates
    CHECKPOINT:  ./infos/{description}/{dataset}/{device}
    LOG:         ./logs/{description}/{dataset}/{id}

### Source Code
    Config:   freerec/parser.py (Parser, CoreParser)
    Data:     freerec/data/datasets/base.py (RecDataSet)
    DataPipe: freerec/data/postprocessing/ (PostProcessor)
    Model:    freerec/models/base.py (RecSysArch)
    Coach:    freerec/launcher.py (Coach, ChiefCoach, Adapter)
    Metrics:  freerec/utils.py (Monitor, AverageMeter)
    Logging:  freerec/utils.py (LOGGER, infoLogger)
""",
    },
}


def get_skill(skill_name: str) -> dict | None:
    """Retrieve a skill dictionary by name.

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
    """Return the names of all registered skills.

    Returns
    -------
    list of str
        Skill names.
    """
    return sorted(SKILLS)


def print_skill(skill_name: str):
    """Print a skill's content to stdout.

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
