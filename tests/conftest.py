import os

import pytest
import torch

FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def toy_root():
    """Root path for the toy dataset (tests/fixtures)."""
    return FIXTURES_ROOT


@pytest.fixture
def toy_filedir():
    """Filedir name for the toy dataset."""
    return "Example_XXX_LOU"


@pytest.fixture
def random_preds_targets():
    """Random prediction scores and binary targets for ranking metrics."""
    preds = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.7]])
    targets = torch.tensor([[1, 0, 1, 0, 1]])
    return preds, targets


@pytest.fixture
def batch_preds_targets():
    """Batch of prediction scores and binary targets."""
    preds = torch.tensor(
        [
            [0.9, 0.1, 0.8, 0.2, 0.7],
            [0.3, 0.6, 0.1, 0.9, 0.5],
        ]
    )
    targets = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]
    )
    return preds, targets
