"""Integration tests for datapipe chains used in examples.

Uses a toy dataset (10 users, 20 items) at tests/fixtures/Processed/Example_XXX_LOU/.
Tests verify that full datapipe chains iterate without errors and produce
correctly shaped outputs.
"""

import os
import shutil

import pytest
import torch

from freerec.data.datasets.base import NextItemRecDataSet
from freerec.data.tags import ID, ITEM, SEQUENCE, USER

FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures")
FILEDIR = "Example_XXX_LOU"
DATASET_PATH = os.path.join(FIXTURES_ROOT, "Processed", FILEDIR)

# Clean up generated cache before/after test session
CACHE_PATHS = [
    os.path.join(DATASET_PATH, "schema.pkl"),
    os.path.join(DATASET_PATH, "chunks"),
]


@pytest.fixture(scope="module")
def dataset():
    """Load the toy dataset once per module."""
    ds = NextItemRecDataSet(root=FIXTURES_ROOT, filedir=FILEDIR)
    yield ds
    # Cleanup generated cache files
    for p in CACHE_PATHS:
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)


class TestDatasetBasics:
    """Test that the toy dataset loads correctly."""

    def test_loads_successfully(self, dataset):
        assert dataset is not None

    def test_user_item_counts(self, dataset):
        User = dataset.fields[USER, ID]
        Item = dataset.fields[ITEM, ID]
        assert User.count == 10
        assert Item.count == 20

    def test_mode_switching(self, dataset):
        dataset.train()
        assert dataset.mode == "train"
        dataset.valid()
        assert dataset.mode == "valid"
        dataset.test()
        assert dataset.mode == "test"
        dataset.train()  # reset

    def test_train_size(self, dataset):
        assert dataset.trainsize == 47

    def test_valid_test_size(self, dataset):
        assert dataset.validsize == 10
        assert dataset.testsize == 10


class TestGenTrainPipeline:
    """Test Gen-based training pipeline (MF-BPR / LightGCN style).

    Chain: choiced_user_ids_source -> gen_train_sampling_pos_
           -> gen_train_sampling_neg_ -> batch_ -> tensor_
    """

    def test_gen_train_pipeline_iterates(self, dataset):
        batch_size = 4
        pipe = (
            dataset.train()
            .choiced_user_ids_source()
            .gen_train_sampling_pos_()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )
        batch = next(iter(pipe))
        assert isinstance(batch, dict)

        # Check that we have User, positive Item, negative Item fields + SIZE
        assert len(batch) >= 3

    def test_gen_train_output_shapes(self, dataset):
        batch_size = 4
        pipe = (
            dataset.train()
            .choiced_user_ids_source()
            .gen_train_sampling_pos_()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )
        batch = next(iter(pipe))

        for field, val in batch.items():
            if hasattr(field, "name") and field.name != "SIZE":
                if isinstance(val, torch.Tensor):
                    assert val.shape[0] == batch_size


class TestSeqTrainPipeline:
    """Test Seq-based training pipeline (SASRec / GRU4Rec style).

    Chain: shuffled_seqs_source -> seq_train_yielding_pos_
           -> seq_train_sampling_neg_ -> add_ -> lpad_ -> batch_ -> tensor_
    """

    def test_seq_train_pipeline_iterates(self, dataset):
        maxlen = 5
        batch_size = 4
        NUM_PADS = 1
        PADDING_VALUE = 0

        ISeq = dataset.fields[ITEM, ID].fork(SEQUENCE)

        pipe = (
            dataset.train()
            .shuffled_seqs_source(maxlen=maxlen)
            .seq_train_yielding_pos_(start_idx_for_target=1, end_idx_for_input=-1)
            .seq_train_sampling_neg_(num_negatives=1)
            .add_(offset=NUM_PADS, modified_fields=(ISeq,))
            .lpad_(
                maxlen,
                modified_fields=(ISeq,),
                padding_value=PADDING_VALUE,
            )
            .batch_(batch_size)
            .tensor_()
        )
        batch = next(iter(pipe))
        assert isinstance(batch, dict)


class TestEvalPipeline:
    """Test evaluation pipeline (valid/test).

    Chain: ordered_user_ids_source -> valid_sampling_
           -> lprune_ -> add_ -> rpad_ -> batch_ -> tensor_
    """

    def test_valid_pipeline_iterates(self, dataset):
        maxlen = 5
        batch_size = 4
        NUM_PADS = 1
        PADDING_VALUE = 0

        ISeq = dataset.fields[ITEM, ID].fork(SEQUENCE)

        pipe = (
            dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking="full")
            .lprune_(maxlen, modified_fields=(ISeq,))
            .add_(offset=NUM_PADS, modified_fields=(ISeq,))
            .rpad_(maxlen, modified_fields=(ISeq,), padding_value=PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )
        batch = next(iter(pipe))
        assert isinstance(batch, dict)

    def test_test_pipeline_iterates(self, dataset):
        maxlen = 5
        batch_size = 4
        NUM_PADS = 1
        PADDING_VALUE = 0

        ISeq = dataset.fields[ITEM, ID].fork(SEQUENCE)

        pipe = (
            dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking="full")
            .lprune_(maxlen, modified_fields=(ISeq,))
            .add_(offset=NUM_PADS, modified_fields=(ISeq,))
            .rpad_(maxlen, modified_fields=(ISeq,), padding_value=PADDING_VALUE)
            .batch_(batch_size)
            .tensor_()
        )
        batch = next(iter(pipe))
        assert isinstance(batch, dict)
