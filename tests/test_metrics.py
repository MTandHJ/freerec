"""Tests for freerec.metrics module."""

import numpy as np
import pytest
import torch

from freerec.metrics import (
    auroc,
    f1_score,
    group_auroc,
    hit_rate,
    log_loss,
    mean_abs_error,
    mean_average_precision,
    mean_reciprocal_rank,
    mean_squared_error,
    normalized_dcg,
    precision,
    recall,
    root_mse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    preds = torch.tensor([[0.2, 0.3, 0.5, 0.0], [0.1, 0.3, 0.5, 0.2]])
    targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    return preds, targets


@pytest.fixture
def ranking_data():
    preds = torch.tensor([[0.2, 0.3, 0.5, 0.0], [0.1, 0.3, 0.5, 0.2]])
    targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    return preds, targets


# ---------------------------------------------------------------------------
# Regression metrics — docstring examples
# ---------------------------------------------------------------------------


class TestMeanAbsError:
    def test_none(self, regression_data):
        preds, targets = regression_data
        result = mean_abs_error(preds, targets, reduction="none")
        torch.testing.assert_close(result, torch.tensor([0.5000, 0.5250]))

    def test_mean(self, regression_data):
        preds, targets = regression_data
        result = mean_abs_error(preds, targets)
        torch.testing.assert_close(result, torch.tensor(0.5125))

    def test_sum(self, regression_data):
        preds, targets = regression_data
        result = mean_abs_error(preds, targets, reduction="sum")
        torch.testing.assert_close(result, torch.tensor(0.5000 + 0.5250))


class TestMeanSquaredError:
    def test_none(self, regression_data):
        preds, targets = regression_data
        result = mean_squared_error(preds, targets, reduction="none")
        torch.testing.assert_close(result, torch.tensor([0.3450, 0.3475]))

    def test_mean(self, regression_data):
        preds, targets = regression_data
        result = mean_squared_error(preds, targets)
        torch.testing.assert_close(result, torch.tensor(0.3462), atol=1e-4, rtol=0)

    def test_sum(self, regression_data):
        preds, targets = regression_data
        result = mean_squared_error(preds, targets, reduction="sum")
        torch.testing.assert_close(result, torch.tensor(0.3450 + 0.3475))


class TestRootMSE:
    def test_none(self, regression_data):
        preds, targets = regression_data
        result = root_mse(preds, targets, reduction="none")
        torch.testing.assert_close(
            result, torch.tensor([0.5874, 0.5895]), atol=5e-5, rtol=0
        )

    def test_mean(self, regression_data):
        preds, targets = regression_data
        result = root_mse(preds, targets)
        torch.testing.assert_close(result, torch.tensor(0.5884), atol=5e-5, rtol=0)

    def test_sum(self, regression_data):
        preds, targets = regression_data
        result = root_mse(preds, targets, reduction="sum")
        expected = torch.tensor([0.5874, 0.5895]).sum()
        torch.testing.assert_close(result, expected, atol=5e-4, rtol=0)


# ---------------------------------------------------------------------------
# Regression metrics — parametrize reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_fn", [mean_abs_error, mean_squared_error, root_mse])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_regression_reduction_shapes(metric_fn, reduction, regression_data):
    preds, targets = regression_data
    result = metric_fn(preds, targets, reduction=reduction)
    if reduction == "none":
        assert result.shape == (2,)
    else:
        assert result.shape == ()


# ---------------------------------------------------------------------------
# Ranking metrics — docstring examples with k
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_k3_none(self, ranking_data):
        preds, targets = ranking_data
        result = precision(preds, targets, k=3, reduction="none")
        torch.testing.assert_close(
            result, torch.tensor([0.3333, 0.6667]), atol=5e-5, rtol=0
        )

    def test_k3_mean(self, ranking_data):
        preds, targets = ranking_data
        result = precision(preds, targets, k=3)
        torch.testing.assert_close(result, torch.tensor(0.5000))


class TestRecall:
    def test_k3_none(self, ranking_data):
        preds, targets = ranking_data
        result = recall(preds, targets, k=3, reduction="none")
        torch.testing.assert_close(result, torch.tensor([0.5000, 1.0000]))

    def test_k3_mean(self, ranking_data):
        preds, targets = ranking_data
        result = recall(preds, targets, k=3)
        torch.testing.assert_close(result, torch.tensor(0.7500))


class TestF1Score:
    def test_k3_none(self, ranking_data):
        preds, targets = ranking_data
        result = f1_score(preds, targets, k=3, reduction="none")
        torch.testing.assert_close(
            result, torch.tensor([0.4000, 0.8000]), atol=5e-5, rtol=0
        )

    def test_k3_mean(self, ranking_data):
        preds, targets = ranking_data
        result = f1_score(preds, targets, k=3)
        torch.testing.assert_close(result, torch.tensor(0.6000), atol=5e-5, rtol=0)


class TestHitRate:
    def test_k3_none(self, ranking_data):
        preds, targets = ranking_data
        result = hit_rate(preds, targets, k=3, reduction="none")
        torch.testing.assert_close(result, torch.tensor([1.0, 1.0]))

    def test_k3_mean(self, ranking_data):
        preds, targets = ranking_data
        result = hit_rate(preds, targets, k=3)
        torch.testing.assert_close(result, torch.tensor(1.0))


class TestNormalizedDCG:
    def test_k3_none(self, ranking_data):
        preds, targets = ranking_data
        result = normalized_dcg(preds, targets, k=3, reduction="none")
        torch.testing.assert_close(
            result, torch.tensor([0.6131, 0.6934]), atol=5e-5, rtol=0
        )

    def test_k3_mean(self, ranking_data):
        preds, targets = ranking_data
        result = normalized_dcg(preds, targets, k=3)
        torch.testing.assert_close(result, torch.tensor(0.6533), atol=5e-5, rtol=0)


class TestMeanReciprocalRank:
    def test_none(self, ranking_data):
        preds, targets = ranking_data
        result = mean_reciprocal_rank(preds, targets, reduction="none")
        torch.testing.assert_close(result, torch.tensor([1.0000, 0.5000]))

    def test_mean(self, ranking_data):
        preds, targets = ranking_data
        result = mean_reciprocal_rank(preds, targets)
        torch.testing.assert_close(result, torch.tensor(0.7500))


class TestMeanAveragePrecision:
    def test_none(self, ranking_data):
        preds, targets = ranking_data
        result = mean_average_precision(preds, targets, reduction="none")
        torch.testing.assert_close(
            result, torch.tensor([0.7500, 0.5833]), atol=5e-5, rtol=0
        )

    def test_mean(self, ranking_data):
        preds, targets = ranking_data
        result = mean_average_precision(preds, targets)
        torch.testing.assert_close(result, torch.tensor(0.6667), atol=5e-5, rtol=0)


# ---------------------------------------------------------------------------
# Ranking metrics — edge case: all-zero targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_fn", [recall, normalized_dcg, mean_reciprocal_rank, mean_average_precision]
)
def test_all_zero_targets(metric_fn):
    """Metrics should return 0 when targets are all zeros."""
    preds = torch.tensor([[0.5, 0.3, 0.2]])
    targets = torch.tensor([[0, 0, 0]])
    result = metric_fn(preds, targets, reduction="none")
    torch.testing.assert_close(result, torch.tensor([0.0]))


# ---------------------------------------------------------------------------
# Ranking metrics — with different k values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4])
def test_precision_various_k(k, ranking_data):
    preds, targets = ranking_data
    result = precision(preds, targets, k=k, reduction="none")
    assert result.shape == (2,)
    assert (result >= 0).all() and (result <= 1).all()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_recall_various_k(k, ranking_data):
    preds, targets = ranking_data
    result = recall(preds, targets, k=k, reduction="none")
    assert result.shape == (2,)
    assert (result >= 0).all() and (result <= 1).all()


# ---------------------------------------------------------------------------
# Classification metrics — log_loss
# ---------------------------------------------------------------------------


class TestLogLoss:
    @pytest.fixture
    def cls_data(self):
        preds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.75, 0.33]
        targets = [0, 1, 0, 1, 1, 0, 1, 1]
        return preds, targets

    def test_mean(self, cls_data):
        preds, targets = cls_data
        result = log_loss(preds, targets)
        np.testing.assert_allclose(result, 0.5804524803061438)

    def test_none(self, cls_data):
        preds, targets = cls_data
        result = log_loss(preds, targets, reduction="none")
        expected = np.array(
            [
                0.1053605,
                1.20397277,
                0.51082561,
                0.69314716,
                0.51082561,
                0.22314354,
                0.28768206,
                1.10866259,
            ]
        )
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sum(self, cls_data):
        preds, targets = cls_data
        result = log_loss(preds, targets, reduction="sum")
        expected_none = log_loss(preds, targets, reduction="none")
        np.testing.assert_allclose(result, expected_none.sum())


# ---------------------------------------------------------------------------
# Classification metrics — auroc
# ---------------------------------------------------------------------------


class TestAUROC:
    def test_docstring(self):
        preds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.75, 0.33]
        targets = [0, 1, 0, 1, 1, 0, 1, 1]
        result = auroc(preds, targets)
        np.testing.assert_allclose(result, 0.8666666666666667)

    def test_single_class_fallback(self):
        """auroc returns 1.0 on ValueError (older sklearn) or NaN (newer sklearn)."""
        preds = [0.1, 0.5, 0.9]
        targets = [1, 1, 1]
        result = auroc(preds, targets)
        # Older sklearn raises ValueError -> fallback returns 1.0
        # Newer sklearn (>=1.8) returns NaN with a warning instead
        assert result == 1.0 or np.isnan(result)


# ---------------------------------------------------------------------------
# Classification metrics — group_auroc
# ---------------------------------------------------------------------------


class TestGroupAUROC:
    @pytest.fixture
    def gauc_data(self):
        preds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.75, 0.33]
        targets = [0, 1, 0, 1, 1, 0, 1, 1]
        groups = [0, 0, 0, 0, 0, 1, 1, 1]
        return preds, targets, groups

    def test_mean(self, gauc_data):
        preds, targets, groups = gauc_data
        result = group_auroc(preds, targets, groups)
        np.testing.assert_allclose(result, 0.8958333333333333)

    def test_none(self, gauc_data):
        preds, targets, groups = gauc_data
        result = group_auroc(preds, targets, groups, reduction="none")
        expected = np.array([0.83333333, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-5)
