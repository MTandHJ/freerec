import pytest
import torch
import torch.nn.functional as F

from freerec.criterions import (
    BaseCriterion,
    BCELoss4Logits,
    BPRLoss,
    CrossEntropy4Logits,
    KLDivLoss4Logits,
    L1Loss,
    MSELoss,
    binary_cross_entropy_with_logits,
    bpr_loss_with_logits,
    cross_entropy_with_logits,
    kl_div_loss_with_logits,
)

# ---------------------------------------------------------------------------
# BaseCriterion.regularize
# ---------------------------------------------------------------------------


class TestRegularize:
    def test_l2(self):
        param = torch.tensor([1.0, 2.0, 3.0])
        expected = param.pow(2).sum() / 2
        result = BaseCriterion.regularize(param, rtype="l2")
        torch.testing.assert_close(result, expected)

    def test_l1(self):
        param = torch.tensor([-1.0, 2.0, -3.0])
        expected = param.abs().sum()
        result = BaseCriterion.regularize(param, rtype="l1")
        torch.testing.assert_close(result, expected)

    def test_multiple_params(self):
        params = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        expected = (1 + 4 + 9) / 2
        result = BaseCriterion.regularize(params, rtype="l2")
        torch.testing.assert_close(result, torch.tensor(expected))

    def test_unsupported_rtype(self):
        with pytest.raises(NotImplementedError):
            BaseCriterion.regularize(torch.tensor([1.0]), rtype="l3")


# ---------------------------------------------------------------------------
# CrossEntropy4Logits
# ---------------------------------------------------------------------------


class TestCrossEntropy:
    def test_matches_pytorch(self):
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        criterion = CrossEntropy4Logits()
        expected = F.cross_entropy(logits, targets, reduction="mean")
        torch.testing.assert_close(criterion(logits, targets), expected)

    def test_helper_function(self):
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        expected = F.cross_entropy(logits, targets, reduction="sum")
        result = cross_entropy_with_logits(logits, targets, reduction="sum")
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_reduction_modes(self, reduction):
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        criterion = CrossEntropy4Logits()
        expected = F.cross_entropy(logits, targets, reduction=reduction)
        result = criterion(logits, targets, reduction=reduction)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# BCELoss4Logits
# ---------------------------------------------------------------------------


class TestBCELoss:
    def test_matches_pytorch(self):
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 2, (4, 5)).float()
        criterion = BCELoss4Logits()
        expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        torch.testing.assert_close(criterion(logits, targets), expected)

    def test_helper_function(self):
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 2, (4, 5)).float()
        expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        result = binary_cross_entropy_with_logits(logits, targets)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# KLDivLoss4Logits
# ---------------------------------------------------------------------------


class TestKLDivLoss:
    def test_matches_pytorch(self):
        logits = torch.randn(4, 10)
        targets = torch.randn(4, 10)
        criterion = KLDivLoss4Logits()
        expected = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(targets, dim=-1),
            reduction="batchmean",
        )
        torch.testing.assert_close(criterion(logits, targets), expected)

    def test_helper_function(self):
        logits = torch.randn(4, 10)
        targets = torch.randn(4, 10)
        expected = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(targets, dim=-1),
            reduction="batchmean",
        )
        result = kl_div_loss_with_logits(logits, targets)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# BPRLoss
# ---------------------------------------------------------------------------


class TestBPRLoss:
    def test_positive_greater_gives_small_loss(self):
        pos = torch.tensor([5.0, 4.0, 3.0])
        neg = torch.tensor([1.0, 1.0, 1.0])
        loss = BPRLoss()(pos, neg)
        assert loss.item() < 1.0

    def test_negative_greater_gives_large_loss(self):
        pos = torch.tensor([1.0, 1.0])
        neg = torch.tensor([5.0, 5.0])
        loss = BPRLoss()(pos, neg)
        assert loss.item() > 1.0

    def test_formula(self):
        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([1.0, 0.5])
        expected = F.softplus(neg - pos).mean()
        result = BPRLoss()(pos, neg)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_reduction_modes(self, reduction):
        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([1.0, 0.5])
        raw = F.softplus(neg - pos)
        if reduction == "none":
            expected = raw
        elif reduction == "mean":
            expected = raw.mean()
        else:
            expected = raw.sum()
        result = bpr_loss_with_logits(pos, neg, reduction=reduction)
        torch.testing.assert_close(result, expected)

    def test_helper_matches_class(self):
        pos = torch.randn(8)
        neg = torch.randn(8)
        class_result = BPRLoss()(pos, neg)
        func_result = bpr_loss_with_logits(pos, neg)
        torch.testing.assert_close(class_result, func_result)


# ---------------------------------------------------------------------------
# MSELoss / L1Loss
# ---------------------------------------------------------------------------


class TestMSELoss:
    def test_matches_pytorch(self):
        inputs = torch.randn(4, 5)
        targets = torch.randn(4, 5)
        expected = F.mse_loss(inputs, targets, reduction="mean")
        result = MSELoss()(inputs, targets)
        torch.testing.assert_close(result, expected)


class TestL1Loss:
    def test_matches_pytorch(self):
        inputs = torch.randn(4, 5)
        targets = torch.randn(4, 5)
        expected = F.l1_loss(inputs, targets, reduction="mean")
        result = L1Loss()(inputs, targets)
        torch.testing.assert_close(result, expected)
