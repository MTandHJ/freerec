import torch

from freerec.utils import (
    AverageMeter,
    export_pickle,
    export_yaml,
    import_pickle,
    import_yaml,
    set_seed,
)

# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------


class TestAverageMeter:
    def _make_meter(self, name="test", best_caster=max):
        return AverageMeter(name=name, metric=lambda x: x, best_caster=best_caster)

    def test_initial_state(self):
        meter = self._make_meter()
        assert meter.avg == 0.0
        assert meter.count == 0
        assert meter.history == []

    def test_step_appends_to_history(self):
        meter = self._make_meter()
        meter.val = 3.0
        meter.sum = 3.0
        meter.count = 1
        meter.avg = 3.0
        meter.active = True
        meter.step()
        assert meter.history == [3.0]
        assert meter.avg == 0.0  # reset after step

    def test_reset(self):
        meter = self._make_meter()
        meter.val = 5.0
        meter.sum = 5.0
        meter.count = 1
        meter.avg = 5.0
        meter.reset()
        assert meter.avg == 0.0
        assert meter.count == 0

    def test_history_setter(self):
        meter = self._make_meter()
        meter.history = [1.0, 2.0, 3.0]
        assert meter.history == [1.0, 2.0, 3.0]

    def test_argbest_max(self):
        meter = self._make_meter(best_caster=max)
        meter.history = [0.1, 0.5, 0.3]
        idx, val = meter.argbest()
        assert idx == 1
        assert val == 0.5

    def test_argbest_min(self):
        meter = self._make_meter(best_caster=min)
        meter.history = [0.5, 0.1, 0.3]
        idx, val = meter.argbest()
        assert idx == 1
        assert val == 0.1

    def test_argbest_empty(self):
        meter = self._make_meter()
        idx, val = meter.argbest()
        assert idx == -1
        assert val == -1

    def test_which_is_better(self):
        meter = self._make_meter(best_caster=max)
        meter.avg = 0.8
        assert meter.which_is_better(0.5) == max(0.8, 0.5)

    def test_check_returns_value(self):
        meter = AverageMeter("m", metric=lambda x: x * 2)
        result = meter.check(3.0)
        assert result == 6.0

    def test_check_with_tensor(self):
        meter = AverageMeter("m", metric=lambda x: torch.tensor(x))
        result = meter.check(5.0)
        assert result == 5.0

    def test_str_format(self):
        meter = self._make_meter()
        meter.avg = 0.12345
        s = str(meter)
        assert "test" in s
        assert "0.12345" in s


# ---------------------------------------------------------------------------
# Pickle / YAML round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_pickle_round_trip(self, tmp_path):
        data = {"key": [1, 2, 3], "nested": {"a": 42}}
        path = str(tmp_path / "test.pkl")
        export_pickle(data, path)
        loaded = import_pickle(path)
        assert loaded == data

    def test_yaml_round_trip(self, tmp_path):
        data = {"name": "freerec", "version": "0.9.7", "items": [1, 2]}
        path = str(tmp_path / "test.yaml")
        export_yaml(data, path)
        loaded = import_yaml(path)
        assert loaded == data


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


class TestSetSeed:
    def test_returns_seed(self):
        result = set_seed(42)
        assert result == 42

    def test_reproducibility(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)
