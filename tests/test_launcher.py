import json

import pytest
import torch

from freerec.dict2obj import Config
from freerec.launcher import Adapter
from freerec.utils import (
    AverageMeter,
    export_pickle,
    export_yaml,
    import_pickle,
    import_yaml,
    set_seed,
)


class DummySummaryWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def add_hparams(self, params, metrics):
        self.params = params
        self.metrics = metrics


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
# Adapter results.json
# ---------------------------------------------------------------------------


class TestAdapterGrid:
    def make_adapter(self):
        adapter = Adapter()
        adapter.cfg = Config(DEFAULTS=Config(config="configs/base.yaml"))
        return adapter

    def test_product_grid_uses_cartesian_product(self):
        adapter = self.make_adapter()
        adapter.deploy_params("a", [1, 2])
        adapter.deploy_params("b", [3, 4])

        assert list(adapter.product_grid()) == [
            {"config": "configs/base.yaml", "a": 1, "b": 3},
            {"config": "configs/base.yaml", "a": 1, "b": 4},
            {"config": "configs/base.yaml", "a": 2, "b": 3},
            {"config": "configs/base.yaml", "a": 2, "b": 4},
        ]

    def test_zip_grid_combines_values_by_position(self):
        adapter = self.make_adapter()
        adapter.deploy_params("a", [1, 2, 3])
        adapter.deploy_params("b", [3, 4, 5])

        assert list(adapter.zip_grid()) == [
            {"config": "configs/base.yaml", "a": 1, "b": 3},
            {"config": "configs/base.yaml", "a": 2, "b": 4},
            {"config": "configs/base.yaml", "a": 3, "b": 5},
        ]

    def test_zip_grid_requires_same_lengths(self):
        adapter = self.make_adapter()
        adapter.deploy_params("a", [1, 2])
        adapter.deploy_params("b", [3, 4, 5])

        with pytest.raises(ValueError, match="same length"):
            list(adapter.zip_grid())


class TestAdapterResults:
    def make_adapter(self, core_log_path):
        adapter = Adapter()
        adapter.cfg = Config(
            CORE_LOG_PATH=str(core_log_path),
            DATA_DIR="data",
            MONITOR_BEST_FILENAME="best.pkl",
            CONFIG_FILENAME="config.json",
            RESULTS_FILENAME="results.json",
            ENVS=Config(
                description="eval_BERT4Rec",
                dataset="Amazon2014Beauty_550_LOU",
            ),
        )
        return adapter

    def test_write_includes_runtime_config_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("freerec.launcher.SummaryWriter", DummySummaryWriter)

        core_log_path = tmp_path / "logs" / "eval_BERT4Rec" / "core"
        run_log_path = (
            tmp_path / "logs" / "eval_BERT4Rec" / "Amazon2014Beauty_550_LOU" / "0426200647"
        )
        (run_log_path / "data").mkdir(parents=True)
        core_log_path.mkdir(parents=True)

        export_pickle(
            {
                "train": {"LOSS": 6.001080085781005},
                "valid": {"NDCG@10": 0.05495695602421134},
                "test": {"NDCG@10": 0.040461423737339484},
                "best": {"NDCG@10": 0.04153840716245088},
            },
            str(run_log_path / "data" / "best.pkl"),
        )
        with open(run_log_path / "config.json", "w", encoding="utf8") as fh:
            json.dump(
                {
                    "SAVED_FILENAME": "model.pt",
                    "BEST_FILENAME": "best.pt",
                    "CHECKPOINT_FILENAME": "checkpoint.tar",
                    "CONFIG_FILENAME": "config.json",
                    "DATA_DIR": "data",
                    "SUMMARY_DIR": "summary",
                    "CHECKPOINT_PATH": "./infos/eval_BERT4Rec/Amazon2014Beauty_550_LOU/1",
                    "LOG_PATH": "./logs/eval_BERT4Rec/Amazon2014Beauty_550_LOU/0426200647",
                    "dataset": "Amazon2014Beauty_550_LOU",
                    "description": "eval_BERT4Rec",
                    "id": "0426200647",
                    "config": "configs/Amazon2014Beauty_550_LOU.yaml",
                    "seed": 4,
                    "eval_valid": True,
                    "eval_test": False,
                    "tasktag": None,
                    "device": "cuda:1",
                    "ddp_backend": "nccl",
                    "num_workers": 4,
                    "benchmark": False,
                    "resume": False,
                    "log2console": True,
                    "log2file": True,
                    "monitors": ["LOSS", "NDCG@10"],
                    "embedding_dim": 64,
                    "lr": 0.005,
                },
                fh,
            )

        adapter = self.make_adapter(core_log_path)
        adapter.write(
            "0426200647",
            str(run_log_path),
            {"config": "configs/Amazon2014Beauty_550_LOU.yaml", "seed": 4},
        )

        with open(core_log_path / "results.json", "r", encoding="utf8") as fh:
            results = json.load(fh)

        assert results["description"] == "eval_BERT4Rec"
        assert results["dataset"] == "Amazon2014Beauty_550_LOU"
        assert results["config"]["seed"] == 4
        assert results["config"]["eval_valid"] is True
        assert results["config"]["eval_test"] is False
        assert results["config"]["tasktag"] is None
        assert results["config"]["monitors"] == ["LOSS", "NDCG@10"]
        assert results["config"]["lr"] == 0.005
        assert "SAVED_FILENAME" not in results["config"]
        assert "CHECKPOINT_PATH" not in results["config"]
        assert "LOG_PATH" not in results["config"]
        assert "description" not in results["config"]
        assert "id" not in results["config"]
        assert "device" not in results["config"]
        assert "ddp_backend" not in results["config"]
        assert "num_workers" not in results["config"]
        assert "benchmark" not in results["config"]
        assert "resume" not in results["config"]
        assert "log2console" not in results["config"]
        assert "log2file" not in results["config"]
        assert results["runs"] == [
            {
                "id": "0426200647",
                "params": {
                    "config": "configs/Amazon2014Beauty_550_LOU.yaml",
                    "seed": 4,
                },
                "metrics": {
                    "train": {"LOSS": 6.001080085781005},
                    "valid": {"NDCG@10": 0.05495695602421134},
                    "test": {"NDCG@10": 0.040461423737339484},
                    "best": {"NDCG@10": 0.04153840716245088},
                },
            }
        ]

    def test_write_uses_empty_config_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("freerec.launcher.SummaryWriter", DummySummaryWriter)

        core_log_path = tmp_path / "logs" / "eval_BERT4Rec" / "core"
        run_log_path = (
            tmp_path / "logs" / "eval_BERT4Rec" / "Amazon2014Beauty_550_LOU" / "0426200648"
        )
        (run_log_path / "data").mkdir(parents=True)
        core_log_path.mkdir(parents=True)
        export_pickle(
            {"valid": {"NDCG@10": 0.1}},
            str(run_log_path / "data" / "best.pkl"),
        )

        adapter = self.make_adapter(core_log_path)
        adapter.write("0426200648", str(run_log_path), {"seed": 5})

        with open(core_log_path / "results.json", "r", encoding="utf8") as fh:
            results = json.load(fh)

        assert results["config"] == {}
        assert results["runs"][0]["params"] == {"seed": 5}


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
