import sys
from pathlib import Path

import pandas as pd
import pytest

from freerec.data.preprocessing.base import AtomicConverter


def write_atomic(root: Path, filedir: str, inter: pd.DataFrame, item: pd.DataFrame = None) -> None:
    path = root / filedir
    path.mkdir(parents=True)
    inter.to_csv(path / f"{filedir}.inter", sep="\t", index=False)
    if item is not None:
        item.to_csv(path / f"{filedir}.item", sep="\t", index=False)


def read_split_size(root: Path, dataset: str, code: str) -> int:
    path = root / "Processed" / f"{dataset}_{code}"
    return sum(
        pd.read_csv(path / f"{mode}.txt", sep="\t").shape[0] for mode in ("train", "valid", "test")
    )


def base_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "USER": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5"],
            "ITEM": ["i1", "i2", "i1", "i3", "i2", "i3", "i_missing", "i1", "i2", "i_missing"],
            "RATING": [5] * 10,
            "TIMESTAMP": list(range(1, 11)),
        }
    )


def item_file() -> pd.DataFrame:
    return pd.DataFrame({"ITEM": ["i1", "i2", "i3"]})


class TestAtomicConverterItemFileMatch:
    def test_filter_inter_by_item_file(self, tmp_path):
        write_atomic(tmp_path, "Toy", base_interactions(), item_file())

        converter = AtomicConverter(root=str(tmp_path), dataset="Toy", filedir="Toy")
        converter.load()
        converter.filter_inter_by_item_file()

        assert "i_missing" not in set(converter.interactions["ITEM"])
        assert set(converter.interactions["ITEM"]) == {"i1", "i2", "i3"}
        assert len(converter.interactions) == 8

    def test_make_dataset_filters_missing_items(self, tmp_path):
        write_atomic(tmp_path, "Toy", base_interactions(), item_file())

        converter = AtomicConverter(root=str(tmp_path), dataset="Toy", filedir="Toy")
        converter.make_dataset(
            kcore4user=0,
            kcore4item=0,
            star4pos=0,
            splitting="ROD",
            ratios=(8, 1, 1),
            match_item_file=True,
        )

        assert "i_missing" not in converter.itemMaps
        assert read_split_size(tmp_path, "Toy", "000811_ROD") == 8

    def test_make_dataset_keeps_default_behavior(self, tmp_path):
        write_atomic(tmp_path, "Toy", base_interactions(), item_file())

        converter = AtomicConverter(root=str(tmp_path), dataset="Toy", filedir="Toy")
        converter.make_dataset(
            kcore4user=0,
            kcore4item=0,
            star4pos=0,
            splitting="ROD",
            ratios=(8, 1, 1),
            match_item_file=False,
        )

        assert "i_missing" in converter.itemMaps
        assert read_split_size(tmp_path, "Toy", "000811_ROD") == 10

    def test_match_item_file_skips_without_item_file(self, tmp_path, capsys):
        write_atomic(tmp_path, "Toy", base_interactions())

        converter = AtomicConverter(root=str(tmp_path), dataset="Toy", filedir="Toy")
        converter.make_dataset(
            kcore4user=0,
            kcore4item=0,
            star4pos=0,
            splitting="ROD",
            ratios=(8, 1, 1),
            match_item_file=True,
        )

        output = capsys.readouterr().out
        assert "Skip `filter_inter_by_item_file`" in output
        assert read_split_size(tmp_path, "Toy", "000811_ROD") == 10

    def test_match_item_file_respects_item_colname(self, tmp_path):
        inter = base_interactions().rename(columns={"ITEM": "BOOK"})
        items = item_file().rename(columns={"ITEM": "BOOK"})
        write_atomic(tmp_path, "Toy", inter, items)

        converter = AtomicConverter(
            root=str(tmp_path), dataset="Toy", filedir="Toy", itemColname="BOOK"
        )
        converter.load()
        converter.filter_inter_by_item_file()

        assert "i_missing" not in set(converter.interactions["ITEM"])
        assert set(converter.interactions["ITEM"]) == {"i1", "i2", "i3"}

    def test_match_item_file_runs_before_kcore(self, tmp_path):
        inter = pd.DataFrame(
            {
                "USER": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
                "ITEM": ["i1", "i_missing", "i1", "i2", "i2", "i3", "i3", "i_missing"],
                "RATING": [5] * 8,
                "TIMESTAMP": list(range(1, 9)),
            }
        )
        write_atomic(tmp_path, "Toy", inter, item_file())

        converter = AtomicConverter(root=str(tmp_path), dataset="Toy", filedir="Toy")
        converter.make_dataset(
            kcore4user=2,
            kcore4item=0,
            star4pos=0,
            splitting="ROD",
            ratios=(2, 1, 1),
            match_item_file=True,
        )

        assert read_split_size(tmp_path, "Toy", "200211_ROD") == 4


class TestMakeCliItemFileMatch:
    def test_make_help_lists_match_item_file(self, monkeypatch, capsys):
        import freerec.__main__ as cli

        monkeypatch.setattr(sys, "argv", ["freerec", "make", "--help"])
        with pytest.raises(SystemExit) as exc:
            cli.main()

        output = capsys.readouterr().out
        assert exc.value.code == 0
        assert "-mif" in output
        assert "--match-item-file" in output

    def test_make_parses_match_item_file(self, monkeypatch):
        import freerec.__main__ as cli

        captured = {}

        def fake_make(args):
            captured["match_item_file"] = args.match_item_file

        monkeypatch.setattr(cli, "make", fake_make)
        monkeypatch.setattr(sys, "argv", ["freerec", "make", "Toy", "--match-item-file"])

        cli.main()

        assert captured["match_item_file"] is True
