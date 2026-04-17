import polars as pl

from freerec.data.normalizer import (
    Counter,
    MinMaxScaler,
    Normalizer,
    ReIndexer,
    StandardScaler,
)


class TestNormalizer:
    """Base normalizer should be an identity transform."""

    def test_normalize_is_identity(self):
        n = Normalizer()
        s = pl.Series("x", [1, 2, 3])
        result = n.normalize(s)
        assert result.to_list() == [1, 2, 3]


class TestCounter:
    def test_partial_fit_counts_unique(self):
        c = Counter()
        c.partial_fit(pl.Series("x", [1, 2, 3, 1, 2]))
        assert c.count == 3

    def test_partial_fit_accumulates(self):
        c = Counter()
        c.partial_fit(pl.Series("x", [1, 2]))
        c.partial_fit(pl.Series("x", [3, 4]))
        assert c.count == 4

    def test_reset(self):
        c = Counter()
        c.partial_fit(pl.Series("x", [1, 2, 3]))
        c.reset()
        assert c.count == 0


class TestReIndexer:
    def test_normalize_maps_to_contiguous(self):
        r = ReIndexer()
        r.partial_fit(pl.Series("x", [10, 20, 30, 10]))
        result = r.normalize(pl.Series("x", [10, 20, 30]))
        values = result.to_list()
        assert sorted(values) == [0, 1, 2]

    def test_count(self):
        r = ReIndexer()
        r.partial_fit(pl.Series("x", [5, 10, 15]))
        assert r.count == 3


class TestStandardScaler:
    def test_mean(self):
        ss = StandardScaler()
        ss.partial_fit(pl.Series("x", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
        assert abs(ss.mean - 5.0) < 1e-6

    def test_normalize(self):
        ss = StandardScaler()
        ss.partial_fit(pl.Series("x", [0.0, 10.0]))
        result = ss.normalize(pl.Series("x", [5.0]))
        # mean=5, (5-5)/std ≈ 0
        assert abs(result.to_list()[0]) < 1e-6

    def test_nums_tracks_count(self):
        ss = StandardScaler()
        ss.partial_fit(pl.Series("x", [1.0, 2.0, 3.0]))
        assert ss.nums == 3


class TestMinMaxScaler:
    def test_normalize_to_01(self):
        mm = MinMaxScaler()
        mm.partial_fit(pl.Series("x", [0.0, 10.0]))
        result = mm.normalize(pl.Series("x", [0.0, 5.0, 10.0]))
        values = result.to_list()
        assert abs(values[0] - 0.0) < 1e-6
        assert abs(values[1] - 0.5) < 1e-6
        assert abs(values[2] - 1.0) < 1e-6

    def test_partial_fit_accumulates(self):
        mm = MinMaxScaler()
        mm.partial_fit(pl.Series("x", [5.0, 10.0]))
        mm.partial_fit(pl.Series("x", [0.0, 15.0]))
        assert mm.min == 0.0
        assert mm.max == 15.0
