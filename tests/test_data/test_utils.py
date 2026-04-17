import numpy as np
import pytest

from freerec.data.utils import (
    check_sha1,
    is_empty_dir,
    negsamp_vectorized_bsearch,
    safe_cast,
)


class TestSafeCast:
    def test_successful_cast(self):
        assert safe_cast("42", int, 0) == 42

    def test_failed_cast_raises(self):
        with pytest.raises(ValueError):
            safe_cast("abc", int, -1)

    def test_float_cast(self):
        assert safe_cast("3.14", float, 0.0) == pytest.approx(3.14)

    def test_identity(self):
        assert safe_cast(42, int, 0) == 42


class TestCheckSha1:
    def test_file_input(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello")
        result = check_sha1(str(f))
        assert isinstance(result, str)
        assert len(result) == 40

    def test_bytes_input(self):
        result = check_sha1(b"hello")
        assert isinstance(result, str)
        assert len(result) == 40

    def test_deterministic(self):
        assert check_sha1(b"test") == check_sha1(b"test")

    def test_different_inputs_differ(self):
        assert check_sha1(b"a") != check_sha1(b"b")


class TestIsEmptyDir:
    def test_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert is_empty_dir(str(empty_dir))

    def test_non_empty_dir(self, tmp_path):
        non_empty = tmp_path / "notempty"
        non_empty.mkdir()
        (non_empty / "file.txt").write_text("content")
        assert not is_empty_dir(str(non_empty))

    def test_nonexistent_dir(self, tmp_path):
        assert is_empty_dir(str(tmp_path / "nonexistent"))


class TestNegsampVectorizedBsearch:
    def test_basic_sampling(self):
        positives = np.array([2, 5, 8])
        result = negsamp_vectorized_bsearch(positives, n_items=10, size=5)
        assert len(result) == 5
        # No positive items in result
        for item in result:
            assert item not in positives

    def test_all_items_valid(self):
        positives = np.array([0, 1])
        result = negsamp_vectorized_bsearch(positives, n_items=10, size=8)
        assert len(result) == 8
        for item in result:
            assert 0 <= item < 10
            assert item not in positives

    def test_single_positive(self):
        positives = np.array([3])
        result = negsamp_vectorized_bsearch(positives, n_items=5, size=4)
        assert len(result) == 4
        assert 3 not in result
