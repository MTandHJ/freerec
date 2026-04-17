import pytest

from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import FEATURE, ID, ITEM, SEQUENCE, USER


@pytest.fixture
def user_field():
    f = Field("user", USER, ID)
    f.count = 10
    return f


@pytest.fixture
def item_field():
    f = Field("item", ITEM, ID)
    f.count = 20
    return f


class TestField:
    def test_attributes(self, user_field):
        assert user_field.name == "user"
        assert user_field.count == 10

    def test_match(self, user_field):
        assert user_field.match(USER)
        assert user_field.match(ID)
        assert user_field.match(USER, ID)
        assert not user_field.match(ITEM)

    def test_match_all(self, user_field):
        assert user_field.match_all(USER, ID)
        assert not user_field.match_all(USER, ITEM)

    def test_match_any(self, user_field):
        assert user_field.match_any(USER, ITEM)
        assert not user_field.match_any(ITEM, FEATURE)

    def test_fork(self, item_field):
        seq_field = item_field.fork(SEQUENCE)
        assert seq_field.match(ITEM, ID, SEQUENCE)
        assert seq_field.name == "item"
        assert seq_field.count == item_field.count
        assert seq_field is not item_field

    def test_hash_and_equality(self, user_field):
        assert user_field == user_field
        other = Field("user", USER, ID)
        assert user_field is not other

    def test_hashable(self, user_field, item_field):
        d = {user_field: "u", item_field: "i"}
        assert d[user_field] == "u"
        assert d[item_field] == "i"


class TestFieldTuple:
    def test_indexing(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        assert ft[0] is user_field
        assert ft[1] is item_field
        assert len(ft) == 2

    def test_match(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        result = ft.match(USER, ID)
        assert len(result) >= 1
        assert user_field in result

    def test_match_returns_empty(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        result = ft.match(FEATURE)
        assert len(result) == 0

    def test_match_any(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        result = ft.match_any(USER, ITEM)
        assert len(result) == 2

    def test_match_not(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        result = ft.match_not(USER)
        assert len(result) == 1
        assert result[0] is item_field

    def test_tag_based_indexing(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        result = ft[USER, ID]
        assert result is user_field

    def test_copy(self, user_field, item_field):
        ft = FieldTuple([user_field, item_field])
        ft2 = ft.copy()
        assert len(ft2) == len(ft)
