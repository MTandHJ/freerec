

from .base import ImplicitRecSet
from ..fields import SparseField
from ..tags import USER, ITEM, ID
from ...dict2obj import Config


__all__ = ['MovieLens1M_m2']


class MovieLens1M_m2(ImplicitRecSet):
    """ MovieLens1M: (user, item, rating, timestamp)
    |    Dataset     | #Users | #Items | #Interactions | #Train  | #Test  | Density |
    | :------------: | :----: | :----: | :-----------: | :-----: | :----: | :-----: |
    | MovieLens1M_m2 | 6,022  | 3,043  |    895,699    | 796,244 | 99,455 | 0.04888 |
    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attritbutes:
    ---

    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
        - dense: DenseField
            Timestamp
        - target: SparseField
            Rating
    open_kw: Config
        - mode: 'rt'
        - delimiter: '\\t'
        - skip_lines: 0

    """

    URL = "https://zenodo.org/record/7297855/files/MovieLens1M_m2.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=None, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=None, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse
