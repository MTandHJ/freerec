

from .base import ImplicitRecSet
from ..fields import SparseField
from ..tags import USER, ITEM, ID
from ...dict2obj import Config


__all__ = ['Yelp18_m1']


class Yelp18_m1(ImplicitRecSet):
    """ Yelp 2018 dataset.
    |  Dataset  | #Users | #Items | #Interactions |  #Train   |  #Test  | Density |
    |:---------:|:------:|:------:|:-------------:|:---------:|:-------:|:-------:|
    | Yelp18_m1 | 31,668 | 38,048 |   1,561,406   | 1,237,259 | 324,147 | 0.00130 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets/Yelp#Yelp18_m1) for details.

    Attributes:
    ---

    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
        - target: SparseField
            Rating
    open_kw: Config
        - mode: 'rt'
        - delimiter: ' '
        - skip_lines: 0

    """

    URL = "https://zenodo.org/record/7297855/files/Yelp18_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse