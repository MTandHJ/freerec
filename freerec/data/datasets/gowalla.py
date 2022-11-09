

import torchdata.datapipes as dp
import random

from .base import ImplicitRecSet
from ..fields import SparseField, DenseField
from ..tags import USER, ITEM, ID, FEATURE, TARGET
from ...dict2obj import Config


__all__ = ['Gowalla_m1']


class Gowalla_m1(ImplicitRecSet):
    """ GowallaM1: (user, items).
    |  Dataset   | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    | :--------: | :----: | :----: | :-----------: | :-----: | :-----: | :-----: |
    | Gowalla_m1 | 29,858 | 40,981 |   1,027,370   | 810,128 | 217,242 | 0.00084 |

    See [here](https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla) for details.

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

    URL = "https://zenodo.org/record/7184851/files/Gowalla_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse

    open_kw = Config(mode='rt', delimiter=' ', skip_lines=0)