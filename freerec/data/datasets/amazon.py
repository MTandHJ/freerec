

import torchdata.datapipes as dp
import random

from .base import ImplicitRecSet
from ..fields import SparseField, DenseField
from ..tags import USER, ITEM, ID, FEATURE, TARGET
from ...dict2obj import Config


__all__ = ['AmazonBooks_m1', 'AmazonCDs_m1', 'AmazonMovies_m1', 'AmazonBeauty_m1', 'AmazonElectronics_m1']





class AmazonBooks_m1(ImplicitRecSet):
    """Amazon-Books dataset.
    |    Dataset     | #Users | #Items | #Interactions |  #Train   |  #Test  | Density |
    |:--------------:|:------:|:------:|:-------------:|:---------:|:-------:|:-------:|
    | AmazonBooks_m1 | 52,643 | 91,599 |   2,984,108   | 2,380,730 | 603,378 | 0.00062 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets/Amazon#AmazonBooks_m1) for details.

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

    URL = "https://zenodo.org/record/7297855/files/AmazonBooks_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ],
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer='none', tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target


class AmazonCDs_m1(ImplicitRecSet):
    """Amazon-CDs dataset.
    |   Dataset    | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    |:------------:|:------:|:------:|:-------------:|:-------:|:-------:|:-------:|
    | AmazonCDs_m1 | 43,169 | 35,648 |    777,426    | 604,475 | 172,951 | 0.00051 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

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

    URL = "https://zenodo.org/record/7297855/files/AmazonCDs_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse


class AmazonMovies_m1(ImplicitRecSet):
    """Amazon-Movies dataset.
    |     Dataset     | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    |:---------------:|:------:|:------:|:-------------:|:-------:|:-------:|:-------:|
    | AmazonMovies_m1 | 44,439 | 25,047 |   1,070,860   | 839,444 | 231,416 | 0.00096 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

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

    URL = "https://zenodo.org/record/7297855/files/AmazonMovies_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse


class AmazonBeauty_m1(ImplicitRecSet):
    """Amazon-Beauty dataset.
    |     Dataset     | #Users | #Items | #Interactions | #Train | #Test  | Density |
    |:---------------:|:------:|:------:|:-------------:|:------:|:------:|:-------:|
    | AmazonBeauty_m1 | 7,068  | 3,570  |    79,506     | 60,818 | 18,688 | 0.00315 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

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

    URL = "https://zenodo.org/record/7297855/files/AmazonBeauty_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse


class AmazonElectronics_m1(ImplicitRecSet):
    """Amazon-Beauty dataset.
    |       Dataset        | #Users | #Items | #Interactions | #Train | #Test | Density |
    | :------------------: | :----: | :----: | :-----------: | :----: | :---: | :-----: |
    | AmazonElectronics_m1 | 1,435  | 1,522  |    35,931     | 31,887 | 4,044 | 0.01645 |
    See [here](https://github.com/xue-pai/UltraGCN/tree/main/data) for details.

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

    URL = "https://zenodo.org/record/7297855/files/AmazonElectronics_m1.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=-1, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=-1, dtype=int, tags=[ITEM, ID]),
        ]
    )

    _cfg.fields = _cfg.sparse