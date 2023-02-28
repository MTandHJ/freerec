

from .base import UserItemPair, BARSUserItemPair


__all__ = [
    'AmazonBooks', 'AmazonCDs', 'AmazonMovies', 'AmazonBeauty', 'AmazonElectronics',
    'AmazonBooks_m1', 'AmazonCDs_m1', 'AmazonMovies_m1', 'AmazonBeauty_m1', 'AmazonElectronics_m1'
]


class AmazonBooks(UserItemPair): ...
class AmazonCDs(UserItemPair): ...
class AmazonMovies(UserItemPair): ...
class AmazonBeauty(UserItemPair): ...
class AmazonElectronics(UserItemPair): ...

#======================================BARS======================================

class AmazonBooksChron811(UserItemPair): ...
class AmazonCDsChron811(UserItemPair): ...
class AmazonMoviesChron811(UserItemPair): ...
class AmazonBeautyChron811(UserItemPair): ...
class AmazonElectronicsChron811(UserItemPair): ...


#======================================BARS======================================


class AmazonBooks_m1(BARSUserItemPair):
    r"""
    Amazon-Books dataset.
    |    Dataset     | #Users | #Items | #Interactions |  #Train   |  #Test  | Density |
    |:--------------:|:------:|:------:|:-------------:|:---------:|:-------:|:-------:|
    | AmazonBooks_m1 | 52,643 | 91,599 |   2,984,108   | 2,380,730 | 603,378 | 0.00062 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets/Amazon#AmazonBooks_m1) for details.

    Attributes:
    -----------
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


class AmazonCDs_m1(BARSUserItemPair):
    r"""
    Amazon-CDs dataset.
    |   Dataset    | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    |:------------:|:------:|:------:|:-------------:|:-------:|:-------:|:-------:|
    | AmazonCDs_m1 | 43,169 | 35,648 |    777,426    | 604,475 | 172,951 | 0.00051 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attributes:
    -----------
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



class AmazonMovies_m1(BARSUserItemPair):
    r"""
    Amazon-Movies dataset.
    |     Dataset     | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    |:---------------:|:------:|:------:|:-------------:|:-------:|:-------:|:-------:|
    | AmazonMovies_m1 | 44,439 | 25,047 |   1,070,860   | 839,444 | 231,416 | 0.00096 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attributes:
    -----------
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


class AmazonBeauty_m1(BARSUserItemPair):
    r"""
    Amazon-Beauty dataset.
    |     Dataset     | #Users | #Items | #Interactions | #Train | #Test  | Density |
    |:---------------:|:------:|:------:|:-------------:|:------:|:------:|:-------:|
    | AmazonBeauty_m1 | 7,068  | 3,570  |    79,506     | 60,818 | 18,688 | 0.00315 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attributes:
    -----------
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


class AmazonElectronics_m1(BARSUserItemPair):
    r"""
    Amazon-Beauty dataset.
    |       Dataset        | #Users | #Items | #Interactions | #Train | #Test | Density |
    | :------------------: | :----: | :----: | :-----------: | :----: | :---: | :-----: |
    | AmazonElectronics_m1 | 1,435  | 1,522  |    35,931     | 31,887 | 4,044 | 0.01645 |
    See [here](https://github.com/xue-pai/UltraGCN/tree/main/data) for details.

    Attributes:
    -----------
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