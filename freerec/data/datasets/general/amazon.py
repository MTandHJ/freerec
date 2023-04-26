

from .base import UserItemPair, BARSUserItemPair


__all__ = [
    'AmazonBooks', 'AmazonCDs', 'AmazonMovies', 'AmazonBeauty', 'AmazonElectronics',
    'AmazonBooks_10104811_Chron', 'AmazonCDs_10104811_Chron', 'AmazonMovies_10104811_Chron', 'AmazonBeauty_10104811_Chron', 'AmazonElectronics_10104811_Chron',
    'AmazonBooks_10104712_Chron', 'AmazonCDs_10104712_Chron', 'AmazonMovies_10104712_Chron', 'AmazonBeauty_10104712_Chron', 'AmazonElectronics_10104712_Chron',
    'AmazonBooks_m1', 'AmazonCDs_m1', 'AmazonMovies_m1', 'AmazonBeauty_m1', 'AmazonElectronics_m1'
]


class AmazonBooks(UserItemPair): ...
class AmazonCDs(UserItemPair): ...
class AmazonMovies(UserItemPair): ...
class AmazonBeauty(UserItemPair): ...
class AmazonElectronics(UserItemPair): ...


#======================================Chronological======================================


class AmazonBooks_10104811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Books dataset.

    Config:
    -------
    filename: Amazon_Books
    dataset: AmazonBooks
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +--------+-------+---------------+---------+--------+--------+------------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |        Density         |
    +--------+-------+---------------+---------+--------+--------+------------------------+
    | 109730 | 96421 |    3181759    | 2502323 | 308885 | 370551 | 0.00030072551044609626 |
    +--------+-------+---------------+---------+--------+--------+------------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonBooks_10104811_Chron.zip"


class AmazonBooks_10104712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Books dataset.

    Config:
    -------
    filename: Amazon_Books
    dataset: AmazonBooks
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (7, 1, 2)

    Statistics:
    -----------
    +--------+-------+---------------+---------+--------+--------+------------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |        Density         |
    +--------+-------+---------------+---------+--------+--------+------------------------+
    | 109730 | 96421 |    3181759    | 2181889 | 320434 | 679436 | 0.00030072551044609626 |
    +--------+-------+---------------+---------+--------+--------+------------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonBooks_10104712_Chron.zip"


class AmazonCDs_10104811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon CDs dataset.

    Config:
    -------
    filename: Amazon_CDs_and_Vinyl
    dataset: AmazonCDs
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  9954 | 10877 |     272086    | 213755 | 26342  | 31989 | 0.002513040172344499 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonCDs_10104811_Chron.zip"


class AmazonCDs_10104712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon CDs dataset.

    Config:
    -------
    filename: Amazon_CDs_and_Vinyl
    dataset: AmazonCDs
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (7, 1, 2)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  9954 | 10877 |     272086    | 186347 | 27408  | 58331 | 0.002513040172344499 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonCDs_10104712_Chron.zip"


class AmazonMovies_10104811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Movies dataset.

    Config:
    -------
    filename: Amazon_Movies_and_TV
    dataset: AmazonMovies
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 21506 | 15269 |     580420    | 455956 | 56162  | 68302 | 0.0017675517274430242 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonMovies_10104811_Chron.zip"


class AmazonMovies_10104712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Movies dataset.

    Config:
    -------
    filename: Amazon_Movies_and_TV
    dataset: AmazonMovies
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (7, 1, 2)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 21506 | 15269 |     580420    | 397557 | 58399  | 124464 | 0.0017675517274430242 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonMovies_10104712_Chron.zip"


class AmazonBeauty_10104811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Beauty dataset.

    Config:
    -------
    filename: Amazon_Beauty
    dataset: AmazonBeauty
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  973  |  645  |     19263     | 15008  |  1845  |  2410 | 0.030693850235426277 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonBeauty_10104811_Chron.zip"


class AmazonBeauty_10104712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Beauty dataset.

    Config:
    -------
    filename: Amazon_Beauty
    dataset: AmazonBeauty
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  973  |  645  |     19263     | 13063  |  1945  |  4255 | 0.030693850235426277 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonBeauty_10104712_Chron.zip"


class AmazonElectronics_10104811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Electronics dataset.

    Config:
    -------
    filename: Amazon_Electronics
    dataset: AmazonElectronics
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  9279 |  6065 |     158979    | 123648 | 14988  | 20343 | 0.002824930586818252 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonElectronics_10104811_Chron.zip"


class AmazonElectronics_10104712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Amazon Electronics dataset.

    Config:
    -------
    filename: Amazon_Electronics
    dataset: AmazonElectronics
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (7, 1, 2)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  9279 |  6065 |     158979    | 107674 | 15974  | 35331 | 0.002824930586818252 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/AmazonElectronics_10104712_Chron.zip"


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