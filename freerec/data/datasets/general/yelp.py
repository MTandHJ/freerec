

from .base import UserItemPair, BARSUserItemPair


__all__ = [
    'Yelp2018', 
    'Yelp2018_10104811_Chron',
    'Yelp2018_10104712_Chron',
    'Yelp18_m1'
]


class Yelp2018(UserItemPair): ...


#======================================Chronological======================================


class Yelp2018_10104811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Yelp2018 dataset.

    Config:
    -------
    filename: yelp2018
    dataset: Yelp2018
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (8, 1, 1)
    strict: True

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 41801 | 26512 |    1022604    | 801621 | 98722  | 122261 | 0.0009227378271017908 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/Yelp2018_10104811_Chron.zip"


class Yelp2018_10104712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Yelp2018 dataset.

    Config:
    -------
    filename: yelp2018
    dataset: Yelp2018
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: (7, 1, 2)
    strict: True

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 41801 | 26512 |    1022604    | 698623 | 102998 | 220983 | 0.0009227378271017908 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/Yelp2018_10104712_Chron.zip"


#======================================BARS======================================


class Yelp18_m1(BARSUserItemPair):
    r"""
    Yelp 2018 dataset.
    |  Dataset  | #Users | #Items | #Interactions |  #Train   |  #Test  | Density |
    |:---------:|:------:|:------:|:-------------:|:---------:|:-------:|:-------:|
    | Yelp18_m1 | 31,668 | 38,048 |   1,561,406   | 1,237,259 | 324,147 | 0.00130 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets/Yelp#Yelp18_m1) for details.

    Attributes:
    -----------
    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
    open_kw: Config
        - mode: 'rt'
        - delimiter: ' '
        - skip_lines: 0
    """

    URL = "https://zenodo.org/record/7297855/files/Yelp18_m1.zip"