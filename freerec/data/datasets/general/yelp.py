

from .base import UserItemPair, BARSUserItemPair


__all__ = ['Yelp2018', 'Yelp18_m1']


class Yelp2018(UserItemPair): ...


#======================================Chronological======================================


class Yelp2018Chron811(UserItemPair): ...


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