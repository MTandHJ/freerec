

from .base import UserItemPair, BARSUserItemPair


__all__ = [
    'MovieLens100K', 'MovieLens1M', 
    'MovieLens100K_10101811_Chron', 'MovieLens1M_10101811_Chron',
    'MovieLens100K_10101712_Chron', 'MovieLens1M_10101712_Chron',
    'MovieLens1M_m2'
]


class MovieLens100K(UserItemPair): ...
class MovieLens1M(UserItemPair): ...


#======================================Chronological======================================


class MovieLens100K_10101811_Chron(UserItemPair):
    r"""
    Chronologically-ordered MovieLens100K dataset.

    Config:
    -------
    filename: ml-100k
    dataset: MovieLens100K
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1152 |     97953     | 77980  |  9744  | 10229 | 0.09016823524213503 |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/MovieLens100K_10101811_Chron.zip"


class MovieLens100K_10101712_Chron(UserItemPair):
    r"""
    Chronologically-ordered MovieLens100K dataset.

    Config:
    -------
    filename: ml-100k
    dataset: MovieLens100K
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1152 |     97953     | 68156  |  9824  | 19973 | 0.09016823524213503 |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/MovieLens100K_10101712_Chron.zip"


class MovieLens1M_10101811_Chron(UserItemPair):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-1m
    dataset: MovieLens1M
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |       Density       |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    |  6040 |  3260 |     998539    | 796389 | 99549  | 102601 | 0.05071197131597124 |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/MovieLens1M_10101811_Chron.zip"


class MovieLens1M_10101712_Chron(UserItemPair):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-1m
    dataset: MovieLens1M
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: (7, 1, 2)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |       Density       |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    |  6040 |  3260 |     998539    | 696267 | 100122 | 202150 | 0.05071197131597124 |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/MovieLens1M_10101712_Chron.zip"


#======================================BARS======================================


class MovieLens1M_m2(BARSUserItemPair):
    r"""
    MovieLens1M: (user, item, rating, timestamp)
    |    Dataset     | #Users | #Items | #Interactions | #Train  | #Test  | Density |
    | :------------: | :----: | :----: | :-----------: | :-----: | :----: | :-----: |
    | MovieLens1M_m2 | 6,022  | 3,043  |    895,699    | 796,244 | 99,455 | 0.04888 |
    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attritbutes:
    ------------
    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
    open_kw: Config
        - mode: 'rt'
        - delimiter: '\t'
        - skip_lines: 0
    """

    URL = "https://zenodo.org/record/7297855/files/MovieLens1M_m2.zip"