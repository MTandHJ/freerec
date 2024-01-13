

from .base import UserItemTimeTriplet


__all__ = [
    'MovieLens1M',
    'MovieLens100K_550_Chron', 'MovieLens100K_500_Chron',
    'MovieLens1M_550_Chron', 'MovieLens1M_500_Chron',
    'MovieLens10M_550_Chron', 'MovieLens10M_500_Chron',
]


class MovieLens1M(UserItemTimeTriplet):
    ...


class MovieLens100K_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-100K
    dataset: MovieLens100K
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1349 |     99287     | 97401  |  943   |  943  | 0.07804925214624242 |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    """
    URL = "https://zenodo.org/records/10499510/files/MovieLens100K_550_Chron.zip"


class MovieLens100K_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-100K
    dataset: MovieLens100K
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1682 |     100000    | 98114  |  943   |  943  | 0.06304669364224531 |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    """
    URL = "https://zenodo.org/records/10499510/files/MovieLens100K_500_Chron.zip"


class MovieLens1M_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-1m
    dataset: MovieLens1M
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  6040 |  3416 |     999611    | 987531 |  6040  |  6040 | 0.048448041549699894 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/MovieLens1M_550_Chron.zip"


class MovieLens1M_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-1m
    dataset: MovieLens1M
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  6040 |  3706 |    1000209    | 988129 |  6040  |  6040 | 0.044683625622312845 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/MovieLens1M_500_Chron.zip"


class MovieLens10M_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-10m
    dataset: MovieLens10M
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+-------+---------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+---------+--------+-------+---------------------+
    | 69878 | 10196 |    9998816    | 9859060 | 69878  | 69878 | 0.01403389695234235 |
    +-------+-------+---------------+---------+--------+-------+---------------------+
    """
    URL = "https://zenodo.org/records/10499510/files/MovieLens10M_550_Chron.zip"


class MovieLens10M_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: ml-10m
    dataset: MovieLens10M
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+-------+----------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+---------+--------+-------+----------------------+
    | 69878 | 10677 |    10000054   | 9860298 | 69878  | 69878 | 0.013403327706083809 |
    +-------+-------+---------------+---------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/records/10499510/files/MovieLens10M_500_Chron.zip"