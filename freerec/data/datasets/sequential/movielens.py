

from .base import UserItemTimeTriplet


__all__ = [
    'MovieLens1M',
    'MovieLens1M_550_Chron',
    'MovieLens1M_500_Chron',
]


class MovieLens1M(UserItemTimeTriplet):
    ...


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

