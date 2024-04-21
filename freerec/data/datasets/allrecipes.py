

from .base import MatchingRecDataSet


class Allrecipes_MMSSL(MatchingRecDataSet):
    r"""
    Allrecipes dataset used in [MMSSL](https://github.com/HKUDS/MMSSL).

    Statistics:
    -----------
    +--------+--------+-------------------+---------------+--------+--------+-------+-----------------------+
    | #Users | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test |        Density        |
    +--------+--------+-------------------+---------------+--------+--------+-------+-----------------------+
    | 19805  | 10068  | 4.975107296137339 |     73494     | 58922  |  6751  |  7821 | 0.0003685817531420022 |
    +--------+--------+-------------------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/records/11003239/files/Allrecipes_MMSSL.zip"