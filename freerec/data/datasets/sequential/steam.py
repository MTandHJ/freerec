

from .base import UserItemTimeTriplet


__all__ = [
    'Steam',
    'Steam_550_Chron',
    'Steam_500_Chron'
]


class Steam(UserItemTimeTriplet): ...


class Steam_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: steam
    dataset: Steam
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    strict: False

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 26247 |  8479 |     340353    | 287863 | 26245  | 26245 | 0.0015293443271349354 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7684496/files/Steam_550_Chron.zip"


class Steam_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered MovieLens1M dataset.

    Config:
    -------
    filename: steam
    dataset: Steam
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    strict: False

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | 26247 |  9327 |     341270    | 288776 | 26247  | 26247 | 0.001394043945100003 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7684496/files/Steam_500_Chron.zip"

