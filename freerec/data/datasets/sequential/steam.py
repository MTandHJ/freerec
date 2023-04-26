

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

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | 25389 |  4089 |     328278    | 277500 | 25389  | 25389 | 0.003162125283631449 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/Steam_550_Chron.zip"


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

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | 26247 |  9327 |     341270    | 288776 | 26247  | 26247 | 0.001394043945100003 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/Steam_500_Chron.zip"

