

from .base import SessionItemTimeTriplet


__all__ = [
    'Diginetica',
    'Diginetica_250811_Chron',
    'Diginetica_250712_Chron',
    'Diginetica_2507_Chron',
]


class Diginetica(SessionItemTimeTriplet):
    ...


class Diginetica_250811_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered Diginetica dataset.

    Config:
    -------
    filename: diginetica
    dataset: Diginetica
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-----------+--------+--------------------+---------------+--------+--------+--------+------------------------+
    | #Sessions | #Items |      Avg.Len       | #Interactions | #Train | #Valid | #Test  |        Density         |
    +-----------+--------+--------------------+---------------+--------+--------+--------+------------------------+
    |   204061  | 42171  | 4.8475896913177925 |     989204    | 777448 | 99310  | 112446 | 0.00011495078825064125 |
    +-----------+--------+--------------------+---------------+--------+--------+--------+------------------------+
    """
    URL = "https://zenodo.org/record/8062815/files/Diginetica_250811_Chron.zip"


class Diginetica_250712_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered Diginetica dataset.

    Config:
    -------
    filename: diginetica
    dataset: Diginetica
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    ratios: (7, 1, 2)

    Statistics:
    -----------
    """
    URL = "https://zenodo.org/record/8062815/files/Diginetica_250712_Chron.zip"


class Diginetica_2507_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered Diginetica dataset.

    Config:
    -------
    filename: diginetica
    dataset: Diginetica
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    days: 7

    Statistics:
    -----------
    """
    URL = "https://zenodo.org/record/8062815/files/Diginetica_2507_Chron.zip"