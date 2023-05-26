

from .base import SessionItemTimeTriplet


__all__ = [
    'Diginetica',
    'Diginetica_250811_Chron',
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
    +-----------+--------+--------------------+---------------+--------+--------+-------+------------------------+
    | #Sessions | #Items |      Avg.Len       | #Interactions | #Train | #Valid | #Test |        Density         |
    +-----------+--------+--------------------+---------------+--------+--------+-------+------------------------+
    |   204061  | 42171  | 4.8475896913177925 |     989204    | 795142 | 96445  | 97617 | 0.00011495078825064125 |
    +-----------+--------+--------------------+---------------+--------+--------+-------+------------------------+
    """
