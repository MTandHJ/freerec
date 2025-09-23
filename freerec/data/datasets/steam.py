

from .base import MatchingRecDataSet, NextItemRecDataSet


#===================================NextItemRecDataSset===================================


class Steam_550_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | 25389 |  4089 |     328278    | 277500 | 25389  | 25389 | 0.003162125283631449 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """