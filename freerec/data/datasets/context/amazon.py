

from .base import TripletWithMeta


class Amazon2023_All_Beauty_550_Chron(TripletWithMeta):
    r"""
    Chronologically-ordered Amazon Books dataset with meta data.

    Config:
    -------
    filename: All_Beauty
    dataset: Amazon2023_All_Beauty
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    | #Users | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test |       Density        |
    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    |  346   |  466   | 9.184971098265896 |      3178     |  2486  |  346   |  346  | 0.019710238408295912 |
    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    """