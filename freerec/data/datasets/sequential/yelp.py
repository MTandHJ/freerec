

from .base import UserItemTimeTriplet


__all__ = [
    'Yelp',
    'Yelp_550_Chron'
]


class Yelp(UserItemTimeTriplet): ...


class Yelp_550_Chron(UserItemTimeTriplet):
    r"""
    +--------+--------+--------------------+---------------+--------+--------+-------+------------------------+
    | #Users | #Items |      Avg.Len       | #Interactions | #Train | #Valid | #Test |        Density         |
    +--------+--------+--------------------+---------------+--------+--------+-------+------------------------+
    | 65039  | 38861  | 12.284475468565015 |     798970    | 668892 | 65039  | 65039 | 0.00031611321037968695 |
    +--------+--------+--------------------+---------------+--------+--------+-------+------------------------+
    """