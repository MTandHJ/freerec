

from .base import MatchingRecDataSet


class MicroLens100K(MatchingRecDataSet):
    r"""
    See [[here](https://github.com/westlake-repl/MicroLens)] for details.

    Note: 100K indicates 100,000 users.

    Statistics:
    -----------
    +--------+--------+-------------------+---------------+--------+--------+--------+-----------------------+
    | #Users | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test  |        Density        |
    +--------+--------+-------------------+---------------+--------+--------+--------+-----------------------+
    | 98129  | 17228  | 7.186193683824353 |     705174    | 500064 | 101121 | 103989 | 0.0004171229210485462 |
    +--------+--------+-------------------+---------------+--------+--------+--------+-----------------------+
    """