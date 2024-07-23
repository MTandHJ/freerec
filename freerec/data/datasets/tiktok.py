

from .base import MatchingRecDataSet


class Tiktok_MMSSL(MatchingRecDataSet):
    r"""
    Tiktok dataset used in [MMSSL](https://github.com/HKUDS/MMSSL).

    Statistics:
    -----------
    +--------+--------+------------------+---------------+--------+--------+-------+-----------------------+
    | #Users | #Items |     Avg.Len      | #Interactions | #Train | #Valid | #Test |        Density        |
    +--------+--------+------------------+---------------+--------+--------+-------+-----------------------+
    |  9308  |  6710  | 8.40363519036352 |     68722     | 59541  |  3051  |  6130 | 0.0011003146500902705 |
    +--------+--------+------------------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/records/11003235/files/Tiktok_MMSSL.zip"


class Tiktok_000811_RAU(MatchingRecDataSet):
    r"""
    Tiktok dataset used in [MMGCN](https://github.com/weiyinwei/MMGCN).

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 36656 | 76085 |     726065    | 562209 | 75685  | 88171 | 0.0002604621965738634 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+

    Notes:
    Some of users have empty interactions.
    """
    URL = "https://zenodo.org/records/11003235/files/Tiktok_MMSSL.zip"