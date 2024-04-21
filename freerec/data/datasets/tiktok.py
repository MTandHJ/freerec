

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