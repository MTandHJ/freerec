

from .base import MatchingRecDataSet, NextItemRecDataSet


#===================================MatchingRecDataSset===================================


class Yelp2018_10104811_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: 8,1,1
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 41801 | 26512 |    1022604    | 801621 | 98722  | 122261 | 0.0009227378271017908 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """


class Yelp2018_10104712_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: 7,1,2
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 41801 | 26512 |    1022604    | 698623 | 102998 | 220983 | 0.0009227378271017908 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """


class Yelp2021_10104811_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: 8,1,1
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+--------+-----------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+---------+--------+--------+-----------------------+
    | 78663 | 44534 |    2059757    | 1616979 | 199231 | 243547 | 0.0005879681178933354 |
    +-------+-------+---------------+---------+--------+--------+-----------------------+
    """


class Yelp2021_10104712_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 4
    ratios: 7,1,2
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+--------+-----------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+---------+--------+--------+-----------------------+
    | 78663 | 44534 |    2059757    | 1409167 | 207812 | 442778 | 0.0005879681178933354 |
    +-------+-------+---------------+---------+--------+--------+-----------------------+
    """


#===================================NextItemRecDataSset===================================


class Yelp2018_550_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +--------+-------+---------------+---------+--------+--------+------------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |        Density         |
    +--------+-------+---------------+---------+--------+--------+------------------------+
    | 213170 | 94304 |    3277932    | 2851592 | 213170 | 213170 | 0.00016305861179122036 |
    +--------+-------+---------------+---------+--------+--------+------------------------+
    """


class Yelp2018_500_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | 227110 | 167876 |    3515537    | 3061317 | 227110 | 227110 | 9.220761802399218e-05 |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    """


class Yelp2021_550_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +--------+--------+---------------+---------+--------+--------+------------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |        Density         |
    +--------+--------+---------------+---------+--------+--------+------------------------+
    | 356622 | 126449 |    5635863    | 4922619 | 356622 | 356622 | 0.00012497895730411994 |
    +--------+--------+---------------+---------+--------+--------+------------------------+
    """


class Yelp2021_500_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +--------+--------+---------------+---------+--------+--------+----------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |       Density        |
    +--------+--------+---------------+---------+--------+--------+----------------------+
    | 365665 | 159108 |    5766970    | 5035640 | 365665 | 365665 | 9.91225080273759e-05 |
    +--------+--------+---------------+---------+--------+--------+----------------------+
    """


class Yelp2019_550_S3Rec(NextItemRecDataSet):
    r"""
    Refer to [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py)
    Timestamp: 2019-01-01 00:00:00 - 2019-12-31 00:00:00
    Note that Yelp dataset is not deduplicated.

    Statistics:
    -----------
    +--------+--------+--------------------+---------------+--------+--------+-------+-----------------------+
    | #Users | #Items |      Avg.Len       | #Interactions | #Train | #Valid | #Test |        Density        |
    +--------+--------+--------------------+---------------+--------+--------+-------+-----------------------+
    | 30431  | 20033  | 10.395780618448294 |     316354    | 255492 | 30431  | 30431 | 0.0005189327918159183 |
    +--------+--------+--------------------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/records/11003210/files/Yelp2019_550_S3Rec.zip"