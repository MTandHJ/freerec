

from .base import MatchingRecDataSet, NextItemRecDataSet


#===================================MatchingRecDataSset===================================


class MovieLens100K_10101811_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 8,1,1
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1152 |     97953     | 77980  |  9744  | 10229 | 0.09016823524213503 |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    """


class MovieLens100K_10101712_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 7,1,2
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1152 |     97953     | 68156  |  9824  | 19973 | 0.09016823524213503 |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    """


class MovieLens1M_10101811_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 8,1,1
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |       Density       |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    |  6040 |  3260 |     998539    | 796389 | 99549  | 102601 | 0.05071197131597124 |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    """


class MovieLens1M_10101712_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 7,1,2
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |       Density       |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    |  6040 |  3260 |     998539    | 696267 | 100122 | 202150 | 0.05071197131597124 |
    +-------+-------+---------------+--------+--------+--------+---------------------+
    """


class MovieLens10M_10101811_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 8,1,1
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+---------+--------------------+
    | #User | #Item | #Interactions |  #Train | #Valid |  #Test  |      Density       |
    +-------+-------+---------------+---------+--------+---------+--------------------+
    | 69862 |  9673 |    9900255    | 7892719 | 985971 | 1021565 | 0.0146502210855272 |
    +-------+-------+---------------+---------+--------+---------+--------------------+
    """


class MovieLens10M_10101712_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 7,1,2
    splitting: ROU

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+---------+--------------------+
    | #User | #Item | #Interactions |  #Train | #Valid |  #Test  |      Density       |
    +-------+-------+---------------+---------+--------+---------+--------------------+
    | 69862 |  9673 |    9900255    | 6899556 | 993163 | 2007536 | 0.0146502210855272 |
    +-------+-------+---------------+---------+--------+---------+--------------------+
    """


class MovieLens20M_10101811_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 8,1,1
    splitting: ROU

    Statistics:
    -----------
    +--------+-------+---------------+----------+---------+---------+----------------------+
    | #User  | #Item | #Interactions |  #Train  |  #Valid |  #Test  |       Density        |
    +--------+-------+---------------+----------+---------+---------+----------------------+
    | 138425 | 15329 |    19725787   | 15726067 | 1964709 | 2035011 | 0.009296211221662753 |
    +--------+-------+---------------+----------+---------+---------+----------------------+
    """


class MovieLens20M_10101712_ROU(MatchingRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 10
    kcore4item: 10
    star4pos: 1
    ratios: 7,1,2
    splitting: ROU

    Statistics:
    -----------
    +--------+-------+---------------+----------+---------+---------+----------------------+
    | #User  | #Item | #Interactions |  #Train  |  #Valid |  #Test  |       Density        |
    +--------+-------+---------------+----------+---------+---------+----------------------+
    | 138425 | 15329 |    19725787   | 13747362 | 1978705 | 3999720 | 0.009296211221662753 |
    +--------+-------+---------------+----------+---------+---------+----------------------+
    """


#===================================NextItemRecDataSset===================================


class MovieLens100K_550_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1349 |     99287     | 97401  |  943   |  943  | 0.07804925214624242 |
    +-------+-------+---------------+--------+--------+-------+---------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens100K_500_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+---------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+--------+--------+-------+---------------------+
    |  943  |  1682 |     100000    | 98114  |  943   |  943  | 0.06304669364224531 |
    +-------+-------+---------------+--------+--------+-------+---------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens1M_550_LOU(NextItemRecDataSet):
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
    |  6040 |  3416 |     999611    | 987531 |  6040  |  6040 | 0.048448041549699894 |
    +-------+-------+---------------+--------+--------+-------+----------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens1M_500_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  6040 |  3706 |    1000209    | 988129 |  6040  |  6040 | 0.044683625622312845 |
    +-------+-------+---------------+--------+--------+-------+----------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens10M_550_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+-------+---------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test |       Density       |
    +-------+-------+---------------+---------+--------+-------+---------------------+
    | 69878 | 10196 |    9998816    | 9859060 | 69878  | 69878 | 0.01403389695234235 |
    +-------+-------+---------------+---------+--------+-------+---------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens10M_500_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +-------+-------+---------------+---------+--------+-------+----------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+---------+--------+-------+----------------------+
    | 69878 | 10677 |    10000054   | 9860298 | 69878  | 69878 | 0.013403327706083809 |
    +-------+-------+---------------+---------+--------+-------+----------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens20M_550_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +--------+-------+---------------+----------+--------+--------+----------------------+
    | #User  | #Item | #Interactions |  #Train  | #Valid | #Test  |       Density        |
    +--------+-------+---------------+----------+--------+--------+----------------------+
    | 138493 | 18345 |    19984024   | 19707038 | 138493 | 138493 | 0.007865700457998398 |
    +--------+-------+---------------+----------+--------+--------+----------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """


class MovieLens20M_500_LOU(NextItemRecDataSet):
    r"""
    Settings:
    ---------
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    splitting: LOU

    Statistics:
    -----------
    +--------+-------+---------------+----------+--------+--------+-----------------------+
    | #User  | #Item | #Interactions |  #Train  | #Valid | #Test  |        Density        |
    +--------+-------+---------------+----------+--------+--------+-----------------------+
    | 138493 | 26744 |    20000263   | 19723277 | 138493 | 138493 | 0.0053998478135544505 |
    +--------+-------+---------------+----------+--------+--------+-----------------------+

    Notes:
    MovieLens is not suitable for next-item recommendation.
    Refer to [this paper](https://ceur-ws.org/Vol-2955/paper8.pdf) for details.
    """