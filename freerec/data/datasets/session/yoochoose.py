

from .base import SessionItemTimeTriplet


__all__ = [
    'YooChooseBuys', 'YooChooseClicks',
    'YooChooseClicks14_250811_Chron', 'YooChooseClicks164_250811_Chron',
    'YooChooseClicks14_250712_Chron', 'YooChooseClicks164_250712_Chron',
]


class YooChooseBuys(SessionItemTimeTriplet):
    ...


class YooChooseClicks(SessionItemTimeTriplet):
    ...


class YooChooseClicks14_250811_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered YooChooseClickss1/4 dataset.

    Config:
    -------
    filename: yoochoose-clicks
    dataset: YooChooseClicks14
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-----------+--------+-------------------+---------------+---------+--------+--------+------------------------+
    | #Sessions | #Items |      Avg.Len      | #Interactions |  #Train | #Valid | #Test  |        Density         |
    +-----------+--------+-------------------+---------------+---------+--------+--------+------------------------+
    |  1995340  | 30576  | 4.090095923501759 |    8161132    | 6524270 | 803053 | 833809 | 0.00013376818169485083 |
    +-----------+--------+-------------------+---------------+---------+--------+--------+------------------------+
    """
    URL = "https://zenodo.org/record/8062815/files/YooChooseClicks14_250811_Chron.zip"


class YooChooseClicks164_250811_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered YooChooseClickss1/64 dataset.

    Config:
    -------
    filename: yoochoose-clicks
    dataset: YooChooseClicks164
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-----------+--------+-------------------+---------------+--------+--------+-------+------------------------+
    | #Sessions | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test |        Density         |
    +-----------+--------+-------------------+---------------+--------+--------+-------+------------------------+
    |   124709  | 17567  | 4.243655229373982 |     529222    | 411917 | 57912  | 59393 | 0.00024156971761678047 |
    +-----------+--------+-------------------+---------------+--------+--------+-------+------------------------+
    """
    URL = "https://zenodo.org/record/8062815/files/YooChooseClicks164_250811_Chron.zip"


class YooChooseClicks14_250712_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered YooChooseClickss1/4 dataset.

    Config:
    -------
    filename: yoochoose-clicks
    dataset: YooChooseClicks14
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    ratios: (7, 1, 2)

    Statistics:
    -----------
    """
    URL = "https://zenodo.org/record/8062815/files/YooChooseClicks14_250712_Chron.zip"


class YooChooseClicks164_250712_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered YooChooseClickss1/64 dataset.

    Config:
    -------
    filename: yoochoose-clicks
    dataset: YooChooseClicks164
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    ratios: (7, 1, 2)

    Statistics:
    -----------
    """
    URL = "https://zenodo.org/record/8062815/files/YooChooseClicks164_250712_Chron.zip"


class YooChooseClicks14_2501_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered YooChooseClickss1/4 dataset.

    Config:
    -------
    filename: yoochoose-clicks
    dataset: YooChooseClicks14
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    days: 1

    Statistics:
    -----------
    """
    URL = "https://zenodo.org/record/8062815/files/YooChooseClicks14_2501_Chron.zip"


class YooChooseClicks164_2501_Chron(SessionItemTimeTriplet):
    r"""
    Chronologically-ordered YooChooseClickss1/64 dataset.

    Config:
    -------
    filename: yoochoose-clicks
    dataset: YooChooseClicks164
    kcore4user: 2
    kcore4item: 5
    star4pos: 0
    days: 1

    Statistics:
    -----------
    """
    URL = "https://zenodo.org/record/8062815/files/YooChooseClicks164_2501_Chron.zip"