

from .base import UserItemTimeTriplet


__all__ = [
    'AmazonBeauty', 'AmazonGames',
    'AmazonBeauty_550_Chron', 'AmazonGames_550_Chron',
    'AmazonBeauty_500_Chron', 'AmazonGames_500_Chron',
]


class AmazonBeauty(UserItemTimeTriplet): ...
class AmazonGames(UserItemTimeTriplet): ...


class AmazonBeauty_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Beauty dataset.

    Config:
    -------
    filename: Amazon_Beauty
    dataset: AmazonBeauty
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    strict: False

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+------------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density         |
    +-------+-------+---------------+--------+--------+-------+------------------------+
    | 52204 | 57289 |     394908    | 300100 | 47404  | 47404 | 0.00013204468022194222 |
    +-------+-------+---------------+--------+--------+-------+------------------------+
    """
    URL = "https://zenodo.org/record/7684496/files/AmazonBeauty_550_Chron.zip"


class AmazonBeauty_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Beauty dataset.

    Config:
    -------
    filename: Amazon_Beauty
    dataset: AmazonBeauty
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    strict: False

    Statistics:
    -----------
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item  | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    | 52374 | 121291 |     469771    | 365023 | 52374  | 52374 | 7.395063077984393e-05 |
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7684496/files/AmazonBeauty_500_Chron.zip"


class AmazonGames_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Video Games dataset.

    Config:
    -------
    filename: Amazon_Video_Games
    dataset: AmazonGames
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    strict: False

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 31013 | 23715 |     287107    | 225849 | 30629  | 30629 | 0.0003903703940739276 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7684496/files/AmazonGames_550_Chron.zip"


class AmazonGames_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Video Games dataset.

    Config:
    -------
    filename: Amazon_Video_Games
    dataset: AmazonGames
    kcore4user: 5
    kcore4item: 0
    star4pos: 0
    strict: False

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 31027 | 33899 |     300003    | 237949 | 31027  | 31027 | 0.0002852324451898322 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7684496/files/AmazonGames_500_Chron.zip"

