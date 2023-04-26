

from .base import UserItemTimeTriplet


__all__ = [
    'AmazonBeauty', 'AmazonGames', 'AmazonHome', 'AmazonOffice', 'AmazonTools', 'AmazonToys',
    'AmazonBeauty_550_Chron', 'AmazonGames_550_Chron', 'AmazonHome_550_Chron', 'AmazonOffice_550_Chron', 'AmazonTools_550_Chron', 'AmazonToys_550_Chron',
    'AmazonBeauty_500_Chron', 'AmazonGames_500_Chron', 'AmazonHome_500_Chron', 'AmazonOffice_500_Chron', 'AmazonTools_500_Chron', 'AmazonToys_500_Chron',
]


class AmazonBeauty(UserItemTimeTriplet): ...
class AmazonGames(UserItemTimeTriplet): ...
class AmazonHome(UserItemTimeTriplet): ...
class AmazonOffice(UserItemTimeTriplet): ...
class AmazonTools(UserItemTimeTriplet): ...
class AmazonToys(UserItemTimeTriplet): ...


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

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 22363 | 12101 |     198502    | 153776 | 22363  | 22363 | 0.0007335227064174272 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonBeauty_550_Chron.zip"


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

    Statistics:
    -----------
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item  | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    | 52374 | 121291 |     469771    | 365023 | 52374  | 52374 | 7.395063077984393e-05 |
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonBeauty_500_Chron.zip"


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

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 24303 | 10672 |     231780    | 183174 | 24303  | 24303 | 0.0008936557520523778 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonGames_550_Chron.zip"


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

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 31027 | 33899 |     300003    | 237949 | 31027  | 31027 | 0.0002852324451898322 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonGames_500_Chron.zip"


class AmazonHome_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Home dataset.

    Config:
    -------
    filename: Amazon_Home_and_Kitchen
    dataset: AmazonHome
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 66519 | 28237 |     551682    | 418644 | 66519  | 66519 | 0.0002937139329503578 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonHome_550_Chron.zip"


class AmazonHome_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Home dataset.

    Config:
    -------
    filename: Amazon_Home_and_Kitchen
    dataset: AmazonHome
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +--------+--------+---------------+--------+--------+--------+-----------------------+
    | #User  | #Item  | #Interactions | #Train | #Valid | #Test  |        Density        |
    +--------+--------+---------------+--------+--------+--------+-----------------------+
    | 115673 | 194446 |     980608    | 749262 | 115673 | 115673 | 4.359778437881578e-05 |
    +--------+--------+---------------+--------+--------+--------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonHome_500_Chron.zip"


class AmazonOffice_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Office dataset.

    Config:
    -------
    filename: Amazon_Office_Products
    dataset: AmazonOffice
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |       Density        |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    |  4905 |  2420 |     53258     | 43448  |  4905  |  4905 | 0.004486735579312727 |
    +-------+-------+---------------+--------+--------+-------+----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonOffice_550_Chron.zip"


class AmazonOffice_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Office dataset.

    Config:
    -------
    filename: Amazon_Office_Products
    dataset: AmazonOffice
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+------------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density         |
    +-------+-------+---------------+--------+--------+-------+------------------------+
    | 16772 | 37956 |     145141    | 111597 | 16772  | 16772 | 0.00022799473561677615 |
    +-------+-------+---------------+--------+--------+-------+------------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonOffice_500_Chron.zip"


class AmazonTools_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Tools dataset.

    Config:
    -------
    filename: Amazon_Tools_and_Home_Improvement
    dataset: AmazonTools
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 16638 | 10217 |     134476    | 101200 | 16638  | 16638 | 0.0007910797527997544 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonTools_550_Chron.zip"


class AmazonTools_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Tools dataset.

    Config:
    -------
    filename: Amazon_Tools_and_Home_Improvement
    dataset: AmazonTools
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item  | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    | 45791 | 112855 |     390454    | 298872 | 45791  | 45791 | 7.555599745486971e-05 |
    +-------+--------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonTools_500_Chron.zip"


class AmazonToys_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Toys dataset.

    Config:
    -------
    filename: Amazon_Toys_and_Games
    dataset: AmazonToys
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 19412 | 11924 |     167597    | 128773 | 19412  | 19412 | 0.0007240590571801228 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonToys_550_Chron.zip"


class AmazonToys_500_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Amazon Toys dataset.

    Config:
    -------
    filename: Amazon_Toys_and_Games
    dataset: AmazonToys
    kcore4user: 5
    kcore4item: 0
    star4pos: 0

    Statistics:
    -----------
    +-------+--------+---------------+--------+--------+-------+------------------------+
    | #User | #Item  | #Interactions | #Train | #Valid | #Test |        Density         |
    +-------+--------+---------------+--------+--------+-------+------------------------+
    | 58315 | 165371 |     525535    | 408905 | 58315  | 58315 | 5.4495671989733694e-05 |
    +-------+--------+---------------+--------+--------+-------+------------------------+
    """
    URL = "https://zenodo.org/record/7866203/files/AmazonToys_500_Chron.zip"

