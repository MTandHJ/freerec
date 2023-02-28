

from .base import UserItemPair, BARSUserItemPair


__all__ = [
    'Gowalla', 
    'Gowalla_10100811_Chron',
    'Gowalla_10100712_Chron',
    'Gowalla_m1'
]


class Gowalla(UserItemPair): ...


#======================================Chronological======================================


class Gowalla_10100811_Chron(UserItemPair):
    r"""
    Chronologically-ordered Gowalla dataset.

    Config:
    -------
    filename: gowalla
    dataset: Gowalla
    kcore4user: 10
    kcore4item: 10
    star4pos: 0
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 29858 | 40988 |    1027464    | 810128 | 100508 | 116828 | 0.0008395550395550749 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/Gowalla_10100811_Chron.zip"


class Gowalla_10100712_Chron(UserItemPair):
    r"""
    Chronologically-ordered Gowalla dataset.

    Config:
    -------
    filename: gowalla
    dataset: Gowalla
    kcore4user: 10
    kcore4item: 10
    star4pos: 0
    ratios: (8, 1, 1)

    Statistics:
    -----------
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test  |        Density        |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    | 29858 | 40988 |    1027464    | 706338 | 103790 | 217336 | 0.0008395550395550749 |
    +-------+-------+---------------+--------+--------+--------+-----------------------+
    """
    URL = "https://zenodo.org/record/7683693/files/Gowalla_10100712_Chron.zip"


#======================================BARS======================================


class Gowalla_m1(BARSUserItemPair):
    r""" 
    GowallaM1: (user, items).
    |  Dataset   | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    | :--------: | :----: | :----: | :-----------: | :-----: | :-----: | :-----: |
    | Gowalla_m1 | 29,858 | 40,981 |   1,027,370   | 810,128 | 217,242 | 0.00084 |

    See [here](https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla) for details.

    Attributes:
    -----------
    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
        - target: SparseField
            Rating
    open_kw: Config
        - mode: 'rt'
        - delimiter: ' '
        - skip_lines: 0
    """

    URL = "https://zenodo.org/record/7184851/files/Gowalla_m1.zip"
