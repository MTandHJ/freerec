

from .base import UserItemPair, BARSUserItemPair


__all__ = ['Gowalla', 'Gowalla_m1']


class Gowalla(UserItemPair): ...


#======================================Chronological======================================


class GowallaChron811(UserItemPair): ...


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
