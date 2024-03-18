

import os
from .base import TripletWithMeta
from ...utils import download_from_url



__all__ = [
    'Amazon2023',
    'Amazon2023_All_Beauty_550_Chron',
    'Amazon2023_Beauty_550_Chron', 'Amazon2023_Beauty_10100_Chron',
    'Amazon2023_Toys_550_Chron', 'Amazon2023_Toys_10100_Chron',
    'Amazon2023_Office_550_Chron',
    'Amazon2023_CDs_550_Chron', 'Amazon2023_CDs_10100_Chron',
]


class Amazon2023(TripletWithMeta):
    r"""
    Amazon datasets (2023).
    See [[here](https://amazon-reviews-2023.github.io/)] for more details.
    """

    def download_images(self, size: str = 'thumb') -> None:
        r"""
        Download images from urls.

        Parameters:
        -----------
        size: str, 'thumb', 'large' or 'hi_res'
        """
        from concurrent.futures import ThreadPoolExecutor
        assert size in ('thumb', 'large', 'hi_res'), f"`size` should be 'thumb', 'large', 'hi_res' ..."
        urls = self.fields['Images'].data
        urls = [eval(url)[0][size] for url in urls]
        with ThreadPoolExecutor() as executor:
            for id_, url in enumerate(urls):
                executor.submit(
                    download_from_url,
                    url=url,
                    root=os.path.join(self.path, f"item_{size}_images"),
                    filename=f"{id_}.jpg"
                )

class Amazon2023_All_Beauty_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon All Beauty dataset with meta data.

    Config:
    -------
    filename: All_Beauty
    dataset: Amazon2023_All_Beauty
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    | #Users | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test |       Density        |
    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    |  346   |  466   | 9.184971098265896 |      3178     |  2486  |  346   |  346  | 0.019710238408295912 |
    +--------+--------+-------------------+---------------+--------+--------+-------+----------------------+
    """

class Amazon2023_Beauty_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Beauty dataset with meta data.

    Config:
    -------
    filename: Beauty_and_Personal_Care
    dataset: Amazon2023_Beauty
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | 697966 | 253928 |    6340876    | 4944944 | 697966 | 697966 | 3.577703953833393e-05 |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    """


class Amazon2023_Beauty_10100_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Beauty dataset with meta data.

    Config:
    -------
    filename: Beauty_and_Personal_Care
    dataset: Amazon2023_Beauty
    by: leave-one-out
    kcore4user: 10
    kcore4item: 10
    star4pos: 0

    +-------+-------+---------------+---------+--------+-------+-----------------------+
    | #User | #Item | #Interactions |  #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+---------+--------+-------+-----------------------+
    | 73234 | 59208 |    1396703    | 1250235 | 73234  | 73234 | 0.0003221149776682619 |
    +-------+-------+---------------+---------+--------+-------+-----------------------+
    """


class Amazon2023_Toys_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Toys dataset with meta data.

    Config:
    -------
    filename: Toys_and_Games
    dataset: Amazon2023_Toys
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item  | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    | 427176 | 175207 |    3830356    | 2976004 | 427176 | 427176 | 5.117770914043441e-05 |
    +--------+--------+---------------+---------+--------+--------+-----------------------+
    """


class Amazon2023_Toys_10100_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Toys dataset with meta data.

    Config:
    -------
    filename: Toys_and_Games
    dataset: Amazon2023_Toys
    by: leave-one-out
    kcore4user: 10
    kcore4item: 10
    star4pos: 0

    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 35690 | 28529 |     611908    | 540528 | 35690  | 35690 | 0.0006009703339130787 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """


class Amazon2023_Office_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon Office dataset with meta data.

    Config:
    -------
    filename: Office_Products
    dataset: Amazon2023_Office
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+-------+---------------+---------+--------+--------+----------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |       Density        |
    +--------+-------+---------------+---------+--------+--------+----------------------+
    | 204681 | 85028 |    1648181    | 1238819 | 204681 | 204681 | 9.47033647237546e-05 |
    +--------+-------+---------------+---------+--------+--------+----------------------+
    """


class Amazon2023_CDs_550_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon CDs dataset with meta data.

    Config:
    -------
    filename: CDs_and_Vinyl
    dataset: Amazon2023_CDs
    by: leave-one-out
    kcore4user: 5
    kcore4item: 5
    star4pos: 0

    +--------+-------+---------------+---------+--------+--------+-----------------------+
    | #User  | #Item | #Interactions |  #Train | #Valid | #Test  |        Density        |
    +--------+-------+---------------+---------+--------+--------+-----------------------+
    | 126767 | 91509 |    1594801    | 1341267 | 126767 | 126767 | 0.0001374790356746118 |
    +--------+-------+---------------+---------+--------+--------+-----------------------+
    """


class Amazon2023_CDs_10100_Chron(Amazon2023):
    r"""
    Chronologically-ordered Amazon CDs dataset with meta data.

    Config:
    -------
    filename: CDs_and_Vinyl
    dataset: Amazon2023_CDs
    by: leave-one-out
    kcore4user: 10
    kcore4item: 10
    star4pos: 0

    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | #User | #Item | #Interactions | #Train | #Valid | #Test |        Density        |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    | 23059 | 20762 |     540118    | 494000 | 23059  | 23059 | 0.0011281815544690774 |
    +-------+-------+---------------+--------+--------+-------+-----------------------+
    """