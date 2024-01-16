

from .base import UserItemTimeTriplet


__all__ = [
    'Yelp',
    'Yelp_550_Chron'
]


class Yelp(UserItemTimeTriplet): ...


class Yelp_550_Chron(UserItemTimeTriplet):
    r"""
    Chronologically-ordered Yelp-2019 dataset.
    Please refer to [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py)

    Config:
    -------
    filename: yelp
    dataset: Yelp
    kcore4user: 5
    kcore4item: 5
    star4pos: 0
    time: 2019-01-01 00:00:00 - 2019-12-31 00:00:00

    +--------+--------+--------------------+---------------+--------+--------+-------+-----------------------+
    | #Users | #Items |      Avg.Len       | #Interactions | #Train | #Valid | #Test |        Density        |
    +--------+--------+--------------------+---------------+--------+--------+-------+-----------------------+
    | 30431  | 20033  | 10.395780618448294 |     316354    | 255492 | 30431  | 30431 | 0.0005189327918159183 |
    +--------+--------+--------------------+---------------+--------+--------+-------+-----------------------+
    """
    DEDUPLICATED = False # Yelp dataset is not deduplicated.
    URL = "https://zenodo.org/records/10416596/files/Yelp_550_Chron.zip"