

from .base import RecDataSet, MatchingRecDataSet, NextItemRecDataSet, PredictionRecDataSet

from .allrecipes import (
    # Matching (MMSSL)
    Allrecipes_MMSSL,
)

from .amazon2014 import (
    # Matching
    Amazon2014Beauty_10104811_ROU, Amazon2014Beauty_10104712_ROU,
    Amazon2014Books_10104811_ROU, Amazon2014Books_10104712_ROU,
    Amazon2014CDs_10104811_ROU, Amazon2014CDs_10104712_ROU,
    Amazon2014Electronics_10104811_ROU, Amazon2014Electronics_10104712_ROU,
    Amazon2014Movies_10104811_ROU, Amazon2014Movies_10104712_ROU,

    # Matching (MMRec)
    Amazon2014Baby_550_MMRec,
    Amazon2014Sports_550_MMRec,
    Amazon2014Clothing_550_MMRec,
    Amazon2014Electronics_550_MMRec,

    # NextItem
    Amazon2014Beauty_550_LOU, Amazon2014Beauty_500_LOU, Amazon2014Beauty_1000_LOU,
    Amazon2014CDs_1000_LOU,
    Amazon2014Clothing_1000_LOU,
    Amazon2014Games_550_LOU, Amazon2014Games_500_LOU,
    Amazon2014Grocery_550_LOU, Amazon2014Grocery_500_LOU,
    Amazon2014Home_550_LOU, Amazon2014Home_500_LOU, Amazon2014Home_1000_LOU,
    Amazon2014Movies_550_LOU, Amazon2014Movies_500_LOU, Amazon2014Movies_1000_LOU,
    Amazon2014Tools_550_LOU, Amazon2014Tools_500_LOU,
    Amazon2014Toys_550_LOU, Amazon2014Toys_500_LOU,
    Amazon2014Office_550_LOU, Amazon2014Office_500_LOU,
    Amazon2014Video_1000_LOU,
)

from .amazon2018 import (
    # NextItem
    Amazon2018Books_550_LOU,
    Amazon2018CDs_550_LOU,
    Amazon2018Clothing_550_LOU,
    Amazon2018Electronics_550_LOU,
    Amazon2018Movies_550_LOU,
    Amazon2018Office_550_LOU,
    Amazon2018Sports_550_LOU,
    Amazon2018Tools_550_LOU,
    Amazon2018Toys_550_LOU
)

from .amazon2023 import(
    # Matching
    Amazon2023Baby_554811_ROU,
    Amazon2023Beauty_554811_ROU, Amazon2023Beauty_10104811_ROU,
    Amazon2023Fashion_554811_ROU,
    Amazon2023Games_554811_ROU,
    Amazon2023Office_554811_ROU,
    Amazon2023Toys_554811_ROU, Amazon2023Toys_10104811_ROU
)

from .criteo import(
    # Prediction
    Criteo_x1_BARS
)

from .frappe import (
    # Prediction
    Frappe_x1_BARS
)

from .gowalla import (
    # Matching
    Gowalla2010_10100811_ROU, Gowalla2010_10100712_ROU
)

from .microlens import (
    # Matching
    MicroLens100K
)

from .movielens import (
    # Matching
    MovieLens100K_10101811_ROU, MovieLens100K_10101712_ROU,
    MovieLens1M_10101811_ROU, MovieLens1M_10101712_ROU,
    MovieLens10M_10101811_ROU, MovieLens10M_10101712_ROU,
    MovieLens20M_10101811_ROU, MovieLens20M_10101712_ROU,

    # NextItem
    MovieLens100K_550_LOU, MovieLens100K_500_LOU,
    MovieLens1M_550_LOU, MovieLens1M_500_LOU,
    MovieLens10M_550_LOU, MovieLens10M_500_LOU,
    MovieLens20M_550_LOU, MovieLens20M_500_LOU,
)

from .steam import (
    # NextItem
    Steam_550_LOU
)

from .tiktok import (
    # Matching (MMSSL)
    Tiktok_MMSSL, 
    # Matching (MMGCN)
    Tiktok_000811_RAU
)

from .yelp import (
    # Matching
    Yelp2018_10104811_ROU, Yelp2018_10104712_ROU, Yelp2018_554311_ROU,
    Yelp2021_10104811_ROU, Yelp2021_10104712_ROU,

    # NextItem
    Yelp2018_550_LOU, Yelp2018_500_LOU,
    Yelp2021_550_LOU, Yelp2021_500_LOU,

    # NextItem (S3Rec)
    Yelp2019_550_S3Rec,
)