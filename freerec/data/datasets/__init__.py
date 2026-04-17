from .allrecipes import (
    # Matching (MMSSL)
    Allrecipes_MMSSL,
)
from .amazon2014 import (
    # Matching (MMRec)
    Amazon2014Baby_550_MMRec,
    Amazon2014Beauty_500_LOU,
    # NextItem
    Amazon2014Beauty_550_LOU,
    Amazon2014Beauty_1000_LOU,
    Amazon2014Beauty_10104712_ROU,
    # Matching
    Amazon2014Beauty_10104811_ROU,
    Amazon2014Books_10104712_ROU,
    Amazon2014Books_10104811_ROU,
    Amazon2014CDs_1000_LOU,
    Amazon2014CDs_10104712_ROU,
    Amazon2014CDs_10104811_ROU,
    Amazon2014Clothing_550_MMRec,
    Amazon2014Clothing_1000_LOU,
    Amazon2014Electronics_550_MMRec,
    Amazon2014Electronics_10104712_ROU,
    Amazon2014Electronics_10104811_ROU,
    Amazon2014Games_500_LOU,
    Amazon2014Games_550_LOU,
    Amazon2014Grocery_500_LOU,
    Amazon2014Grocery_550_LOU,
    Amazon2014Home_500_LOU,
    Amazon2014Home_550_LOU,
    Amazon2014Home_1000_LOU,
    Amazon2014Movies_500_LOU,
    Amazon2014Movies_550_LOU,
    Amazon2014Movies_1000_LOU,
    Amazon2014Movies_10104712_ROU,
    Amazon2014Movies_10104811_ROU,
    Amazon2014Office_500_LOU,
    Amazon2014Office_550_LOU,
    Amazon2014Sports_550_MMRec,
    Amazon2014Tools_500_LOU,
    Amazon2014Tools_550_LOU,
    Amazon2014Toys_500_LOU,
    Amazon2014Toys_550_LOU,
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
    Amazon2018Toys_550_LOU,
)
from .amazon2023 import (
    # Matching
    Amazon2023Baby_554811_ROU,
    Amazon2023Beauty_554811_ROU,
    Amazon2023Beauty_10104811_ROU,
    Amazon2023Fashion_554811_ROU,
    Amazon2023Games_554811_ROU,
    Amazon2023Office_554811_ROU,
    Amazon2023Toys_554811_ROU,
    Amazon2023Toys_10104811_ROU,
)
from .base import (
    MatchingRecDataSet,
    NextItemRecDataSet,
    PredictionRecDataSet,
    RecDataSet,
)
from .criteo import Criteo_x1_BARS  # Prediction
from .frappe import Frappe_x1_BARS  # Prediction
from .gowalla import (
    Gowalla2010_10100712_ROU,
    # Matching
    Gowalla2010_10100811_ROU,
)
from .microlens import MicroLens100K  # Matching
from .movielens import (
    MovieLens1M_500_LOU,
    MovieLens1M_550_LOU,
    MovieLens1M_10101712_ROU,
    MovieLens1M_10101811_ROU,
    MovieLens10M_500_LOU,
    MovieLens10M_550_LOU,
    MovieLens10M_10101712_ROU,
    MovieLens10M_10101811_ROU,
    MovieLens20M_500_LOU,
    MovieLens20M_550_LOU,
    MovieLens20M_10101712_ROU,
    MovieLens20M_10101811_ROU,
    MovieLens100K_500_LOU,
    # NextItem
    MovieLens100K_550_LOU,
    MovieLens100K_10101712_ROU,
    # Matching
    MovieLens100K_10101811_ROU,
)
from .steam import Steam_550_LOU  # NextItem
from .tiktok import (
    # Matching (MMGCN)
    Tiktok_000811_RAU,
    # Matching (MMSSL)
    Tiktok_MMSSL,
)
from .yelp import (
    Yelp2018_500_LOU,
    # NextItem
    Yelp2018_550_LOU,
    Yelp2018_554311_ROU,
    Yelp2018_10104712_ROU,
    # Matching
    Yelp2018_10104811_ROU,
    # NextItem (S3Rec)
    Yelp2019_550_S3Rec,
    Yelp2021_500_LOU,
    Yelp2021_550_LOU,
    Yelp2021_10104712_ROU,
    Yelp2021_10104811_ROU,
)
