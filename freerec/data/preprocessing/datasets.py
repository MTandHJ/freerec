

from typing import Iterable

import numpy as np
from math import floor, ceil

from .base import AtomicConverter
from ..tags import USER, SESSION, ITEM, RATING, TIMESTAMP
from ...utils import infoLogger


class AmazonBeauty(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	title:token	    sales_type:token	sales_rank:float	categories:token_seq	price:float 	brand:token
    """
    filename = "Amazon_Beauty"


class AmazonBooks(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	sales_type:token	sales_rank:float	categories:token_seq	title:token     price:float    brand:token
    """
    filename = "Amazon_Books"


class AmazonCDs(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	title:token	    categories:token_seq	brand:token	sales_type:token	sales_rank:float	price:float
    """
    filename = "Amazon_CDs_and_Vinyl\Amazon_CDs_and_Vinyl"


class AmazonElectronics(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	categories:token_seq	title:token 	price:float	    sales_type:token	sales_rank:float	brand:token
    """
    filename = "Amazon_Electronics"


class AmazonHome(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	sales_type:token	sales_rank:float	categories:token_seq	title:token	price:float	brand:token
    """
    filename = "Amazon_Home_and_Kitchen"


class AmazonMovies(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	categories:token_seq	title:token	    price:float	    sales_type:token	sales_rank:float	brand:token
    """
    filename = "Amazon_Movies_and_TV"


class AmazonOffice(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	price:float	sales_type:token	sales_rank:float	categories:token_seq	title:token	brand:token
    """
    filename = "Amazon_Office_Products"


class AmazonTools(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	categories:token_seq	title:token	price:float	brand:token	sales_type:token	sales_rank:float
    """
    filename = "Amazon_Tools_and_Home_Improvement"


class AmazonToys(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	title:token	price:float	sales_type:token	sales_rank:float	brand:token	categories:token_seq
    """
    filename = "Amazon_Toys_and_Games"


class AmazonGames(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	price:float	sales_type:token	sales_rank:float	categories:token_seq	title:token	brand:token
    """
    filename = "Amazon_Video_Games"


class Amazon2023_All_Beauty(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float	parent_asin:token
    item:
        item_id:token	parent_asin:token ...
    """
    filename = "All_Beauty"


class Amazon2023_Beauty(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float	parent_asin:token
    item:
        item_id:token	parent_asin:token ...
    """
    filename = "Beauty"


class Amazon2023_CDs(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float	parent_asin:token
    item:
        item_id:token	parent_asin:token ...
    """
    filename = "CDs"


class Amazon2023_Office(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float	parent_asin:token
    item:
        item_id:token	parent_asin:token ...
    """
    filename = "Office"


class Amazon2023_Toys(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float	parent_asin:token
    item:
        item_id:token	parent_asin:token ...
    """
    filename = "Toys"


class Amazon2023_Tools(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float	parent_asin:token
    item:
        item_id:token	parent_asin:token ...
    """
    filename = "Tools"


class Diginetica(AtomicConverter):
    r"""
    inter:
        session_id:token	item_id:token	timestamp:float
    item:
        item_id:token	item_priceLog2:float	item_name:token	item_category:token
    """
    filename = "diginetica"


class FourSquareNYC(AtomicConverter):
    r"""
    inter:
        user_id:token	venue_id:token	timezone_offset:float	timestamp:float	    click_times:float
    item:
        venue_id:token	venue_category_id:token	    venue_category_name:token	latitude:float	  longitude:float
    """
    filename = "foursquare_NYC"
    _name_format_dict = {'venue_id': ITEM.name}


class FourSquareTKY(AtomicConverter):
    r"""
    inter:
        user_id:token	venue_id:token	timezone_offset:float	timestamp:float	    click_times:float
    item:
        venue_id:token	venue_category_id:token	    venue_category_name:token	latitude:float	  longitude:float
    """
    filename = "foursquare_TKY"


class Gowalla(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	timestamp:float	    latitude:float	longitude:float	    num_repeat:float
    """
    filename = "gowalla"
    _name_format_dict = {'venue_id': ITEM.name}


class LastFM(AtomicConverter):
    r"""
    inter:
        user_id:token	artist_id:token	    weight:float	tag_value:token_seq
    item:
        artist_id:token	    name:token	    url:token	    picture_url:token
    """
    filename = "lastfm"


class MovieLens1M(AtomicConverter):
    r"""
    inter: 
        user_id:token	item_id:token	rating:float	timestamp:float
    user: 
        user_id:token	age:token	gender:token	occupation:token	zip_code:token
    item:
        item_id:token	movie_title:token_seq	release_year:token	class:token_seq
    """
    filename = "ml-1m"


class MovieLens10M(AtomicConverter):
    r"""
    inter: 
        user_id:token	item_id:token	rating:float	timestamp:float
    user: 
        user_id:token	age:token	gender:token	occupation:token	zip_code:token
    item:
        item_id:token	movie_title:token_seq	release_year:token	class:token_seq
    """
    filename = "ml-10m"


class MovieLens100K(AtomicConverter):
    r"""
    inter: 
        user_id:token	item_id:token	rating:float	timestamp:float
    user: 
        user_id:token	age:token	gender:token	occupation:token	zip_code:token
    item:
        item_id:token	movie_title:token_seq	release_year:token	class:token_seq
    """
    filename = "ml-100K"


class Steam(AtomicConverter):
    r"""
    inter: 
        user_id:token	play_hours:float	products:float	    product_id:token	page_order:float	timestamp:float	    early_access:token	    page:float	    found_funny:token	compensation:token	    times:float
    item:
        app_name:token	developer:token	early_access:token	genres:token_seq	id:token	metascore:float	    price:float	    publisher:token	    timestamp:float	    sentiment:token	    specs:token_seq	    tags:token_seq	    title:token
    """
    filename = "steam\steam"
    _name_format_dict = {'product_id': ITEM.name, 'id': ITEM.name}


class TmallBuy(AtomicConverter):
    r"""
    inter: 
        user_id:token	seller_id:token 	item_id:token	category_id:token	timestamp:float	    interactions:float
    """
    filename = "tmall-buy"


class TmallClick(AtomicConverter):
    r"""
    inter: 
        user_id:token	seller_id:token 	item_id:token	category_id:token	timestamp:float	    interactions:float
    """
    filename = "tmall-click"


class Yelp2018(AtomicConverter):
    r"""
    inter: 
        user_id:token	item_id:token	rating:float	timestamp:float	    useful:float	funny:float	cool:float	review_id:token
    user:
        user_id:token	user_name:token	    user_review_count:float	    yelping_since:float	    user_useful:float	user_funny:float	user_cool:float	    elite:token	    fans:float	average_stars:float
    item:
        item_id:token	item_name:token_seq	    address:token_seq	city:token_seq	    state:token	postal_code:token	latitude:float	longitude:float	    item_stars:float	item_review_count:float	is_open:float	categories:token_seq
    """
    filename = "yelp2018\yelp2018"


class Yelp2021(AtomicConverter):
    r"""
    inter: 
        user_id:token	item_id:token	rating:float	timestamp:float	    useful:float	funny:float	cool:float	review_id:token
    user:
        user_id:token	user_name:token	    user_review_count:float	    yelping_since:float	    user_useful:float	user_funny:float	user_cool:float	    elite:token	    fans:float	average_stars:float
    item:
        item_id:token	item_name:token_seq	    address:token_seq	city:token_seq	    state:token	postal_code:token	latitude:float	longitude:float	    item_stars:float	item_review_count:float	is_open:float	categories:token_seq
    """
    filename = "yelp2021\yelp2021"


class Yelp2022(AtomicConverter):
    r"""
    inter: 
        user_id:token	item_id:token	rating:float	timestamp:float	    useful:float	funny:float	cool:float	review_id:token
    user:
        user_id:token	user_name:token	    user_review_count:float	    yelping_since:float	    user_useful:float	user_funny:float	user_cool:float	    elite:token	    fans:float	average_stars:float
    item:
        item_id:token	item_name:token_seq	    address:token_seq	city:token_seq	    state:token	postal_code:token	latitude:float	longitude:float	    item_stars:float	item_review_count:float	is_open:float	categories:token_seq
    """
    filename = "yelp2022\yelp2022"


class YooChooseBuys(AtomicConverter):
    r"""
    inter: 
        session_id:token	timestamp:float	item_id:token	price:float	quantity:float
    """
    filename = "yoochoose-buys"

class YooChooseBuys14(YooChooseBuys):

    def sess_split_by_ratio(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by ratios: {ratios} ...")

        groups = list(self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values().index)
        groups = groups[-ceil(len(groups) / 4):] # recent 1/4 sessions

        markers = np.cumsum(ratios)
        l = max(floor(markers[0] * len(groups) / markers[-1]), 1)
        r = floor(markers[1] * len(groups) / markers[-1])

        traingroups = groups[:l]
        validgroups = groups[l:r]
        testgroups = groups[r:]

        self.trainiter = self.interactions[self.interactions[SESSION.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[SESSION.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[SESSION.name].isin(testgroups)]


class YooChooseBuys164(YooChooseBuys):

    def sess_split_by_ratio(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by ratios: {ratios} ...")

        groups = list(self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values().index)
        groups = groups[-ceil(len(groups) / 64):] # recent 1/64 sessions

        markers = np.cumsum(ratios)
        l = max(floor(markers[0] * len(groups) / markers[-1]), 1)
        r = floor(markers[1] * len(groups) / markers[-1])

        traingroups = groups[:l]
        validgroups = groups[l:r]
        testgroups = groups[r:]

        self.trainiter = self.interactions[self.interactions[SESSION.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[SESSION.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[SESSION.name].isin(testgroups)]


class YooChooseClicks(AtomicConverter):
    r"""
    inter: 
        session_id:token	timestamp:float	item_id:token	category:token
    """
    filename = "yoochoose-clicks"


class YooChooseClicks14(YooChooseClicks):

    def sess_split_by_ratio(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by ratios: {ratios} ...")

        groups = list(self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values().index)
        groups = groups[-ceil(len(groups) / 4):] # recent 1/4 sessions

        markers = np.cumsum(ratios)
        l = max(floor(markers[0] * len(groups) / markers[-1]), 1)
        r = floor(markers[1] * len(groups) / markers[-1])

        traingroups = groups[:l]
        validgroups = groups[l:r]
        testgroups = groups[r:]

        self.trainiter = self.interactions[self.interactions[SESSION.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[SESSION.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[SESSION.name].isin(testgroups)]

    def sess_split_by_day(self, days: int = 1):
        infoLogger(f"[Converter] >>> Split by days: {days} ...")

        groups = list(self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values().index)
        groups = groups[-ceil(len(groups) / 4):] # recent 1/4 sessions
        self.interactions = self.interactions[self.interactions[SESSION.name].isin(groups)]

        seconds_per_day = 86400
        seconds = seconds_per_day * days

        last_date = self.interactions[TIMESTAMP.name].max().item()

        # Group interactions by session and calculate session timestamps
        session_timestamps = self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values()

        # Split interactions into train, validation, and test sets based on session timestamps
        traingroups = session_timestamps[session_timestamps < (last_date - 2 * seconds)].index
        validgroups = session_timestamps[(session_timestamps >= (last_date - 2 * seconds)) & (session_timestamps < (last_date - seconds))].index
        testgroups = session_timestamps[session_timestamps >= (last_date - seconds)].index

        assert len(traingroups) >= 0, f"The given `days` of {days} leads to zero-size trainsets ..."
        assert len(validgroups) >= 0, f"The given `days` of {days} leads to zero-size validsets ..."
        assert len(testgroups) >= 0, f"The given `days` of {days} leads to zero-size testsets ..."

        self.trainiter = self.interactions[self.interactions[SESSION.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[SESSION.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[SESSION.name].isin(testgroups)]


class YooChooseClicks164(YooChooseClicks):

    def sess_split_by_ratio(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by ratios: {ratios} ...")

        groups = list(self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values().index)
        groups = groups[-ceil(len(groups) / 64):] # recent 1/64 sessions

        markers = np.cumsum(ratios)
        l = max(floor(markers[0] * len(groups) / markers[-1]), 1)
        r = floor(markers[1] * len(groups) / markers[-1])

        traingroups = groups[:l]
        validgroups = groups[l:r]
        testgroups = groups[r:]

        self.trainiter = self.interactions[self.interactions[SESSION.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[SESSION.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[SESSION.name].isin(testgroups)]

    def sess_split_by_day(self, days: int = 1):
        infoLogger(f"[Converter] >>> Split by days: {days} ...")

        groups = list(self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values().index)
        groups = groups[-ceil(len(groups) / 64):] # recent 1/64 sessions
        self.interactions = self.interactions[self.interactions[SESSION.name].isin(groups)]

        seconds_per_day = 86400
        seconds = seconds_per_day * days

        last_date = self.interactions[TIMESTAMP.name].max().item()

        # Group interactions by session and calculate session timestamps
        session_timestamps = self.interactions.groupby(SESSION.name)[TIMESTAMP.name].max().sort_values()

        # Split interactions into train, validation, and test sets based on session timestamps
        traingroups = session_timestamps[session_timestamps < (last_date - 2 * seconds)].index
        validgroups = session_timestamps[(session_timestamps >= (last_date - 2 * seconds)) & (session_timestamps < (last_date - seconds))].index
        testgroups = session_timestamps[session_timestamps >= (last_date - seconds)].index

        assert len(traingroups) >= 0, f"The given `days` of {days} leads to zero-size trainsets ..."
        assert len(validgroups) >= 0, f"The given `days` of {days} leads to zero-size validsets ..."
        assert len(testgroups) >= 0, f"The given `days` of {days} leads to zero-size testsets ..."

        self.trainiter = self.interactions[self.interactions[SESSION.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[SESSION.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[SESSION.name].isin(testgroups)]

