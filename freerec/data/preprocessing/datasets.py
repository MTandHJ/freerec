

from .base import AtomicConverter
from ..tags import USER, ITEM, RATING, TIMESTAMP


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
    filename = "Amazon_Beauty"


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


class AmazonMovies(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	categories:token_seq	title:token	    price:float	    sales_type:token	sales_rank:float	brand:token
    """
    filename = "Amazon_Movies_and_TV"


class AmazonGames(AtomicConverter):
    r"""
    inter:
        user_id:token	item_id:token	rating:float	timestamp:float
    item:
        item_id:token	price:float	    sales_type:token	sales_rank:float	categories:token_seq	title:token	brand:token
    """
    filename = "Amazon_Video_Games"


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
    filename = "gowalla"


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
        session_id:token	item_id:token	count:float	    timestamp:float
    """
    filename = "yoochoose-buys"
    _name_format_dict = {'session_id': USER.name, 'count': RATING.name}


class YooChooseClicks(AtomicConverter):
    r"""
    inter: 
        session_id:token	item_id:token	count:float	timestamp:float
    """
    filename = "yoochoose-clicks"
    _name_format_dict = {'session_id': USER.name, 'count': RATING.name}

