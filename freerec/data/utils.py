



from typing import TypeVar, Callable
from ..utils import errorLogger

T = TypeVar('T')

def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    try:
        if val:
            return dest_type(val)
        else: # fill_na
            return default
    except ValueError:
        errorLogger(
            f"Using '{dest_type.__name__}' to convert '{val}' where the default value is {default}", 
            ValueError
        )