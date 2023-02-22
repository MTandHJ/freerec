

from typing import Any


class Config(dict):
    """
    A dictionary subclass that converts a dictionary to an object.

    Parameters:
    ----------
    prefix: str, optional (default=">>>")
        The prefix for the string representation of the object.

    Examples:
    ---------
    >>> cfg = Config({1:2}, a=3)
    Traceback (most recent call last):
    ...
    TypeError: attribute name must be string, not 'int'
    >>> cfg = Config(a=1, b=2)
    >>> cfg.a
    1
    >>> cfg['a']
    1
    >>> cfg['c'] = 3
    >>> cfg.c
    3
    >>> cfg.d = 4
    >>> cfg['d']
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    >>> cfg.update(**Config({'a':4, 'd':5, 'e':6}))
    >>> cfg.a
    4
    >>> cfg['d']
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    >>> cfg.e
    Traceback (most recent call last):
    ...
    AttributeError: 'Config' object has no attribute 'e'
    """
    def __init__(
        self, *args, 
        prefix: str = ">>>",
        **kwargs
    ):
        super(Config, self).__init__(*args, **kwargs)
        for name, attr in self.items():
            self.__setattr__(name, attr)
        
        self.prefix = prefix

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute and corresponding item in the dictionary.

        Parameters:
        -----------
        name : str
            The name of the attribute to be set.
        value : any
            The value to be set for the attribute.
        """
        super(Config, self).__setattr__(name, value)
        if name in self:
            super(Config, self).__setitem__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set an item in the dictionary and corresponding attribute.

        Parameters:
        -----------
        key : str
            The key to be set in the dictionary.
        value : any
            The value to be set for the key in the dictionary.
        """
        super(Config, self).__setitem__(key, value)
        super(Config, self).__setattr__(key, value)

    def update(self, **kwargs) -> None:
        """
        Update the dictionary with new keys and values.

        Parameters
        ----------
        kwargs : dict
            A dictionary of key-value pairs to update the object.
        """
        for key, value in kwargs.items():
            if key in self.keys():
                self[key] = value

    def __str__(self) -> str:
        """
        Return a string representation of the object.

        Returns:
        --------
        str
            A string representation of the object.
        """
        item = "[{name}: {val}] \n"
        infos = f"[{self.__class__.__name__.upper()}] " + self.prefix + "\n"
        for name, val in self.items():
            infos += item.format(name=name, val=val)
        return infos


if __name__ == "__main__":
    import doctest
    doctest.testmod()











