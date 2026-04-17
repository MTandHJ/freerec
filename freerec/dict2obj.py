from typing import Any


class Config(dict):
    r"""A dictionary subclass that allows attribute-style access to its items.

    Keys set via ``__setitem__`` are mirrored as attributes, but attributes
    set directly via ``__setattr__`` are **not** added to the underlying
    dictionary.

    Parameters
    ----------
    prefix : str, optional
        Prefix string used in the string representation.  Default is ``">>>"``.
    *args
        Positional arguments forwarded to :class:`dict`.
    **kwargs
        Keyword arguments forwarded to :class:`dict`.

    Examples
    --------
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

    def __init__(self, *args, prefix: str = ">>>", **kwargs):
        r"""Initialize the Config and mirror initial items as attributes."""
        super(Config, self).__init__(*args, **kwargs)
        for name, attr in self.items():
            self.__setattr__(name, attr)

        self.prefix = prefix

    def __setattr__(self, name: str, value: Any) -> None:
        r"""Set an attribute, updating the dict entry only if the key exists.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Value to assign.
        """
        super(Config, self).__setattr__(name, value)
        if name in self:
            super(Config, self).__setitem__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        r"""Set an item in the dictionary and mirror it as an attribute.

        Parameters
        ----------
        key : str
            Dictionary key.
        value : Any
            Value to assign.
        """
        super(Config, self).__setitem__(key, value)
        super(Config, self).__setattr__(key, value)

    def update(self, **kwargs) -> None:
        r"""Update existing keys with new values; unknown keys are ignored.

        Parameters
        ----------
        **kwargs
            Key-value pairs.  Only keys already present in the dictionary
            are updated.
        """
        for key, value in kwargs.items():
            if key in self.keys():
                self[key] = value

    def __str__(self) -> str:
        r"""Return a human-readable string representation."""
        item = "[{name}: {val}] \n"
        infos = f"[{self.__class__.__name__.upper()}] " + self.prefix + "\n"
        for name, val in self.items():
            infos += item.format(name=name, val=val)
        return infos


if __name__ == "__main__":
    import doctest

    doctest.testmod()
