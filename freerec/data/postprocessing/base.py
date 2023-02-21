

from typing import Callable, Union, Iterable

import torchdata.datapipes as dp
from ..fields import Field, FieldTuple


__all__ = ['BaseProcessor', 'Postprocessor', 'Adapter']


class BaseProcessor(dp.iter.IterDataPipe):
    """A base processor that defines the property of fields.

    Args:
        fields (None, Field, Iterable):
            - `None': Pass.
            - `Field`: FieldTuple with one Field.
            - `Iterable`: FieldTuple with multi Fields
    
    Raises:
        AttributeError: If `fields' are not given or `None` before using.

    """

    def __init__(self, fields: Union[None, Field, Iterable] = None) -> None:
        super().__init__()
        self.fields = fields

    @property
    def fields(self) -> FieldTuple:
        try:
            return self.__fields
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.___name__}.fields should be given (not None) before using."
            )

    @fields.setter
    def fields(self, fields: Union[None, Field, Iterable] = None):
        """Set fields.

        Args:
            fields (None, Field, Iterable):
                - `None': Pass.
                - `Field`: FieldTuple with one Field.
                - `Iterable`: FieldTuple with multi Fields.
        """
        if fields is None:
            pass
        elif isinstance(fields, Field):
            self.__fields = FieldTuple((Field,))
        elif isinstance(fields, Iterable):
            self.__fields = FieldTuple(fields)
        else:
            raise ValueError(
                f"None|Field|Iterable type expected but {type(fields)} received ..."
            )
 
    @staticmethod
    def listmap(func: Callable, *iterables):
        """Apply a function to multiple iterables and return a list.

        Args:
            func (Callable): The function to be applied.
            *iterables: Multiple iterables to be processed.

        Returns:
            List: The results after applying the function to the iterables.
        """
        return list(map(func, *iterables))

   

class Postprocessor(BaseProcessor):
    """A post-processor that wraps another IterDataPipe object.

    Args:
        source_dp (dp.iter.IterDataPipe): The data pipeline to be wrapped.
    """


    def __init__(
        self, source_dp: dp.iter.IterDataPipe,
        *, 
        fields: Union[None, Field, Iterable] = None
    ) -> None:
        super().__init__(fields=fields)
        self.source = source_dp


class Adapter(BaseProcessor):
    """A base class for data pipeline adapters."""

    def __init__(self, fields: Union[None, Field, Iterable] = None) -> None:
        super().__init__(fields)

        self.__mode = 'train'
        
    @property
    def mode(self):
        """Get the current mode of the adapter.

        Returns:
            str: The current mode of the adapter.
        """
        return self.__mode

    def train(self):
        """Set the mode of the adapter to 'train' and return itself.

        Returns:
            Adapter: Itself.
        """
        self.__mode = 'train'
        return self

    def valid(self):
        """Set the mode of the adapter to 'valid' and return itself.

        Returns:
            Adapter: Itself.
        """
        self.__mode = 'valid'
        return self

    def test(self):
        """Set the mode of the adapter to 'test' and return itself.

        Returns:
            Adapter: Itself.
        """
        self.__mode = 'test'
        return self


    def __getattr__(self, attribute_name):
        if attribute_name in dp.iter.IterDataPipe.functions:
            raise AttributeError(
                f"`{self.__class__.__name__}' must be the end of the pipeline."
                f"The follow-up operation of `{attribute_name}' is invalid here."
            )
        else:
            return super().__getattr__(attribute_name)