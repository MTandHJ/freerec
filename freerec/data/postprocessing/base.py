

from typing import Callable, Union, Iterable

import torchdata.datapipes as dp
from ..fields import Field, FieldTuple


__all__ = ['BaseProcessor', 'Postprocessor']


class BaseProcessor(dp.iter.IterDataPipe):
    r"""
    A base processor that defines the property of fields.

    Parameters:
    -----------
    fields: Field or Iterable, optional
        - `None': Pass.
        - `Field`: FieldTuple with one Field.
        - `Iterable`: FieldTuple with multi Fields
    
    Raises:
    -------
    AttributeError: 
        If `fields' are not given or `None` before using.
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
        r"""
        Set fields.

        Parameters:
        -----------
        fields: Field or Iterable, optional
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
        r"""
        Apply a function to multiple iterables and return a list.

        Parameters:
        -----------
        func (Callable): The function to be applied.
        *iterables: Multiple iterables to be processed.

        Returns:
        --------
        List: The results after applying the function to the iterables.
        """
        return list(map(func, *iterables))
   

class Postprocessor(BaseProcessor):
    r"""
    A post-processor that wraps another IterDataPipe object.

    Parameters:
    -----------
    source_dp: dp.iter.IterDataPipe 
        The data pipeline to be wrapped.
    """


    def __init__(
        self, source_dp: dp.iter.IterDataPipe,
        *, 
        fields: Union[None, Field, Iterable] = None
    ) -> None:
        super().__init__(fields=fields)
        self.source = source_dp