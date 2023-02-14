

from typing import Iterator, Iterable, Union, List, Optional

import torchdata.datapipes as dp

from .base import Postprocessor
from ..fields import FieldList, BufferField
from ..tags import FieldTags, USER, ITEM, TARGET


__all__ = ['Grouper', 'Wrapper']


@dp.functional_datapipe("group_")
class Grouper(Postprocessor):
    """Group batch into several groups."""
    def __init__(
        self, datapipe: Postprocessor, 
        groups: Iterable[Union[FieldTags, Iterable[FieldTags]]] = (USER, ITEM, TARGET)
    ) -> None:
        """
        Parameters:
        ---
        groups: Iterable[Union[FieldTags, Iterable[FieldTags]]]
            Gathering fields for each group of tags in groups.

        Examples:
        ---

        >>> from freerec.data.tags import SPARSE, TARGET, USER, ITEM
        >>> from freerec.data.datasets import MovieLens1M
        >>> basepipe = MovieLens1M("../../data/MovieLens1M")
        >>> datapipe = basepipe.tensor_().chunk_(1024)
        >>> dataset = datapipe.wrap_().group_((USER, ITEM, TARGET))
        >>> len(next(iter(dataset)))
        3
        >>> dataset = datapipe.wrap_().group_((ID, TARGET))
        >>> len(next(iter(dataset)))
        2
        """
        super().__init__(datapipe)
        self.groups = []
        for tags in groups:
            if isinstance(tags, FieldTags):
                tags = (tags,)
            self.groups.append([field for field in self.fields if field.match(*tags)])

    def forward(self) -> List:
        for batch in self.source:
            yield [{field.name: batch[field.name] for field in group} for group in self.groups]


@dp.functional_datapipe("wrap_")
class Wrapper(Postprocessor):
    """Wrap trainpipe, validpipe and testpipe together."""

    def __init__(
        self, datapipe: Postprocessor,
        validpipe: Optional[Postprocessor] = None,
        testpipe: Optional[Postprocessor] = None,
    ) -> None:
        """
        Parameters:
        ---

        datapipe: trainpipe
        validpipe: validpipe or None
            - `None`: Set validpipe == trainpipe

        testpipe: testpipe
            - `None`: Set testpipe == validpipe

        Examples:
        ---

        >>> from freerec.data.datasets import MovieLens1M
        >>> basepipe = MovieLens1M("../../data/MovieLens1M")
        >>> datapipe = basepipe.shuffle_().shard_()
        >>> trainpipe = datapipe.negatives_for_train_(num_negatives=4)
        >>> validpipe = datapipe.negatives_for_eval_(num_negatives=99) # 1:99
        >>> dataset = trainpipe.wrap_(validpipe).tensor_().chunk_(cfg.batch_size).group_()
        """
        super().__init__(datapipe)
        self.validpipe = datapipe if validpipe is None else validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe
        self.train()

    def train(self):
        super().train()
        self.source.train()
        return self

    def valid(self):
        super().valid()
        self.validpipe.valid()
        return self

    def test(self):
        super().test()
        self.testpipe.test()
        return self

    def __len__(self):
        if self.mode == 'train':
            return len(self.source)
        elif self.mode == 'valid':
            return len(self.validpipe)
        else:
            return len(self.testpipe)

    def forward(self) -> Iterator[FieldList[BufferField]]:
        if self.mode == 'train':
            yield from self.source
        elif self.mode == 'valid':
            yield from self.validpipe
        else:
            yield from self.testpipe
