

from typing import Iterator, Iterable, Union, List, Optional

import torchdata.datapipes as dp

from .base import Postprocessor
from ..tags import FieldTags, USER, ITEM, TARGET


__all__ = ['Grouper', 'Wrapper']


@dp.functional_datapipe("group_")
class Grouper(Postprocessor):
    """Group batch into several groups
    For example, RS generally requires
        for users, items, targets in datapipe: ...
    Note that we assume the last group is TARGET, which should be returned in List form.
    So the final returns are in the form of:
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], ...
    """
    def __init__(
        self, datapipe: Postprocessor, 
        groups: Iterable[Union[FieldTags, Iterable[FieldTags]]] = (USER, ITEM, TARGET)
    ) -> None:
        super().__init__(datapipe)
        self.groups = [[field for field in self.fields if field.match(tags)] for tags in groups]

    def __iter__(self) -> List:
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
        Args:
            datapipe: trainpipe
        Kwargs:
            validpipe: validpipe <- trainpipe if validpipe is None
            testpipe: testpipe <- validpipe if testpipe is None
        """
        super().__init__(datapipe)
        self.validpipe = datapipe if validpipe is None else validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe
        self.train()

    def train(self):
        super().train()
        self.source.train()

    def valid(self):
        super().valid()
        self.validpipe.valid()

    def test(self):
        super().test()
        self.testpipe.test()

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            yield from self.source
        elif self.mode == 'valid':
            yield from self.validpipe
        else:
            yield from self.testpipe
