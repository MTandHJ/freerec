

from typing import Optional

import torchdata.datapipes as dp

from .base import Adapter


__all__ = ['Wrapper']


@dp.functional_datapipe("wrap_")
class Wrapper(Adapter):
    """Wrap trainpipe, validpipe and testpipe together.
    
    Notes:
        If `validpipe' is `None', we will use `trainpipe' instead.
        If `testpipe' is `None', we will use `validpipe' instead.
    
    """

    def __init__(
        self, 
        trainpipe: dp.iter.IterDataPipe,
        validpipe: Optional[dp.iter.IterDataPipe] = None,
        testpipe: Optional[dp.iter.IterDataPipe] = None,
    ) -> None:
        """Initialize Wrapper with train, valid and test datapipe.

        Args:
            trainpipe (dp.iter.IterDataPipe): Training datapipe.
            validpipe (Optional[dp.iter.IterDataPipe], optional): Validation datapipe. Defaults to None.
            testpipe (Optional[dp.iter.IterDataPipe], optional): Test datapipe. Defaults to None.
        """
        super().__init__()
        self.trainpipe = trainpipe
        self.validpipe = trainpipe if validpipe is None else validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe
        self.train()

    def __len__(self) -> int:
        """Return the length of datapipe depending on the mode.

        Returns:
            int: The length of datapipe in train, valid or test mode.
        """
        if self.mode == 'train':
            return len(self.source)
        elif self.mode == 'valid':
            return len(self.validpipe)
        else:
            return len(self.testpipe)

    def __iter__(self):
        """Iterate through the datapipe based on the mode.

        Yields:
            The data from the datapipe.
        """
        if self.mode == 'train':
            yield from self.trainpipe
        elif self.mode == 'valid':
            yield from self.validpipe
        else:
            yield from self.testpipe