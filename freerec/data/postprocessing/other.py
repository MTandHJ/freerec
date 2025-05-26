

import torchdata.datapipes as dp


@dp.functional_datapipe('mark_')
class Marker(dp.iter.IterDataPipe):
    r"""
    Mark a piece of data.

    Parameters:
    -----------
    source: dp.iter.IterDataPipe
    markers: Dict

    Examples:
    ---------
    >>> source_dp1 = IterableWrapper([{'i': i} for i in range(3)]).mark_(dataset='A')
    >>> source_dp2 = IterableWrapper([{'i': i} for i in range(3)]).mark_(dataset='B')
    >>> d = {source_dp1: 1, source_dp2: 1}
    >>> sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d)
    >>> list(sample_mul_dp)
    [{'i': 0, 'dataset': 'B'},
     {'i': 1, 'dataset': 'B'},
     {'i': 0, 'dataset': 'A'},
     {'i': 1, 'dataset': 'A'},
     {'i': 2, 'dataset': 'B'},
     {'i': 2, 'dataset': 'A'}]
    """

    def __init__(
        self, source: dp.iter.IterDataPipe,
        **markers
    ):
        super().__init__()

        self.source = source
        self.markers = markers

    def __iter__(self):
        for d in self.source:
            yield d | self.markers

