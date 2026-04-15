


import torchdata.datapipes as dp


@dp.functional_datapipe('mark_')
class Marker(dp.iter.IterDataPipe):
    r"""Attach constant key-value markers to every row yielded by a datapipe.

    Parameters
    ----------
    source : :class:`~IterDataPipe`
        The upstream datapipe.
    **markers
        Arbitrary keyword arguments that will be merged into each row dict.

    Examples
    --------
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
        r"""Initialize the Marker."""
        super().__init__()

        self.source = source
        self.markers = markers

    def __iter__(self):
        r"""Yield rows with markers merged in."""
        for d in self.source:
            yield d | self.markers
