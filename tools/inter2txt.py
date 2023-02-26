
from typing import Callable, Iterable, List, Tuple

import numpy as np
import os
import pandas as pd
import torchdata.datapipes as dp
from math import ceil, floor
from collections import defaultdict
from itertools import chain


def mkdirs(*paths: str) -> None:
    r"""
    Create directories.

    Parameters:
    -----------
    *paths : str
        Paths of directories to create.
    """
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


class Inter2Txt:

    def __init__(
        self,
        path4loading: str, filename: str, path4saving: str,
        kcore4user: int = 10, kcore4item: int = 10, star4pos: float = 4.,
        ratios: Iterable = (8, 1, 1)
    ) -> None:

        self.path4loading = path4loading
        self.filename = filename
        self.path4saving = path4saving
        self.kcore4user = kcore4user
        self.kcore4item = kcore4item
        self.star4pos = star4pos
        self.ratios = ratios

    def _where_is_xxx(self, xxx: str, names: Iterable[str]):
        for i, name in enumerate(names):
            if name.startswith(xxx):
                return i
        return None
    
    def load_data(self, row_processor: Callable):
        datapipe = dp.iter.FileLister(self.path4loading)
        datapipe = datapipe.filter(filter_fn=lambda file_: file_.endswith(self.filename))
        datapipe = datapipe.open_files(mode='rt')
        names = next(iter(
            datapipe.parse_csv(delimiter='\t', skip_lines=0)
        ))
        userIdx = self._where_is_xxx('user', names)
        itemIdx = self._where_is_xxx('item', names)
        starIdx = self._where_is_xxx('rating', names)
        timestampIdx = self._where_is_xxx('timestamp', names)
        datapipe = datapipe.parse_csv(delimiter='\t', skip_lines=1)
        datapipe = datapipe.map(row_processor)
        if starIdx is None: # (User, Item, Time)
            datapipe = datapipe.map(
                lambda row: (row[userIdx], row[itemIdx], row[timestampIdx])
            )
        else: # (User, Item, Star, Time)
            datapipe = datapipe.map(
                lambda row: (row[userIdx], row[itemIdx], row[starIdx], row[timestampIdx])
            )
        return datapipe

    def filter_by_star(self, datapipe: Iterable) -> List:
        data = []
        for row in datapipe:
            if row[2] >= self.star4pos:
                data.append((
                    row[0], row[1], row[3]
                ))
        return data

    def filter_by_core(self, data: List) -> List:
        datasize = 0

        while datasize != len(data):
            datasize = len(data)
            print(f"datasize: {datasize}")

            count_per_user = defaultdict(int)
            count_per_item = defaultdict(int)
            users, items = set(), set()

            for user, item, time in data:
                count_per_user[user] += 1
                count_per_item[item] += 1
                if count_per_user[user] >= self.kcore4user:
                    users.add(user)
                if count_per_item[item] >= self.kcore4item:
                    items.add(item)
            data = list(filter(
                lambda row: row[0] in users and row[1] in items,
                data
            ))
        
        return data

    def sort_by_timestamp(self, data: List) -> List:
        return sorted(data, key=lambda row: (row[0], row[2])) # (User, Item)

    def group_by_user(self, data: List) -> List:
        users, items, _ = zip(*data)
        users, items = set(users), set(items)
        userMap = dict(zip(users, range(len(users))))
        itemMap = dict(zip(items, range(len(items))))
        self.userCount = len(users)
        self.itemCount = len(items)

        data = list(map(
            lambda row: (userMap[row[0]], itemMap[row[1]], row[2]),
            data
        ))

        data_by_user = defaultdict(list)
        for row in data:
            data_by_user[row[0]].append(row) # (User, Item, TimeStamp)
        
        return data_by_user


class GenInter2Txt(Inter2Txt):

    def split_by_ratio(self, data_by_user: List) -> Tuple[List, List, List]:
        trainset = []
        validset = []
        testset = []
        markers = np.cumsum(self.ratios)
        for user in range(self.userCount):
            pairs = data_by_user[user]
            if len(pairs) == 0:
                continue
            l = max(floor(markers[0] * len(pairs) / markers[-1]), 1)
            r = floor(markers[1] * len(pairs) / markers[-1])
            trainset.append(pairs[:l])
            if l < r:
                validset.append(pairs[l:r])
            if r < len(pairs):
                testset.append(pairs[r:])
        trainset = list(chain(*trainset))
        validset = list(chain(*validset))
        testset = list(chain(*testset))

        self.trainsize = len(trainset)
        self.validsize = len(validset)
        self.testsize = len(testset) 
        return trainset, validset, testset

    def save(self, trainset, validset, testset):
        mkdirs(self.path4saving)

        df = pd.DataFrame(trainset, columns=['User', 'Item'])
        df.to_csv(os.path.join(self.path4saving, 'train.txt'), sep='\t', index=False)

        df = pd.DataFrame(validset, columns=['User', 'Item'])
        df.to_csv(os.path.join(self.path4saving, 'valid.txt'), sep='\t', index=False)

        df = pd.DataFrame(testset, columns=['User', 'Item'])
        df.to_csv(os.path.join(self.path4saving, 'test.txt'), sep='\t', index=False)

    
class SeqInter2Txt(Inter2Txt):



    def split_by_ratio(self, data_by_user: List) -> Tuple[List, List, List]:
        trainset = []
        validset = []
        testset = []
        markers = np.cumsum(self.ratios)
        for user in range(self.userCount):
            pairs = data_by_user[user]
            if len(pairs) == 0:
                continue
            l = max(floor(markers[0] * len(pairs) / markers[-1]), 1)
            r = floor(markers[1] * len(pairs) / markers[-1])
            trainset.append(pairs[:l])
            if l < r:
                validset.append(pairs[l:r])
            if r < len(pairs):
                testset.append(pairs[r:])
        trainset = list(chain(*trainset))
        validset = list(chain(*validset))
        testset = list(chain(*testset))

        self.trainsize = len(trainset)
        self.validsize = len(validset)
        self.testsize = len(testset) 
        return trainset, validset, testset

    def save(self, trainset, validset, testset):
        mkdirs(self.path4saving)

        df = pd.DataFrame(trainset, columns=['User', 'Item'])
        df.to_csv(os.path.join(self.path4saving, 'train.txt'), sep='\t', index=False)

        df = pd.DataFrame(validset, columns=['User', 'Item'])
        df.to_csv(os.path.join(self.path4saving, 'valid.txt'), sep='\t', index=False)

        df = pd.DataFrame(testset, columns=['User', 'Item'])
        df.to_csv(os.path.join(self.path4saving, 'test.txt'), sep='\t', index=False)

