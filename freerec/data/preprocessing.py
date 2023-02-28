


from typing import Callable, Iterable, List, Tuple

import numpy as np
import os
import pandas as pd
import torchdata.datapipes as dp
from math import ceil, floor
from collections import defaultdict
from itertools import chain

from ..utils import mkdirs



def reporter(*attributes: str):
    def decorator(func: Callable):
        def wrapper(self, *args, **kwargs) -> Iterable:
            infos = [f"[{attr}: {getattr(self, attr)}]" for attr in attributes]
            infos = "  ".join(infos)
            print(f">>> Do `{func.__name__}' under the settings: {infos}")
            return func(self, *args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

class Inter2Txt:

    def __init__(
        self,
        root: str, filename: str, dataset: str,
        kcore4user: int = 1, kcore4item: int = 1, star4pos: int = 0,
        ratios: Iterable = (8, 1, 1)
    ) -> None:

        self.root = root
        self.filename = filename
        self.dataset = dataset
        self.kcore4user = int(kcore4user)
        self.kcore4item = int(kcore4item)
        self.star4pos = int(star4pos)
        self.ratios = ratios
        self.code = f"{self.kcore4user}{self.kcore4item}{self.star4pos}{''.join(map(str, self.ratios))}"

    @staticmethod
    def listmap(func: Callable, *args):
        return list(map(
            func, *args
        ))

    @staticmethod
    def where_is_(xxx: str, names: Iterable[str]):
        for i, name in enumerate(names):
            if name.startswith(xxx):
                return i
        return None
    
    @reporter()
    def load_data(self):
        datapipe = dp.iter.FileLister(os.path.join(self.root, self.filename))
        datapipe = datapipe.filter(filter_fn=lambda file_: file_.endswith('.inter'))
        datapipe = datapipe.open_files(mode='rt')
        names = next(iter(
            datapipe.parse_csv(delimiter='\t', skip_lines=0)
        ))
        userIdx = self.where_is_('user', names)
        itemIdx = self.where_is_('item', names)
        starIdx = self.where_is_('rating', names)
        timestampIdx = self.where_is_('timestamp', names)
        print(f"The .inter file includes columns: {names}")
        print(f"Eg. User@{userIdx} Item@{itemIdx} Timestamp@{timestampIdx} Rating@{starIdx}")

        datapipe = datapipe.parse_csv(delimiter='\t', skip_lines=1)
        if starIdx is None: # (User, Item, Time)
            datapipe = datapipe.map(
                lambda row: (str(row[userIdx]), str(row[itemIdx]), float(row[timestampIdx]))
            )
            data = list(datapipe)
        else: # (User, Item, Time, Star)
            datapipe = datapipe.map(
                lambda row: (str(row[userIdx]), str(row[itemIdx]), float(row[timestampIdx]), float(row[starIdx]))
            )
            data = self.filter_by_star(list(datapipe))
        return data

    @reporter('star4pos')
    def filter_by_star(self, data: Iterable) -> List:
        filtered = []
        for row in data:
            if row[3] >= self.star4pos:
                filtered.append(row[:3])
        return filtered

    @reporter('kcore4user', 'kcore4item')
    def filter_by_core(self, data: Iterable) -> List:
        datasize = 0

        while datasize != len(data):
            datasize = len(data)
            print(f"datasize: {datasize}")

            count_per_user = defaultdict(int)
            count_per_item = defaultdict(int)
            users, items = set(), set()

            for row in data:
                user, item = row[0], row[1]
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

    @reporter()
    def sort_by_timestamp(self, data: List) -> List:
        return sorted(data, key=lambda row: (row[0], row[2]))

    @reporter()
    def UI2ID(self, data: List) -> List:
        users, items = list(zip(*data))[:2]
        users, items = set(users), set(items)
        userMap = dict(zip(users, range(len(users))))
        itemMap = dict(zip(items, range(len(items))))
        self.userCount = len(users)
        self.itemCount = len(items)

        data = self.listmap(
            lambda row: (userMap[row[0]], itemMap[row[1]], row[2]),
            data
        )

        return data

    @reporter()
    def user_item_only(self, data: List) -> List:
        return self.listmap(lambda row: (row[0], row[1]), data)

    @reporter()
    def user_item_time_only(self, data: List) -> List:
        return self.listmap(lambda row: (row[0], row[1], row[2]), data)

    @reporter()
    def group_by_user(self, data: Iterable) -> List:
        data_by_user = defaultdict(list)
        for row in data:
            data_by_user[row[0]].append(row) # (User, Item, TimeStamp)
        return data_by_user

    def summary(self):
        from prettytable import PrettyTable
        table = PrettyTable(['#User', '#Item', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        table.add_row([
            self.userCount, self.itemCount, self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (self.userCount * self.itemCount)
        ])
        print(table)


class GenInter2Txt(Inter2Txt):

    @reporter('ratios')
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

    @reporter('root')
    def save(self, trainset, validset, testset):
        path = os.path.join(
            self.root, 'General', 
            '_'.join([self.dataset, self.code, 'Chron'])
        )
        mkdirs(path)

        df = pd.DataFrame(trainset, columns=['User', 'Item'])
        df.to_csv(os.path.join(path, 'train.txt'), sep='\t', index=False)

        df = pd.DataFrame(validset, columns=['User', 'Item'])
        df.to_csv(os.path.join(path, 'valid.txt'), sep='\t', index=False)

        df = pd.DataFrame(testset, columns=['User', 'Item'])
        df.to_csv(os.path.join(path, 'test.txt'), sep='\t', index=False)

    def compile(self):
        data = self.load_data()
        data = self.filter_by_core(data)
        data = self.UI2ID(data)
        data = self.sort_by_timestamp(data)
        data = self.user_item_only(data)
        data = self.group_by_user(data)
        self.save(*self.split_by_ratio(data))
        self.summary()
    
class SeqInter2Txt(Inter2Txt):

    @reporter()
    def split(self, data_by_user: List) -> Tuple[List, List, List]:
        trainset = []
        validset = []
        testset = []
        for user in range(self.userCount):
            pairs = data_by_user[user]
            if len(pairs) == 0:
                continue
            if len(pairs) <= 3:
                trainset.append(pairs)
            else:
                trainset.append(pairs[:-2])
                validset.append(pairs[-2:-1])
                testset.append(pairs[-1:])

        trainset = list(chain(*trainset))
        validset = list(chain(*validset))
        testset = list(chain(*testset))

        self.trainsize = len(trainset)
        self.validsize = len(validset)
        self.testsize = len(testset) 
        return trainset, validset, testset

    @reporter('root')
    def save(self, trainset, validset, testset):
        path = os.path.join(self.root, 'Sequential', self.dataset)
        mkdirs(path)

        df = pd.DataFrame(trainset, columns=['User', 'Item', 'Timestamp'])
        df.to_csv(os.path.join(path, 'train.txt'), sep='\t', index=False)

        df = pd.DataFrame(validset, columns=['User', 'Item', 'Timestamp'])
        df.to_csv(os.path.join(path, 'valid.txt'), sep='\t', index=False)

        df = pd.DataFrame(testset, columns=['User', 'Item', 'Timestamp'])
        df.to_csv(os.path.join(path, 'test.txt'), sep='\t', index=False)

    def compile(self):
        data = self.load_data()
        data = self.filter_by_core(data)
        data = self.UI2ID(data)
        data = self.sort_by_timestamp(data)
        data = self.user_item_time_only(data)
        data = self.group_by_user(data)
        self.save(*self.split(data))
        self.summary()
  