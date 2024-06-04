

from typing import Literal, Tuple, Iterable, Optional, Union, Mapping

import os, random
import numpy as np
import pandas as pd
from math import floor, ceil

from ..tags import USER, ITEM, ID, RATING, TIMESTAMP, FEATURE
from ..fields import Field, FieldTuple
from ..utils import download_from_url, extract_archive
from ...utils import infoLogger, warnLogger, mkdirs


__all__ = ['AtomicConverter']


class AtomicConverter:
    r"""
    How to convert Atomic files into train|valid|test.txt ?

    Flows:
    ------
    1. Download corresponding files from [[RecBole](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj)].
    2. Import corresponding dataset.
    3. Call `make_xxx_dataset'.

    Parameters:
    -----------
    root: str
        The root of data.
    filedir: str
        The file with Atomic files.
        - `None': Use cls.filename instead.
    dataset: str, optional
        The filename saving dataset.
        - `None': Use classname instead.

    Notes:
    ------
    1. Most of datasets in RecBole have two versions: 
        merged: remove duplicate interactions.
        not_merged: not remove duplicate interactions.
    """
    filename: str

    def __init__(
        self, root: str, 
        dataset: str,
        filedir: Optional[str],
        userColname: str = USER.name,
        itemColname: str = ITEM.name,
        ratingColname: str = RATING.name,
        timestampColname: str = TIMESTAMP.name
    ) -> None:
        super().__init__()

        self.root = root
        self.dataset = dataset
        filedir = filedir if filedir else dataset
        self.path = os.path.join(self.root, filedir)

        if not os.path.exists(self.path) or not any(True for _ in os.scandir(self.path)):
            infoLogger(f"[Converter] >>> {self.path} is not available ...")
            zipfile = dataset + ".zip"
            zippath = os.path.join(self.root, zipfile)
            if os.path.exists(zippath):
                infoLogger(f"[Converter] >>> Find a compressed file: {zipfile} ...")
                extract_archive(
                    zippath, 
                    self.path
                )
            else:
                try:
                    extract_archive(
                        download_from_url(
                            URLS[self.dataset], 
                            root=self.root, filename=zipfile, 
                            overwrite=False
                        ),
                        self.path
                    )
                except KeyError:
                    raise FileNotFoundError(f"No such file of {self.path} and no such dataset {self.dataset} online...")

        self.dataset = dataset if dataset else self.__class__.__name__

        self.name_converter = {
            userColname.upper(): USER.name,
            itemColname.upper(): ITEM.name,
            ratingColname.upper(): RATING.name,
            timestampColname.upper(): TIMESTAMP.name
        }

    def convert_by_column(self, df: pd.DataFrame):
        old_columns = df.columns
        new_columns = []

        for colname in old_columns:
            newname = colname.upper()
            newname =  self.name_converter.get(newname, newname)
            if newname in (RATING.name, TIMESTAMP.name):
                df[colname] = df[colname].astype(float)
            else:
                df[colname] = df[colname].astype(str)
            new_columns.append(newname)
        df.columns = new_columns
        return df

    def load_inter_file(self):
        for file_ in os.scandir(self.path):
            filename = file_.name
            if not filename.endswith('.inter'):
                continue
            file_ = os.path.join(self.path, filename)
            df = pd.read_csv(file_, sep='\t')
            infoLogger(f"[Converter] >>> Load `{filename}' ...")
            return self.convert_by_column(df)
        infoLogger(f"[Converter] >>> No file ends with `.inter' ...")

    def load_user_file(self):
        for file_ in os.scandir(self.path):
            filename = file_.name
            if not filename.endswith('.user'):
                continue
            file_ = os.path.join(self.path, filename)
            df = pd.read_csv(file_, sep='\t')
            infoLogger(f"[Converter] >>> Load `{filename}' ...")
            return self.convert_by_column(df)
        infoLogger(f"[Converter] >>> No file ends with `.user' ...")

    def load_item_file(self):
        for file_ in os.scandir(self.path):
            filename = file_.name
            if not filename.endswith('.item'):
                continue
            file_ = os.path.join(self.path, filename)
            df = pd.read_csv(file_, sep='\t')
            infoLogger(f"[Converter] >>> Load `{filename}' ...")
            return self.convert_by_column(df)
        infoLogger(f"[Converter] >>> No file ends with `.item' ...")

    def load(self):
        self.interactions = self.load_inter_file()
        self.userFeats = self.load_user_file()
        self.itemFeats = self.load_item_file()
        self.pools = (self.interactions, self.userFeats, self.itemFeats)

    def reserve(self, columns: Iterable[str]):
        all_columns = self.interactions.columns
        columns_ = set(columns) & set(all_columns)
        columns = sorted(columns_, key=columns.index)
        infoLogger(f"[Converter] >>> Reserve fields: {columns} ...")
        self.interactions = self.interactions[columns]

    def exclude(self, columns: Iterable[str]):
        all_columns = self.interactions.columns
        columns = set(all_columns) - set(columns)
        columns = sorted(columns, key=all_columns.index)
        infoLogger(f"[Converter] >>> Exclude fields: {columns} ...")
        self.interactions = self.interactions[columns]

    def reorder(self, columns: Iterable[str] = (USER.name, ITEM.name)):
        all_columns = self.interactions.columns
        columns = columns + all_columns
        columns_ = set(columns) & set(all_columns)
        columns = sorted(columns_, key=columns.index)
        infoLogger(f"[Converter] >>> Reorder fields: {columns} ...")
        self.interactions = self.interactions[columns]

    def map_col(
        self, col: str, mapper: Mapping, 
        pools: Optional[Iterable[pd.DataFrame]] = None
    ):
        pools = self.pools if pools is None else pools
        for df in pools:
            if df is not None:
                df[col] = df[col].map(mapper)

    def filter_by_rating(
        self, 
        low: Union[None, int, float] = None, 
        high: Union[None, int, float] = None
    ):
        try:
            df = self.interactions
            ratings = df[RATING.name]
            low = low if low is not None else ratings.min()
            high = high if high is not None else ratings.max()
            self.interactions = df[(low <= ratings) & (ratings <= high)]
            infoLogger(f"[Converter] >>> Filter dataframe according to {RATING.name} ...")
        except KeyError:
            infoLogger(
                f"[Converter] >>> Skip `filter_by_rating` because of `inter` has no field of `{RATING.name}' ..."
            )
    
    def filter_by_core(
        self,
        low4user: Union[None, int, float] = None, 
        high4user: Union[None, int, float] = None,
        low4item: Union[None, int, float] = None, 
        high4item: Union[None, int, float] = None,
        master: str = USER.name,
        strict: bool = True
    ):
        r"""
        Filter (user, item) by k-core settings.

        Parameters:
        -----------
        low4user: int, float, optional
            Minimum core for user.
            - `None`: float('-inf')
        max4user: int, float, optional
            Maximum core for user.
            - `None`: float('inf')
        low4item: int, float, optional
            Minimum core for item.
            - `None`: float('-inf')
        max4item: int, float, optional
            Maximum core for item.
            - `None`: float('inf')
        strict: bool, default to `True`
            `True`: strictly filter by core
            `False`: filter by core only once
        """
        low4user = low4user if low4user else float('-inf')
        high4user = high4user if high4user else float('inf')
        low4item = low4item if low4item else float('-inf')
        high4item = high4item if high4item else float('inf')
        df = self.interactions
        dsz = -1
        infoLogger(
            f"[Converter] >>> Filter dataframe: "
            f"{master} in [{low4user}, {high4user}]; "
            f"ITEM in [{low4item}, {high4item}] ..."
        )
        infoLogger(f"[Converter] >>> Current datasize: {len(df)} ...")
        while dsz != len(df):
            dsz = len(df)

            # filter by user
            users = df[master]
            counts = users.value_counts()
            bool_indices = users.isin(
                counts[(low4user <= counts) & (counts <= high4user)].index
            )
            df = df[bool_indices]

            # filter by item
            items = df[ITEM.name]
            counts = items.value_counts()
            bool_indices = items.isin(
                counts[(low4item <= counts) & (counts <= high4item)].index
            )
            df = df[bool_indices]

            infoLogger(f"[Converter] >>> Current datasize: {len(df)}")

            if not strict:
                break

        self.interactions = df

    def user2token(self, master: str = USER.name):
        infoLogger(f"[Converter] >>> Map user ID to Token ...")
        user_ids = sorted(self.interactions[master].unique().tolist())
        self.userCount = len(user_ids)

        user_tokens = list(range(len(user_ids)))
        self.userMaps = dict(
            zip(user_ids, user_tokens)
        )

        if self.userFeats is not None:
            self.userFeats = self.userFeats[self.userFeats[master].isin(user_ids)]
            self.userFeats = self.userFeats.sort_values([master])

        self.map_col(
            master, self.userMaps, 
            pools=(self.interactions, self.userFeats)
        )

    def item2token(self):
        infoLogger(f"[Converter] >>> Map item ID to Token ...")
        item_ids = sorted(self.interactions[ITEM.name].unique().tolist())
        self.itemCount = len(item_ids)

        item_tokens = range(len(item_ids))
        self.itemMaps = dict(
            zip(item_ids, item_tokens)
        )

        if self.itemFeats is not None:
            self.itemFeats = self.itemFeats[self.itemFeats[ITEM.name].isin(item_ids)]
            self.itemFeats = self.itemFeats.sort_values([ITEM.name])

        self.map_col(
            ITEM.name, self.itemMaps, 
            pools=(self.interactions, self.itemFeats)
        )

    def sort_by_timestamp(self, df: pd.DataFrame, master: Optional[str] = USER.name):
        try:
            if master is not None:
                df = df.sort_values(by=[master, TIMESTAMP.name])
                infoLogger(f"[Converter] >>> Sort by [{master}] [{TIMESTAMP.name}] ...")
            else:
                df = df.sort_values(by=[TIMESTAMP.name])
                infoLogger(f"[Converter] >>> Sort by [{TIMESTAMP.name}] ...")
        except KeyError:
            raise KeyError(f"{master} or {TIMESTAMP.name} is not in dataframe ...")
        finally:
            return df

    def split_by_ROU(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by ROU (Ratio On User): {ratios} ...")
        traingroups = []
        validgroups = []
        testgroups = []
        markers = np.cumsum(ratios)
        for _, group in self.interactions.groupby(USER.name):
            if len(group) == 0:
                continue
            l = max(floor(markers[0] * len(group) / markers[-1]), 1)
            r = floor(markers[1] * len(group) / markers[-1])
            traingroups.append(group[:l])
            if l < r:
                validgroups.append(group[l:r])
            if r < len(group):
                testgroups.append(group[r:])

        self.trainiter = pd.concat(traingroups)
        self.validiter = pd.concat(validgroups)
        self.testiter = pd.concat(testgroups)

        return ''.join(map(str, ratios)) + '_ROU'

    def split_by_RAU(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by RAU (Ratio and At least one on User): {ratios} ...")
        traingroups = []
        validgroups = []
        testgroups = []
        markers = np.cumsum(ratios)
        for _, group in self.interactions.groupby(USER.name):
            if len(group) == 0:
                continue
            l = floor(markers[0] * len(group) / markers[-1])
            l = max(min(l, len(group) - 2), 1)
            r = floor(markers[1] * len(group) / markers[-1])
            r = min(r, len(group) - 1)
            traingroups.append(group[:l])
            if l < r:
                validgroups.append(group[l:r])
            testgroups.append(group[r:])

        self.trainiter = pd.concat(traingroups)
        self.validiter = pd.concat(validgroups)
        self.testiter = pd.concat(testgroups)

        return ''.join(map(str, ratios)) + '_RAU'

    def split_by_ROD(self, ratios: Iterable = (8, 1, 1)):
        infoLogger(f"[Converter] >>> Split by ROT (Ratio On Dataset): {ratios} ...")
        self.interactions = self.sort_by_timestamp(
            self.interactions, master=None
        )
        markers = np.floor(
            (np.cumsum(ratios) / np.sum(ratios)) * len(self.interactions)
        ).astype(int)

        self.trainiter = self.interactions.iloc[:markers[0]]
        self.validiter = self.interactions.iloc[markers[0]:markers[1]]
        self.testiter = self.interactions.iloc[markers[1]:]

        return ''.join(map(str, ratios)) + '_ROD'

    def split_by_LOU(self):
        infoLogger(f"[Converter] >>> Split by LOU (Leave-one-out On User) ...")
        traingroups = []
        validgroups = []
        testgroups = []
        for _, group in self.interactions.groupby(USER.name):
            if len(group) == 0:
                continue
            if len(group) <= 3:
                # Note that sequence with len(group) == 3 cannot be splited into train/valid/test,
                # because train sequence needs at least length >= 2 for prediction.
                traingroups.append(group)
            else:
                traingroups.append(group[:-2])
                validgroups.append(group[-2:-1])
                testgroups.append(group[-1:])

        self.trainiter = pd.concat(traingroups)
        self.validiter = pd.concat(validgroups)
        self.testiter = pd.concat(testgroups)

        return '_LOU'

    def split_by_DOU(self, days: int = 1):
        infoLogger(f"[Converter] >>> Split by DOU (Day On User): {days} ...")

        seconds_per_day = 86400
        seconds = seconds_per_day * days
        last_date = self.interactions[TIMESTAMP.name].max().item()

        # Group interactions by user and calculate user timestamps
        user_timestamps = self.interactions.groupby(USER.name)[TIMESTAMP.name].max().sort_values()

        # Split interactions into train, validation, and test sets based on user timestamps
        traingroups = user_timestamps[user_timestamps < (last_date - 2 * seconds)].index
        validgroups = user_timestamps[(user_timestamps >= (last_date - 2 * seconds)) & (user_timestamps < (last_date - seconds))].index
        testgroups = user_timestamps[user_timestamps >= (last_date - seconds)].index

        assert len(traingroups) >= 0, f"The given `days` of {days} leads to zero-size trainsets ..."
        assert len(validgroups) >= 0, f"The given `days` of {days} leads to zero-size validsets ..."
        assert len(testgroups) >= 0, f"The given `days` of {days} leads to zero-size testsets ..."

        self.trainiter = self.interactions[self.interactions[USER.name].isin(traingroups)]
        self.validiter = self.interactions[self.interactions[USER.name].isin(validgroups)]
        self.testiter = self.interactions[self.interactions[USER.name].isin(testgroups)]

        return f"{days}_DOU"

    def split_by_DOD(self, days: int = 1):
        infoLogger(f"[Converter] >>> Split by DOD (Day On Dataset): {days} ...")

        seconds_per_day = 86400
        seconds = seconds_per_day * days
        timestamps =  self.interactions[TIMESTAMP.name]
        last_date = self.interactions[TIMESTAMP.name].max().item()

        # Split interactions into train, validation, and test sets based on user timestamps
        traintimes  = timestamps < (last_date - 2 * seconds) 
        validtimes = (timestamps >= (last_date - 2 * seconds)) & (timestamps < (last_date - seconds))
        testtimes = timestamps >= (last_date - seconds)

        assert traintimes.any(), f"The given `days` of {days} leads to zero-size trainsets ..."
        assert validtimes.any(), f"The given `days` of {days} leads to zero-size validsets ..."
        assert testtimes.any(), f"The given `days` of {days} leads to zero-size testsets ..."

        self.trainiter = self.interactions[traintimes]
        self.validiter = self.interactions[validtimes]
        self.testiter = self.interactions[testtimes]

        return f"{days}_DOD"

    def resort_iters(self):
        infoLogger(f"[Converter] >>> Resort for train|valid|test iters ...")
        self.trainiter = self.sort_by_timestamp(
            self.trainiter.reset_index(drop=True),
            master=USER.name
        )
        self.validiter = self.sort_by_timestamp(
            self.validiter.reset_index(drop=True),
            master=USER.name
        )
        self.testiter = self.sort_by_timestamp(
            self.testiter.reset_index(drop=True),
            master=USER.name
        )

    def save(self, path: str):
        mkdirs(path)

        infoLogger(f"[Converter] >>> Save `train.txt' to {path} ...")
        df = pd.DataFrame(self.trainiter)
        df.to_csv(os.path.join(path, 'train.txt'), sep='\t', index=False)

        infoLogger(f"[Converter] >>> Save `valid.txt' to {path} ...")
        df = pd.DataFrame(self.validiter)
        df.to_csv(os.path.join(path, 'valid.txt'), sep='\t', index=False)

        infoLogger(f"[Converter] >>> Save `test.txt' to {path} ...")
        df = pd.DataFrame(self.testiter)
        df.to_csv(os.path.join(path, 'test.txt'), sep='\t', index=False)

        if self.userFeats is not None:
            infoLogger(f"[Converter] >>> Save `user.txt' to {path} ...")
            self.userFeats.to_csv(
                os.path.join(path, 'user.txt'),
                sep='\t', index=False
            )

        if self.itemFeats is not None:
            infoLogger(f"[Converter] >>> Save `item.txt' to {path} ...")
            self.itemFeats.to_csv(
                os.path.join(path, 'item.txt'),
                sep='\t', index=False
            )

        self.summary()

    def make_dataset(
        self,
        kcore4user: int = 10, kcore4item: int = 10, star4pos: int = 0,
        splitting: Literal['ROU', 'ROD', 'LOU', 'DOU', 'DOD'] = 'ROU', 
        ratios: Tuple[int, int, int] = (8, 1, 1), days: int = 1,
        strict: bool = True,
    ):
        r"""
        Make dataset.

        Parameters:
        -----------
        kcore4user: int, default to 10
            Select kcore interactions according to User.
        kcore4item: int, default to 10
            Select kcore interactions according to Item.
        star4pos: int, default to 0
            Select interactions with `Rating >= star4pos'.
        splitting: str ('ROU', 'ROD', 'LOU', 'DOU', 'DOD')
            `ROU`: Ratio on User
            `RAU`: Ratio and At least one on User
            `ROD`: Ratio on Dataset
            `LOU`: Leave-one-out on User
            `DOU`: Day on User
            `DOD`: Day on Dataset
        ratios: Tuple[int, int, int], default to (8, 1, 1)
            ROU|ROD: The ratios of training|validation|test set.
        days: int
            DOU|DOD: The days remained for validation|test set
        strict: bool, default to `True`
            `True`: strictly filter by core
            `False`: filter by core only once
        """
        assert len(ratios) == 3, f"'ratios' should in length of 3 but a length of {len(ratios)} is received ..."
        self.load()
        self.filter_by_rating(low=star4pos, high=None)
        self.filter_by_core(low4user=kcore4user, low4item=kcore4item, strict=strict)
        self.user2token()
        self.item2token()
        self.interactions = self.sort_by_timestamp(self.interactions)
        if splitting == 'ROU':
            s = self.split_by_ROU(ratios)
        elif splitting == 'RAU':
            s = self.split_by_RAU(ratios)
        elif splitting == 'ROD':
            s = self.split_by_ROD(ratios)
        elif splitting == 'LOU':
            s = self.split_by_LOU()
        elif splitting == 'DOU':
            s = self.split_by_DOU(days)
        elif splitting == 'DOD':
            s = self.split_by_DOD(days)
        else:
            raise NotImplementedError(f"Invalid splitting method: {splitting}")
        self.resort_iters()

        code = f"{kcore4user}{kcore4item}{star4pos}{s}"
        path = os.path.join(
            self.root, 'Processed',
            '_'.join([self.dataset, code])
        )

        self.save(path)

    def summary(self):
        from prettytable import PrettyTable
        table = PrettyTable(['#User', '#Item', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        trainsize = len(self.trainiter)
        validsize = len(self.validiter)
        testsize = len(self.testiter)
        table.add_row([
            self.userCount, self.itemCount, trainsize + validsize + testsize,
            trainsize, validsize, testsize,
            (trainsize + validsize + testsize) / (self.userCount * self.itemCount)
        ])
        infoLogger(table)


URLS = {
    # =====================================Amazon2014=====================================
    "Amazon2014APPs": "https://zenodo.org/records/10995912/files/Amazon2014Apps.zip",
    "Amazon2014Automotive": "https://zenodo.org/records/10995912/files/Amazon2014Automotive.zip",
    "Amazon2014Baby": "https://zenodo.org/records/10995912/files/Amazon2014Baby.zip",
    "Amazon2014Beauty": "https://zenodo.org/records/10995912/files/Amazon2014Beauty.zip",
    "Amazon2014Books": "https://zenodo.org/records/10995912/files/Amazon2014Books.zip",
    "Amazon2014CDs": "https://zenodo.org/records/10995912/files/Amazon2014CDs.zip",
    "Amazon2014Cell": "https://zenodo.org/records/10995912/files/Amazon2014Cell.zip",
    "Amazon2014Clothing": "https://zenodo.org/records/10995912/files/Amazon2014Clothing.zip",
    "Amazon2014Digital": "https://zenodo.org/records/10995912/files/Amazon2014Digital.zip",
    "Amazon2014Electronics": "https://zenodo.org/records/10995912/files/Amazon2014Electronics.zip",
    "Amazon2014Grocery": "https://zenodo.org/records/10995912/files/Amazon2014Grocery.zip",
    "Amazon2014Health": "https://zenodo.org/records/10995912/files/Amazon2014Health.zip",
    "Amazon2014Home": "https://zenodo.org/records/10995912/files/Amazon2014Home.zip",
    "Amazon2014Instant": "https://zenodo.org/records/10995912/files/Amazon2014Instant.zip",
    "Amazon2014Kindle": "https://zenodo.org/records/10995912/files/Amazon2014Kindle.zip",
    "Amazon2014Movies": "https://zenodo.org/records/10995912/files/Amazon2014Movies.zip",
    "Amazon2014Musical": "https://zenodo.org/records/10995912/files/Amazon2014Musical.zip",
    "Amazon2014Office": "https://zenodo.org/records/10995912/files/Amazon2014Office.zip",
    "Amazon2014Patio": "https://zenodo.org/records/10995912/files/Amazon2014Patio.zip",
    "Amazon2014Pet": "https://zenodo.org/records/10995912/files/Amazon2014Pet.zip",
    "Amazon2014Sports": "https://zenodo.org/records/10995912/files/Amazon2014Sports.zip",
    "Amazon2014Tools": "https://zenodo.org/records/10995912/files/Amazon2014Tools.zip",
    "Amazon2014Toys": "https://zenodo.org/records/10995912/files/Amazon2014Toys.zip",
    "Amazon2014Video": "https://zenodo.org/records/10995912/files/Amazon2014Video.zip",

    # =====================================Amazon2018=====================================
    "Amazon2018AllBeauty": "https://zenodo.org/records/10997743/files/Amazon2018AllBeauty.zip",
    "Amazon2018Appliances": "https://zenodo.org/records/10997743/files/Amazon2018Appliances.zip",
    "Amazon2018Arts": "https://zenodo.org/records/10997743/files/Amazon2018Arts.zip",
    "Amazon2018Automotive": "https://zenodo.org/records/10997743/files/Amazon2018Automotive.zip",
    "Amazon2018Books": "https://zenodo.org/records/10997743/files/Amazon2018Books.zip",
    "Amazon2018CDs": "https://zenodo.org/records/10997743/files/Amazon2018CDs.zip",
    "Amazon2018Cell": "https://zenodo.org/records/10997743/files/Amazon2018Cell.zip",
    "Amazon2018Clothing": "https://zenodo.org/records/10997743/files/Amazon2018Clothing.zip",
    "Amazon2018Digital": "https://zenodo.org/records/10997743/files/Amazon2018Digital.zip",
    "Amazon2018Electronics": "https://zenodo.org/records/10997743/files/Amazon2018Electronics.zip",
    "Amazon2018Fashion": "https://zenodo.org/records/10997743/files/Amazon2018Fashion.zip",
    "Amazon2018Gift": "https://zenodo.org/records/10997743/files/Amazon2018Gift.zip",
    "Amazon2018Grocery": "https://zenodo.org/records/10997743/files/Amazon2018Grocery.zip",
    "Amazon2018Home": "https://zenodo.org/records/10997743/files/Amazon2018Home.zip",
    "Amazon2018Industrial": "https://zenodo.org/records/10997743/files/Amazon2018Industrial.zip",
    "Amazon2018Kindle": "https://zenodo.org/records/10997743/files/Amazon2018Kindle.zip",
    "Amazon2018Luxury": "https://zenodo.org/records/10997743/files/Amazon2018Luxury.zip",
    "Amazon2018Magazine": "https://zenodo.org/records/10997743/files/Amazon2018Magazine.zip",
    "Amazon2018Movies": "https://zenodo.org/records/10997743/files/Amazon2018Movies.zip",
    "Amazon2018Musical": "https://zenodo.org/records/10997743/files/Amazon2018Musical.zip",
    "Amazon2018Office": "https://zenodo.org/records/10997743/files/Amazon2018Office.zip",
    "Amazon2018Patio": "https://zenodo.org/records/10997743/files/Amazon2018Patio.zip",
    "Amazon2018Pet": "https://zenodo.org/records/10997743/files/Amazon2018Pet.zip",
    "Amazon2018Prime": "https://zenodo.org/records/10997743/files/Amazon2018Prime.zip",
    "Amazon2018Software": "https://zenodo.org/records/10997743/files/Amazon2018Software.zip",
    "Amazon2018Sports": "https://zenodo.org/records/10997743/files/Amazon2018Sports.zip",
    "Amazon2018Tools": "https://zenodo.org/records/10997743/files/Amazon2018Tools.zip",
    "Amazon2018Toys": "https://zenodo.org/records/10997743/files/Amazon2018Toys.zip",
    "Amazon2018Video": "https://zenodo.org/records/10997743/files/Amazon2018Video.zip",

    # =====================================MovieLens=====================================
    "MovieLens100K": "https://zenodo.org/records/10998034/files/MovieLens100K.zip",
    "MovieLens10M": "https://zenodo.org/records/10998034/files/MovieLens10M.zip",
    "MovieLens1M": "https://zenodo.org/records/10998034/files/MovieLens1M.zip",
    "MovieLens20M": "https://zenodo.org/records/10998034/files/MovieLens20M.zip",

    # =====================================Tmall=====================================
    "Tmall2016Buy": "https://zenodo.org/records/10998081/files/Tmall2016Buy.zip",
    "Tmall2016Click": "https://zenodo.org/records/10998081/files/Tmall2016Click.zip",

    # =====================================Gowalla=====================================
    "Gowalla2010": "https://zenodo.org/records/10997653/files/Gowalla2010.zip",

    # =====================================Yelp=====================================
    "Yelp2018": "https://zenodo.org/records/10998102/files/Yelp2018.zip",
    "Yelp2021": "https://zenodo.org/records/10998102/files/Yelp2021.zip",
    "Yelp2022": "https://zenodo.org/records/10998102/files/Yelp2022.zip",

    # =====================================Steam=====================================
    "Steam": "https://zenodo.org/records/10998197/files/Steam.zip",

    # =====================================Retailrocket=====================================
    "RetailrocketAddtocart": "https://zenodo.org/records/10998222/files/RetailrocketAddtocart.zip",
    "RetailrocketTransaction": "https://zenodo.org/records/10998222/files/RetailrocketTransaction.zip",
    "RetailrocketView": "https://zenodo.org/records/10998222/files/RetailrocketView.zip",

    # =====================================YahooMusic=====================================
    "YahooMusicR1": "https://zenodo.org/records/10998284/files/YahooMusicR1.zip",
}