


# Refer to
# https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Gowalla.md
# for gowalla.inter:
# user_id:token item_id:token timestamp:float latitude:float longitude:float num_repeat:float
# %%

import os
import numpy as np
import pandas as pd
import torchdata.datapipes as dp
from math import ceil, floor
from collections import defaultdict
from itertools import chain
from freerec.utils import mkdirs

# %%

#==============================Config==============================

path = r"E:\Desktop\data\General\Gowalla"
dataset = "gowalla.inter"
saved_path = r"E:\Desktop\data\General\Gowalla"
kcore_user = 10 # select the user interacted >=k items
kcore_item = 10 # select the item interacted >=k users
ratios = (8, 1, 1) # train:valid:test
# %%

datapipe = dp.iter.FileLister(path)
datapipe = datapipe.filter(filter_fn=lambda file_: file_.endswith(dataset))
datapipe = datapipe.open_files(mode='rt')
datapipe = datapipe.parse_csv(delimiter='\t', skip_lines=1)
datapipe = datapipe.map(lambda row: (int(row[0]), int(row[1]), float(row[2]))) # (User, Item, Timestamp)

# %%

#==============================filter out repeated pairs==============================

data = []
visited = set()

for row in datapipe:
    if (row[0], row[1]) in visited:
        continue
    else:
        data.append(row)
        visited.add((int(row[0]), int(row[1])))

# %%

#==============================filter out 'inactive' users and items==============================

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
        if count_per_user[user] >= kcore_user:
            users.add(user)
        if count_per_item[item] >= kcore_item:
            items.add(item)
    data = list(filter(
        lambda row: row[0] in users and row[1] in items,
        data
    ))

# Out:
# datasize: 3981334
# datasize: 1339108
# datasize: 1174010
# datasize: 1090400
# datasize: 1064843
# datasize: 1047467
# datasize: 1040489
# datasize: 1035097
# datasize: 1032424
# datasize: 1030383
# datasize: 1029389
# datasize: 1028665
# datasize: 1028355
# datasize: 1028010
# datasize: 1027851
# datasize: 1027680
# datasize: 1027563
# datasize: 1027491
# datasize: 1027473
# datasize: 1027464


# %%

#==============================Sort by timestamp==============================

data = sorted(data, key=lambda row: (row[0], row[2])) # (User, Item)

#==============================Map int to id==============================

users, items, _ = zip(*data)
users, items = set(users), set(items)
userMap = dict(zip(users, range(len(users))))
itemMap = dict(zip(items, range(len(items))))
userCount = len(users)
itemCount = len(items)

print(f"#Users: {len(users)} #Items: {len(items)}")
# Out: 
# #Users: 29858 #Items: 40988

data = list(map(
    lambda row: (userMap[row[0]], itemMap[row[1]]),
    data
))

#==============================Group by user==============================
data_by_user = defaultdict(list)
for row in data:
    data_by_user[row[0]].append((row[0], row[1])) # (User, Item)


# %%

#==============================Splitting==============================

trainset = []
validset = []
testset = []
markers = np.cumsum(ratios)
for user in range(userCount):
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
# %%

trainset = list(chain(*trainset))
validset = list(chain(*validset))
testset = list(chain(*testset))

print(f"#Train: {len(trainset)} #Valid: {len(validset)} #Test: {len(testset)}")

# Out:
# #Train: 810128 #Valid: 100508 #Test: 116828

# %%

#==============================Saving==============================

mkdirs(saved_path)

df = pd.DataFrame(trainset, columns=['User', 'Item'])
df.to_csv(os.path.join(saved_path, 'train.txt'), sep='\t', index=False)

df = pd.DataFrame(validset, columns=['User', 'Item'])
df.to_csv(os.path.join(saved_path, 'valid.txt'), sep='\t', index=False)

df = pd.DataFrame(testset, columns=['User', 'Item'])
df.to_csv(os.path.join(saved_path, 'test.txt'), sep='\t', index=False)