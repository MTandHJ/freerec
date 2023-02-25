

# Refer to
# https://github.com/RUCAIBox/RecDatasets/blob/master/conversion_tools/usage/Yelp.md
# for yelp2018.inter:
# user_id:token item_id:token rating:float timestamp:float useful:float funny:float cool:float review_id:token
# %%

import numpy as np
import os
import pandas as pd
import torchdata.datapipes as dp
from math import ceil, floor
from collections import defaultdict
from itertools import chain

# %%

#==============================Config==============================

path = r"E:\Desktop\data\General\Yelp2018"
dataset = "yelp2018.inter"
kcore_user = 10 # select the user interacted >=k items
kcore_item = 10 # select the item interacted >=k users
threshold_of_star = 4 # select pairs with star >= k
ratios = (8, 1, 1) # train:valid:test
# %%
datapipe = dp.iter.FileLister(path)
datapipe = datapipe.filter(filter_fn=lambda file_: file_.endswith(dataset))
datapipe = datapipe.open_files(mode='rt')
datapipe = datapipe.parse_csv(delimiter='\t', skip_lines=1)
datapipe = datapipe.map(lambda row: (str(row[0]), str(row[1]), float(row[2]), float(row[3]))) # (User, Item, star, Timestamp)

# %%

#==============================filter out repeated pairs and low-star pairs==============================

data = []
visited = set()

for user, item, star, timestamp in datapipe:
    if (user, item) in visited:
        continue
    elif star >= threshold_of_star:
        data.append((user, item, timestamp))
        visited.add((user, item))

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
# datasize: 3476663
# datasize: 1330032
# datasize: 1137568
# datasize: 1061294
# datasize: 1039587
# datasize: 1028944
# datasize: 1025485
# datasize: 1023654
# datasize: 1023135
# datasize: 1022829
# datasize: 1022712
# datasize: 1022649
# datasize: 1022622
# datasize: 1022613
# datasize: 1022604

# %%

#==============================sort by timestamp==============================

data = sorted(data, key=lambda row: (row[0], row[2])) # (User, Item)


#==============================map str to int==============================

users, items, _ = zip(*data)
users, items = set(users), set(items)
userMap = dict(zip(users, range(len(users))))
itemMap = dict(zip(items, range(len(items))))
userCount = len(users)
itemCount = len(items)

print(f"#Users: {len(users)} #Items: {len(items)}")
# Out: 
# #Users: 41801 #Items: 26512

data = list(map(
    lambda row: (userMap[row[0]], itemMap[row[1]]),
    data
))

#==============================group by user==============================

data_by_user = defaultdict(list)
for row in data:
    data_by_user[row[0]].append((row[0], row[1])) # (User, Item)


# %%



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
# #Train: 832513 #Valid: 102190 #Test: 87901

# %%


#==============================Saving==============================

df = pd.DataFrame(trainset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'train.txt'), sep='\t', index=False)

df = pd.DataFrame(validset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'valid.txt'), sep='\t', index=False)

df = pd.DataFrame(testset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'test.txt'), sep='\t', index=False)
# %%
