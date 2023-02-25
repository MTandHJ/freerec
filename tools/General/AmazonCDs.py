

# Refer to
# https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Amazon.md
# for Amazon_XXX.inter:
# user_id:token item_id:token rating:float timestamp:float
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

path = r"E:\Desktop\data\General\AmazonCDs"
dataset = "Amazon_CDs_and_Vinyl.inter"
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
# datasize: 3191727
# datasize: 631100
# datasize: 402226
# datasize: 333662
# datasize: 303875
# datasize: 289846
# datasize: 282139
# datasize: 278141
# datasize: 275467
# datasize: 274078
# datasize: 272988
# datasize: 272480
# datasize: 272212
# datasize: 272113
# datasize: 272086

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
# #Users: 109730 #Items: 96421

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
# #Users: 15592 #Items: 16184

# %%


#==============================Saving==============================

df = pd.DataFrame(trainset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'train.txt'), sep='\t', index=False)

df = pd.DataFrame(validset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'valid.txt'), sep='\t', index=False)

df = pd.DataFrame(testset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'test.txt'), sep='\t', index=False)
# %%
