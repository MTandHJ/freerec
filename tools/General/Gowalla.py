


# Refer to
# https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/Gowalla.md
# for gowalla.inter:
# user_id:token item_id:token timestamp:float latitude:float longitude:float num_repeat:float
# %%

import os
import pandas as pd
import torchdata.datapipes as dp
from collections import defaultdict
from itertools import chain

# %%

#==============================Config==============================

path = "..."
dataset = "gowalla.inter"
kcore_user = 10 # select the user interacted >=k items
kcore_item = 10 # select the item interacted >=k users

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
data_by_user = defaultdict(list)
for row in data:
    data_by_user[row[0]].append((row[0], row[1])) # (User, Item)

# %%

#==============================Splitting==============================

trainset = []
validset = []
testset = []

for user, pairs in data_by_user.items():
    testset.append(
        pairs[-len(pairs) // 10:]
    )
    pairs = pairs[:-len(pairs) // 10]
    validset.append(
        pairs[-len(pairs) // 9:]
    )
    trainset.append(
        pairs[:-len(pairs) // 9]
    )
# %%

trainset = list(chain(*trainset))
validset = list(chain(*validset))
testset = list(chain(*testset))

print(f"#Train: {len(trainset)} #Valid: {len(validset)} #Test: {len(testset)}")

# Out:
# #Train: 797566 #Valid: 113070 #Test: 116828

# %%

#==============================Saving==============================

df = pd.DataFrame(trainset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'train.txt'), sep='\t', index=False)

df = pd.DataFrame(validset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'valid.txt'), sep='\t', index=False)

df = pd.DataFrame(testset, columns=['User', 'Item'])
df.to_csv(os.path.join(path, 'test.txt'), sep='\t', index=False)