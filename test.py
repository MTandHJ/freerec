


# %%
import torch
import freerec
# %%

# dataset
from freerec.data.tags import ID
dataset = 'MovieLens1M_m2'

basepipe = getattr(freerec.data.datasets, dataset)("../data")
User, Item = basepipe.fields[ID]
print(basepipe)

data = dict()
data['#User'] = User.count
data['#Item'] = Item.count
data['#Train'] = basepipe.train().datasize
data['#Test'] = basepipe.test().datasize
data['#Interactions'] = basepipe.train().datasize + basepipe.test().datasize
data['density'] = data['#Interactions'] / (User.count * Item.count)
print(data)


# %%
