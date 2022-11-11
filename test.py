


# %%
import torch
import freerec
# %%

# dataset
datasets = ['Gowalla_m1', 'Yelp18_m1', 'AmazonBooks_m1', 'AmazonCDs_m1', 'AmazonMovies_m1', 'AmazonBeauty_m1', 'AmazonElectronics_m1']

for dataset in datasets:
    basepipe = getattr(freerec.data.datasets, dataset)("../data")


# %%