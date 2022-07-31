





# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



import torch
import logging
from .dict2obj import Config



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "../data" # the path saving the data
DOWNLOAD = False # whether to download the data

SAVED_FILENAME = "paras.pt" # the filename of saved model paramters
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}-{time}"
TIMEFMT = "%m%d%H"

# logger
LOGGER = Config(
    name='RecSys', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)
