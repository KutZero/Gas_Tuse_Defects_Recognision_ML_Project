import os
import re
import sys
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import ticker

#import custom_modules.pipe_data as pidf
import custom_modules.data_worker as dw

os.environ["ROCM_PATH"] = "/opt/rocm"
# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(name)s :: %(funcName)20s() :: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)