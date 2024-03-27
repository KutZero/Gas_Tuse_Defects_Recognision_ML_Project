import os
import re
import sys
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
import itertools

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import ticker
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Reshape, Input, Dense, Flatten, Conv2D, Dropout, 
    Conv2DTranspose, Lambda, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization)

#import custom_modules.pipe_data as pidf
import custom_modules.data_worker as dw

os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "enable"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.ERROR)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(name)s :: %(funcName)20s() :: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)