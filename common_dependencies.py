import os
import re
import sys
import logging
import itertools
import numpy as np
import pandas as pd
import pathlib

#os.environ["ROCM_PATH"] = "/opt/rocm"
#os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "enable"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    concatenate, Flatten, Dense, UpSampling2D,
    BatchNormalization
)

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle

from IPython.display import display
from typing import NamedTuple

import custom_modules.data_worker as dw
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

PATH_TO_DATA = {
    'run_1': 
        ('data/original_data/run_1/run1_WM32_data.csv',
        'data/original_data/run_1/run1_WM32_defects.csv',
        'data/original_data/run_1/run1_WM32_pipe.csv'),
    'run_2':
        ('data/original_data/run_2/run2_WM32_data.csv',
        'data/original_data/run_2/run2_WM32_defects.csv',
        'data/original_data/run_2/run2_WM32_pipe.csv')
}

DataCrop = NamedTuple("DataCrop", [('left', int), ('top', int), ('width', int), ('height', int)])
SlidingCrop = NamedTuple("SlidingCrop", [('crop_size', int), ('crop_step', int)])
DatasetPartDescription = NamedTuple("DatasetPartDescription", 
                                    [('data_path_tuple', tuple[str,str,str]), 
                                     ('file_data_crop', DataCrop), 
                                     ('sliding_crop', SlidingCrop),
                                     ('data_x_shift', int)])

def get_dataset_gen(desc_part: DatasetPartDescription):
    logger.debug("start")

    left = desc_part.file_data_crop.left
    top = desc_part.file_data_crop.top
    width = desc_part.file_data_crop.width
    height = desc_part.file_data_crop.height
    crop_size = desc_part.sliding_crop.crop_size
    crop_step = desc_part.sliding_crop.crop_step
    
    x_df, y_df = dw.get_x_and_y_data(*desc_part.data_path_tuple)
    x_df = dw.roll_df(x_df, desc_part.data_x_shift, 1)
    y_df = dw.roll_df(y_df, desc_part.data_x_shift, 1)

    x_arr = dw.df_to_numpy(x_df)
    y_arr = y_df.to_numpy()
    
    # encode x_arr
    '''model_ver = 13
    model_num = 1
    PATH_TO_MODEL = next(pathlib.Path(f'networks/AE/').rglob(f'id=v{model_ver:04}n{model_num:04}*.keras'))
    model = keras.models.load_model(PATH_TO_MODEL)
    model = keras.Model(inputs=model.input, outputs=min([layer.output for layer in model.layers], key=lambda x: x.shape[1]))

    pred = model.predict(np.reshape(x_arr,(-1,64)), verbose=0)
    x_arr = np.reshape(pred, (*x_arr.shape[:2],-1))'''
    
    # normalize_data
    # standardize_data
    x_arr = np.concatenate([dw.normalize_data(x_arr[:,:,:32]), dw.normalize_data(x_arr[:,:,32:])],axis=2)
    #x_arr = dw.normalize_data(x_arr)
    y_arr = dw.normalize_data(y_arr)
    
    x_arr = x_arr[top:top+height, left:left+width]
    y_arr = y_arr[top:top+height, left:left+width]

    x_arr = dw.extend_ndarray_for_prediction(x_arr, crop_size, only_horizontal=True)
    y_arr = dw.extend_ndarray_for_prediction(y_arr, crop_siz, only_horizontal=Truee)

    x_arr = dw.extend_ndarray_for_crops_dividing(x_arr, crop_size, crop_step)
    y_arr = dw.extend_ndarray_for_crops_dividing(y_arr, crop_size, crop_step)
    
    x_crops_gen = dw.get_crop_generator(x_arr, crop_size, crop_step)
    #x_time_crops_gen = dw.get_crop_generator(x_arr[:,:,:32], crop_size, crop_step)
    #x_amp_crops_gen = dw.get_crop_generator(x_arr[:,:,32:], crop_size, crop_step)
    y_crops_gen = dw.get_crop_generator(y_arr, crop_size, crop_step)
    
    y_binary_gen = (np.array([1]) if np.sum(crop > 0) else np.array([0]) for crop in dw.get_crop_generator(y_arr, crop_size, crop_step))
    #y_depth_gen = (np.max(crop) for crop in dw.get_crop_generator(y_arr, crop_size, crop_step))

    logger.debug("end")
    #return x_time_crops_gen, x_amp_crops_gen, y_bool_crops_gen, y_binary_gen, y_depth_gen
    return (x_crops_gen, y_crops_gen, y_binary_gen)

def chain_dataset_gens(datasets_desc: tuple[DatasetPartDescription]):
    # get output data tuple for each data part from dataset_desc
    data_tuples_arr = [get_dataset_gen(desc_part) for desc_part in datasets_desc]
    # concat data tuples throught 1 axis (like concat every first item from all tuple into one first item of result tuple, same to second and etc.)
    # output tuple has the same quantity of values as every tuple got from previous step but each element of it is concatination of all elements
    # from that position from all read tuples
    return (*[itertools.chain(*[arr for arr in item]) for item in zip(*data_tuples_arr)],)