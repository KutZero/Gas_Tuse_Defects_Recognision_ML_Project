__all__ = ['get_crop_generator', 'get_data_df', 'cast_df_to_2d', 'cast_df_to_3d']

import logging
import itertools
import pandas as pd
import numpy as np
import os
import re
from pydantic import validate_call, PositiveInt
from typing import Callable, Optional, Generator, Iterable
import pathlib

from custom_modules.data_worker import (
    DataPart,
    match_ndarray_for_crops_dividing,
)

# create logger
logger = logging.getLogger('main.'+__name__)

@validate_call(config=dict(arbitrary_types_allowed=True))
@validate_call(config=dict(arbitrary_types_allowed=True))
def cast_df_to_2d(df: pd.DataFrame, data_parts_dict = [('Time', 32), ('Amplitude', 32), ('DefectDepth', 1)]) -> pd.DataFrame:
    """Cast 3d pandas to 2d one
    
    Parameters
    ----------
    df: pandas.DataFrame
        A dataframe with data. Every cell - numpy array.
        Every numpy array have equal size.
    data_parts_dict: list[tuple[str,int]]
        A list defining every cell's numpy arrays dividing
        to subcolums. Default array have 65 values. First 32 - 
        time values, second 32 - amplitude values, last - defect
        depth. The data_parts_dict default is [('Time', 32), 
        ('Amplitude', 32), ('DefectDepth', 1)]. In that case
        65 values will be annotated with pandas.MultiIndex as:
        First 32 values under the Time column, second under the
        Amplitude column, second under the DefectDepth column on
        level named DataPart. Each subcolumn of every DataPart 
        level column has local id from 0 to value passed next to
        str in tuple from the data_parts_dict

    Returns
    -------
    df : pandas.DataFrame
        Pandas dataframe with multiindex and only
        one float/int value in cells.
    """
    if not set(['File', 'ScanNum']).issubset(df.index.names):
        raise ValueError(f'The df should have index with levels: "File", "ScanNum", but got: {df.index.names=}')
    if not set(['DetectorNum']).issubset(df.columns.names):
        raise ValueError(f'The df should have columns with levels: "DetectorNum", but got: {df.columns.names=}')
        
    df = df.copy()
    df = df.stack(-1, future_stack=True)
    
    tuples = []
    for tup in data_parts_dict:
        tuples = [*tuples, *[(tup[0], i) for i in range(tup[1])]]
        
    df = pd.DataFrame(data=np.stack(df.to_list()), index=df.index, 
                      columns=pd.MultiIndex.from_tuples(tuples, names=['DataPart','Id']))
    return df

@validate_call(config=dict(arbitrary_types_allowed=True))
def cast_df_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    if not list(df.index.names) == ['File', 'ScanNum', 'DetectorNum']:
        raise ValueError(f'The df should have index with levels: "File", "ScanNum", "DetectorNum", but got: {df.index.names=}')
    if not list(df.columns.names) == ['DataPart', 'Id']:
        raise ValueError(f'The df should have columns with levels: "DataPart", "Id", but got: {df.columns.names=}')
    if df.index.get_level_values('ScanNum').dtype.name != 'int64':
        raise ValueError(f"The df's index level 'ScanNum' should have dtype 'int64', but got: {df.index.get_level_values('ScanNum').dtype=}")
    if df.index.get_level_values('DetectorNum').dtype.name != 'int64':
        raise ValueError(f"The df's index level 'DetectorNum' should have dtype 'int64', but got: {df.index.get_level_values('DetectorNum').dtype=}")
    if df.columns.get_level_values('Id').dtype.name != 'int64':
        raise ValueError(f"The df's columns level 'Id' should have dtype 'int64', but got: {df.index.get_level_values('Id').dtype=}")
        
    df = df.copy()
    df = pd.Series(data=map(lambda x: np.array(x), df.to_numpy().tolist()), index=df.index)
    df = df.unstack('DetectorNum')
    return df


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_crop_generator(arr: np.ndarray, 
                       crop_size: PositiveInt, 
                       crop_step: PositiveInt, 
                       augmentations: bool = False) -> Iterable[np.array]:
    """
    Create generator for sliding window across arr with given step and crop size

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with crops data
    crop_size: PositiveInt
        The size of the square crop side
    crop_step: PositiveInt
        The size of the sliding window
        step making crops
    augmentations: bool
        Apply of not augmentations to the arr
        sliding crops. Augmentations to be used: 
        rotation for 90 degree (4 times); 
        horizontal and vertical mirroring.

    Yields
    -------
    out : numpy.ndarray
        The numpy array with sliding crops data
    """
    in_arr_shape = arr.shape
        
    logger.debug(f'''
    The input array cells: {in_arr_shape} 
    The crop size: {crop_size} 
    The crop step: {crop_step}''')

    arr = match_ndarray_for_crops_dividing(arr, crop_size, crop_step, mode='crop')

    if arr.shape != in_arr_shape:
        logger.warning(f'''
    The arr with shape = {in_arr_shape} cant be divided by {crop_size=} with {crop_step=}. 
    The custom_modules.data_worker.match_ndarray_for_crops_dividing() with mode="crop" used. The new
    arr shape={arr.shape}''')
    
    if augmentations:
        arr1 = np.concatenate([arr, np.flip(arr,0)],axis=0)
        arr1 = np.concatenate([arr1, np.flip(arr1,1)],axis=0)
        arr2 = np.rot90(arr1,1,[0,1])
        arrs = np.split(arr1, 4, axis=0) + np.split(arr2, 4, axis=1)

        for arr in arrs:
            for i in range(0, arr.shape[0] - crop_size + 1, crop_step):  
                for j in range(0, arr.shape[1] - crop_size + 1, crop_step):  
                    yield arr[i:i+crop_size, j:j+crop_size]

    else:
        for i in range(0, arr.shape[0] - crop_size + 1, crop_step):  
            for j in range(0, arr.shape[1] - crop_size + 1, crop_step):  
                yield arr[i:i+crop_size, j:j+crop_size]


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_data_df(path: os.PathLike, *args, **kvargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data from set of .csv files

    Parameters
    ----------
    path : os.PathLike
        The path to one scanning result folder or
        to folder with all scanning results folders.

    Returns
    -------
    data_df : pandas.DataFrame
    """
    run_names = next(os.walk(path))[1]
    if run_names:
        runs_list = []
        for run_name in run_names:
            runs_list.append(get_data_df(os.path.join(path, run_name), 
                                         *args, **kvargs))
        return pd.concat(runs_list, axis=0)
        
    data_df = _get_df_from_data_file(next(pathlib.Path(path).rglob('*_data.csv')),*args, **kvargs)
    defects_df = _get_df_from_defects_file(next(pathlib.Path(path).rglob('*_defects.csv')))
    
    # create defects depths mask
    # create base zeros dataframe with size like data_df
    base_df = pd.DataFrame(data = 0.0, index = data_df.index,
                           columns = data_df.columns)
    # read line-by-line defects_df
    # get defects location and mark by ones
    for row_name in defects_df.index.values.tolist():
        (row_min, row_max,
         detector_min, detector_max,
         fea_depth ) = defects_df.astype('object').loc[row_name].to_list()
        
        # mark defect location in base dataframe
        if (detector_min < detector_max):
            base_df.iloc[row_min:row_max+1,detector_min:detector_max+1] = fea_depth
            continue
    
        base_df.iloc[row_min:row_max+1,detector_min:data_df.shape[1]] = fea_depth
        base_df.iloc[row_min:row_max+1,:detector_max+1] = fea_depth
    defects_df = base_df
    
    for i in range(data_df.shape[0]):
        for j in range(data_df.shape[1]):
            data_df.iloc[i,j] = np.array([*data_df.iloc[i,j], defects_df.iloc[i,j]])

    data_df.index = pd.MultiIndex.from_product([[pathlib.Path(path).name,], 
                                                data_df.index.values.astype('int')], names=['File','ScanNum'])
    data_df.columns = data_df.columns.map(lambda x: x.split('_')[-1]).astype('int')
    data_df.columns.name = 'DetectorNum'
    
    logger.debug(f"""
    Read detectors data shape: {data_df.shape}""")
    return data_df
    

@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_df_from_defects_file(path_to_defects_file: os.PathLike) -> pd.DataFrame:
    """Read data file like "*_defects.csv" and returns
       pandas dataframe with preprocessed data from it"""
    using_columns = ['row_min', 'row_max', 'detector_min', 
                     'detector_max', 'fea_depth']
    df = pd.read_csv(path_to_defects_file, delimiter=';')
    return df[using_columns]


@validate_call(config=dict(arbitrary_types_allowed=True))
def _split_cell_string_value_to_numpy_array_of_64_values(df_cell_value: str, pad_type: str = 'start') -> np.ndarray:
    """Converte all data cells values from given pandas dataframe from
    string (describes 2D values array) to 1D float numpy array of 64 items"""
    num_pars = re.findall(r'(-?\d+(\.\d+)*)*\s*:\s*(-?\d+(\.\d+)*)*', df_cell_value)

    if not pad_type in ['start','end']:
        raise ValueError('The pad_type param should be either "start" or "end"')
    
    if len([item[0] for item in num_pars if item[0] and item[2]]) == 0:
        #logger.debug(f"""Got cell value without any full time-value pars""")
        return np.zeros((64))
    
    check = False
    for item in num_pars:
        if not item[0] or not item[2]:
            if not check:
                logger.warning("Got input df cell str with uncompleted" + 
                "time-amplitude value pars. Uncompleted pars deleted. " +
                f"Input str: {df_cell_value}")
            check = True
            logger.warning(f'The uncompleted par: ({item[1]}: {item[2]})')
            
    num_pars = np.array([[item[0], item[2]] for item in num_pars if item[0] and item[2]]).astype(float)

    if num_pars.shape[0] > 32:
        raise ValueError(f'Too much time-amplitude values pars in a cell. Got: {num_pars.shape[0]}. Max: 32')
    
    time_vals = num_pars[:,0]
    amp_vals = num_pars[:,1]

    if pad_type == 'start':
        time_vals = np.pad(time_vals, (abs(time_vals.size-32), 0), constant_values=(0))
        amp_vals = np.pad(amp_vals, (abs(amp_vals.size-32), 0), constant_values=(0))
    elif pad_type == 'end':
        time_vals = np.pad(time_vals, (0, abs(time_vals.size-32)), constant_values=(0))
        amp_vals = np.pad(amp_vals, (0, abs(amp_vals.size-32)), constant_values=(0))
    return np.concatenate((time_vals, amp_vals), axis=0)


@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_df_from_data_file(path_to_data_file: os.PathLike, *args, **kvargs) -> str:
    """Read data file like "*_data.csv" and returns
       pandas dataframe with preprocessed data from it"""
    df = pd.read_csv(path_to_data_file, delimiter=';').astype(str)
    df = df.drop(['position'], axis=1)
    df = df.set_index('row')
    df = df.map(lambda x: _split_cell_string_value_to_numpy_array_of_64_values(x, *args, **kvargs))
    return df