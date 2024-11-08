""" Модуль для всяческой обработки данных перед обучением модели.

Функции:

Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import logging
import itertools
import pandas as pd
import numpy as np
import os
import re
from pydantic import validate_call, PositiveInt

# create logger
logger = logging.getLogger('main.'+__name__)

@validate_call(config=dict(arbitrary_types_allowed=True))
def get_batch_generator(generator, batch_size: PositiveInt):
    """Transform np.ndarray or simple data type generator into ndarray batch generator"""
    #if batch_size < 1:
    #    raise ValueError('Batch size should be bigger than 1')
    # check is generator itarable
    try:
        iterator = iter(generator)
    except TypeError:
        raise TypeError('The generator param should be iterable')
    
    batch = list()
    i = 0
    try:
        while True:
            if i < batch_size:
                batch.append(next(generator))
                i+=1
            else:
                res = np.stack(batch) if type(batch[0]) == np.ndarray else np.array(batch)
                i=0
                batch = list()
                yield res
    except:
        if batch:
            yield np.stack(batch) if type(batch[0]) == np.ndarray else np.array(batch)


@validate_call(config=dict(arbitrary_types_allowed=True))
def get_crop_generator(arr: np.ndarray, crop_size: PositiveInt, crop_step: PositiveInt):
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

    Yields
    -------
    out : numpy.ndarray
        The augmented numpy array with crops data
    """
    for i in range(0, arr.shape[0] - crop_size + 1, crop_step):  
        for j in range(0, arr.shape[1] - crop_size + 1, crop_step):  
            yield arr[i:i+crop_size, j:j+crop_size]

@validate_call(config=dict(arbitrary_types_allowed=True))
def get_augmented_crop_generator(arr: np.ndarray, crop_size: PositiveInt, crop_step: PositiveInt):
    """
    Augnment data of the arr which store crops data.
    Used augmentations: rotation for 90 degree (4 times);
    horizontal and vertical mirroring.
    
    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with crops data
    crop_size: PositiveInt
        The size of the square crop side
    crop_step: PositiveInt
        The size of the sliding window
        step making crops
    
    Yields
    -------
    out : numpy.ndarray
        The augmented numpy array with crops data
    """
    message = f'''
    The input array cells: {arr.shape} 
    The crop size: {crop_size} 
    The crop step: {crop_step}'''
    
    arr = np.concatenate([arr, np.flip(arr,1)],axis=0)
    arr = np.concatenate([arr, np.flip(arr,0)],axis=0)

    arr1 = np.concatenate([arr, np.rot90(arr,2,[0,1])],axis=0)
    arr2 = np.concatenate([np.rot90(arr,1,[0,1]), np.rot90(arr,3,[0,1])],axis=1)

    arrs = np.split(arr1, 8, axis=0) + np.split(arr2, 8, axis=1)
    
    logger.debug(message)
    return itertools.chain(*[get_crop_generator(arr, crop_size, crop_step) for arr in arrs])

@validate_call(config=dict(arbitrary_types_allowed=True))
def get_x_and_y_data(path_to_data_file: str,
                     path_to_defects_file: str,
                     path_to_pipe_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data from set of .csv files

    Parameters
    ----------
    path_to_data_file : str
        The path to data file like "*_data.csv"
    path_to_defects_file : str
        The path to data file like "*_defects.csv"
    path_to_pipe_file : str
        The path to data file like "*_pipe.csv"

    Returns
    -------
    data_df : pandas.DataFrame
        The dataframe with data got from detectors
    defects_df : pandas.DataFrame
        The dataframe with data got from specialists
        about defects depths and locations for data_df
        
    """
    data_df = _get_df_from_data_file(path_to_data_file)
    defects_df = _get_df_from_defects_file(path_to_defects_file)
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

    logger.debug(f"""
    Read detectors data shape: {data_df.shape}
    Read defect data shape: {defects_df.shape}""")
    return data_df, defects_df

@validate_call
def _get_df_from_defects_file(path_to_defects_file: os.PathLike) -> pd.DataFrame:
    """Read data file like "*_defects.csv" and returns
       pandas dataframe with preprocessed data from it"""
    using_columns = ['row_min', 'row_max', 'detector_min', 
                     'detector_max', 'fea_depth']
    df = pd.read_csv(path_to_defects_file, delimiter=';')
    return df[using_columns]

@validate_call
def _split_cell_string_value_to_numpy_array_of_64_values(df_cell_value: str) -> np.ndarray:
    """Converte all data cells values from given pandas dataframe from
    string (describes 2D values array) to 1D float numpy array of 64 items"""
    num_pars = re.findall(r'(-?\d+(\.\d+)*)*\s*:\s*(-?\d+(\.\d+)*)*', df_cell_value)

    if len([item[0] for item in num_pars if item[0] and item[2]]) == 0:
        #logger.debug(f"""Got cell value without any full time-value pars""")
        return np.zeros((64))
    
    for item in num_pars:
        if not item[0] or not item[2]:
            logger.debug(f"""Got input df cell str with uncompleted 
            time-amplitude value pars. Uncompleted pars deleted.
            Input str: {df_cell_value}""")
            
    num_pars = np.array([[item[0], item[2]] for item in num_pars if item[0] and item[2]]).astype(float)

    if num_pars.shape[0] > 32:
        raise ValueError(f'Too much time-amplitude values pars in a cell. Got: {num_pars.shape[0]}. Max: 32')
    
    time_vals = num_pars[:,0]
    amp_vals = num_pars[:,1]
    
    time_vals = np.pad(time_vals, (abs(time_vals.size-32), 0), constant_values=(0))
    amp_vals = np.pad(amp_vals, (abs(amp_vals.size-32), 0), constant_values=(0))
    return np.concatenate((time_vals, amp_vals), axis=0)

@validate_call
def _get_df_from_data_file(path_to_data_file: os.PathLike) -> str:
    """Read data file like "*_data.csv" and returns
       pandas dataframe with preprocessed data from it"""
    df = pd.read_csv(path_to_data_file, delimiter=';').astype(str)
    df = df.drop(['position'], axis=1)
    df = df.set_index('row')
    df = df.map(_split_cell_string_value_to_numpy_array_of_64_values)
    return df