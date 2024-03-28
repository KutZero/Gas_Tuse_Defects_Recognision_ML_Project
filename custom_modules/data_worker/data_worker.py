""" Модуль для всяческой обработки данных перед обучением модели.

Функции:

Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import logging
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from typing import Union
import itertools

from typing_extensions import Annotated
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

PositiveInt = Annotated[int, Field(gt=0), AfterValidator(lambda x: int(x))]
PercentFloat = Annotated[float, Field(ge=0,le=1), AfterValidator(lambda x: float(x))]

# create logger
logger = logging.getLogger('main.'+__name__)

@validate_call(config=dict(arbitrary_types_allowed=True))
def get_crop_generator(arr: np.ndarray, crop_size: int, crop_step: int):
    """
    Creates generator for sliding window across arr with given step and crop size
    """
    for i in range(0, arr.shape[0] - crop_size + 1, crop_step):  
        for j in range(0, arr.shape[1] - crop_size + 1, crop_step):  
            yield arr[i:i+crop_size, j:j+crop_size]

def get_augmented_crop_generator(arr: np.ndarray, crop_size: int, crop_step: int) -> np.ndarray:
    """
    Augnment data of the arr which store crops data.
    Used augmentations: rotation for 90 degree (4 times);
    horizontal and vertical mirroring.
    
    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with crops data
    
    Returns
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
def calc_model_prediction_accuracy(pred_df: pd.DataFrame, 
                                   ref_df: pd.DataFrame,
                                   use_defect_depth: bool = False) -> float:
    """
    Calc model prediciton loss by dividing model prediction map from
    reference map. Then calc it summ and normalize it by dividing on df.shape[0] *
    df.shape[1]. So in ideal case you will get 0, what means that the model is 
    100% accurate. In the other hand the 1 output means that the model is awful

    Parameters
    ----------
    pred_df : pd.DataFrame
        The pandas dataframe with prediction map.
    ref_df : pd.DataFrame
        The pandas dataframe with reference map.
    use_defect_depth : bool
        The flag. If true defect zones in ref_df will store defect depth info.
        If false all defect zones will store value "1".

    Returns
    ----------
    model_loss: float
        The model test loss.
    
    """
    if not use_defect_depth:
        ref_df = ref_df.map(lambda x: 1 if x > 0 else 0)
    
    pred_arr = pred_df.to_numpy()
    ref_arr = ref_df.to_numpy()

    return np.sum(np.abs(pred_arr - ref_arr)) / (ref_df.shape[0] * ref_df.shape[1])
    

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

@validate_call(config=dict(arbitrary_types_allowed=True))
def df_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Reshape df with numpy.array in each cell of 64 items
    to numpy array of shape (df.shape[0], df.shape[1], 64)

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe with data got from detectors

    Returns
    -------
    out : numpy.ndarray
        The array of size (df.shape[0], df.shape[1], 64) with float values
        
    """
    df.map(_check_df_cell_is_correct_numpy_array)
    x = df.to_numpy()
    return np.stack([np.stack([x[i,j] for i in range(x.shape[0])],axis=0)
        for j in range(x.shape[1])],axis=1)


@validate_call(config=dict(arbitrary_types_allowed=True))
def normalize_data(arr: np.ndarray) -> np.ndarray:
    """
    Normalize the arr to (0-1) borders.

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with some float values

    Returns
    -------
    out : numpy.ndarray
        The numpy array with normalized some float values
    
    """
    max_val = arr.max()
    min_val = arr.min()

    arr = (arr - min_val) / (max_val - min_val)

    logger.debug(f"""
    The arr max before normalization: {max_val}
    The arr min before normalization: {min_val}
    The arr max after normalization: {arr.max()}
    The arr min after normalization: {arr.min()}""")

    return arr

@validate_call(config=dict(arbitrary_types_allowed=True))
def standardize_data(arr: np.ndarray) -> np.ndarray:
    """
    Standartize the arr so it max value less or equal than 1
    and min value greater or equal than -1.

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with some float values

    Returns
    -------
    out : numpy.ndarray
        The numpy array with standartized some float values
    
    """
    max_val = arr.max()
    min_val = arr.min()
    
    arr = np.divide(arr, max_val, out=np.zeros_like(arr), where=max_val!=0)

    logger.debug(f"""
    The arr max before standardization: {max_val}
    The arr min before standardization: {min_val}
    The arr max after standardization: {arr.max()}
    The arr min after standardization: {arr.min()}""")
    
    return arr

def _check_df_cell_is_correct_numpy_array(cell_value):
    """Check that every pandas dataframe cell is a flat numpy array of floats"""
    if not isinstance(cell_value, np.ndarray):
        raise TypeError(f'Every cell of the dataframe should store numpy array, but got: {type(cell_value)}')
    if cell_value.ndim > 1:
        raise ValueError(f'Every numpy array in the dataframe should be flat, but got shape: {cell_value.shape}')
    if not isinstance(cell_value[0].item(), float):
        raise TypeError(f'Every numpy array in the dataframe should store float values, but got: {cell_value.dtype}')

@validate_call
def _get_df_from_defects_file(path_to_defects_file: str) -> pd.DataFrame:
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
        logger.debug(f"""Got cell value without any full time-value pars""")
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
def _get_df_from_data_file(path_to_data_file: str) -> pd.DataFrame:
    """Read data file like "*_data.csv" and returns
       pandas dataframe with preprocessed data from it"""
    df = pd.read_csv(path_to_data_file, delimiter=';').astype(str)
    df = df.drop(['position'], axis=1)
    df = df.set_index('row')
    df = df.map(_split_cell_string_value_to_numpy_array_of_64_values)
    return df
