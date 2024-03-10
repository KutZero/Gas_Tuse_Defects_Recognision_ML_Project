""" Модуль для всяческой обработки данных перед обучением модели.

Функции:

Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from typing import Union

from ._draw_defects_map import draw_defects_map, draw_defects_map_with_reference_owerlap, draw_zeros_quantity_in_data_df
from ._dataframe_utils import roll_df, extend_df_for_crops_dividing, extend_df_for_prediction 

from typing_extensions import Annotated
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

PositiveInt = Annotated[int, Field(gt=0), AfterValidator(lambda x: int(x))]
PercentFloat = Annotated[float, Field(ge=0,le=1), AfterValidator(lambda x: float(x))]


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
    
    pred_arr = normalize_data(pred_df.to_numpy())
    ref_arr = normalize_data(ref_df.to_numpy())

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
    print('||||||||||||||||||')
    print('Original data reading')
    
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

    print(f'Read detectors data shape: {data_df.shape}')
    print(f'Read defect data shape: {defects_df.shape}')
    print('||||||||||||||||||\n')
    return data_df, defects_df

def _check_df_cell_is_correct_numpy_array(cell_value):
    """Check that every pandas dataframe cell is a flat numpy array of floats"""
    if not isinstance(cell_value, np.ndarray):
        raise TypeError(f'Every cell of the dataframe should store numpy array, but got: {type(cell_value)}')
    if cell_value.ndim > 1:
        raise ValueError(f'Every numpy array in the dataframe should be flat, but got shape: {cell_value.shape}')
    if not isinstance(cell_value[0].item(), float):
        raise TypeError(f'Every numpy array in the dataframe should store float values, but got: {cell_value.dtype}')


@validate_call(config=dict(arbitrary_types_allowed=True))
def _df_to_image_like_numpy(df: pd.DataFrame) -> np.ndarray:
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
def reshape_x_df_to_image_like_numpy(df: pd.DataFrame,
                                     crop_size: PositiveInt, 
                                     crop_step: PositiveInt = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice the df with data got from detectors with square sliging window of size 
    crop_size and step crop_step and return 2 numpy arrays. The first one - array 
    of slised crops with 32 data channels (last dimension size) where every one - 
    value of time got from the df. The second has equal shape but store amplitude 
    values got from the df.
    Input df shape: (rows, cols, channels(time+amp)) 
        -> output shape: (batch, rows, cols, channels(time)) 
        and (batch, rows, cols, channels(amp))

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe with data got from detectors
    crop_size : int
        The dimension size of the square sliding window
    crop_step : int
        The step of the square sliding window

    Returns
    -------
    x_time : numpy.ndarray
        The numpy array of crops with times values
    x_amp : numpy.ndarray
        The numpy array of crops with amplitudes values
    
    """
    print('||||||||||||||||||')
    print('X df reshaping to 4D')
    print('Original df size: ', df.shape)
    print('Crop windows height/width: ', crop_size)
    print('Crop windows step across rows and cols: ', crop_step)

    if crop_step == 0:
        crop_step = crop_size
    
    temp = np.concatenate([np.stack(
        [_df_to_image_like_numpy(
            df.iloc[i:i+crop_size,j:j+crop_size])
             for i in range(0,df.shape[0] - crop_size + 1, crop_step)]
                , axis=0) for j in range(0,df.shape[1] - crop_size + 1, crop_step)]
                    , axis=0)

    # поделим x выборку на значения времен и амплитуд
    x_time = temp[:,:,:,:32]
    x_amp = temp[:,:,:,32:]

    print('New x_time shape: ', x_time.shape)
    print('New x_amp shape: ', x_amp.shape)
    print('||||||||||||||||||\n')

    return (x_time, x_amp)

@validate_call(config=dict(arbitrary_types_allowed=True))
def reshape_y_df_to_image_like_numpy(df: pd.DataFrame,
                                     crop_size: PositiveInt, 
                                     crop_step: PositiveInt = 0) -> np.ndarray:
    """
    Slice the df with data got from specialists about defects depths and locations 
    with square sliging window of size crop_size and step crop_step and return numpy 
    array of slised crops with 1 data channel - defect depth for each cell of the crop.
    Input df shape: (rows, cols, channel(defect depth)) 
        -> output shape: (batch, rows, cols, channel(defect depth)) 


    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe with data got from specialists about defects depths and locations
    crop_size : int
        The dimension size of the square sliding window
    crop_step : int
        The step of the square sliding window

    Returns
    -------
    res : numpy.ndarray
        The numpy array of crops with defect depths values
    
    """
    print('||||||||||||||||||')
    print('Y df reshaping to 3D')
    print('Original df size: ', df.shape)
    print('Crop windows height/width: ', crop_size)
    print('Crop windows step across rows and cols: ', crop_step)
    
    if crop_step == 0:
        crop_step = crop_size

    res = np.concatenate([np.stack(
        [df.iloc[i:i+crop_size,j:j+crop_size].to_numpy().astype('float32')
             for i in range(0,df.shape[0] - crop_size + 1, crop_step)]
                , axis=0) for j in range(0,df.shape[1] - crop_size + 1, crop_step)]
                    , axis=0)


    res = np.expand_dims(res,axis=3)

    print('New numpy shape: ', res.shape)
    print('||||||||||||||||||\n')

    return res

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
    print('||||||||||||||||||')
    print('Data normalizing')

    print(f'arr_max before normalization: {arr.max()}')
    print(f'arr_min before normalization: {arr.min()}')

    arr = (arr - arr.min()) / (arr.max() - arr.min())

    print(f'arr_max after normalization: {arr.max()}')
    print(f'arr_min after normalization: {arr.min()}')
    print('||||||||||||||||||\n')
    
    return arr

@validate_call(config=dict(arbitrary_types_allowed=True))
def standartize_data(arr: np.ndarray) -> np.ndarray:
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
    print('||||||||||||||||||')
    print('Data standartizing')

    print(f'arr_max before standartize: {arr.max()}')
    print(f'arr_min before standartize: {arr.min()}')

    arr = np.divide(arr, arr.max(), out=np.zeros_like(arr), where=arr.max()!=0)

    print(f'arr_max after standartize: {arr.max()}')
    print(f'arr_min after standartize: {arr.min()}')
    print('||||||||||||||||||\n')
    
    return arr

@validate_call(config=dict(arbitrary_types_allowed=True))
def split_def_and_non_def_data(x_time: np.ndarray, 
                               x_amp: np.ndarray, 
                               y_mask: np.ndarray, 
                               crop_size: PositiveInt) -> tuple[tuple[np.ndarray, np.ndarray],
                                                        tuple[np.ndarray, np.ndarray],
                                                        tuple[np.ndarray, np.ndarray]]:
    """
    Got 3 numpy arrays. The x_time and x_amp store crops data got from detectors.
    The y_mask stores crops data about defects locations and depths got from specialists.
    By this data calculate crops that describes defect zones and not and return it as
    tuple. Where x_time are splitted for x_time_def, x_time_non_def, x_amp for 
    x_amp_def, x_amp_non_def, y_mask for y_mask_def, y_mask_non_def.

    Parameters
    ----------
    x_time : numpy.ndarray
        The numpy array with time values
    x_amp : numpy.ndarray
        The numpy array with amplitude values
    y_mask : numpy.ndarray
        The numpy array with defect depths values

    Returns
    -------
    x_time_def : numpy.ndarray
        The numpy array with crops data with time values refer to defect zone
    x_time_non_def : numpy.ndarray
        The numpy array with crops data with time values refer to non defect zone
    x_amp_def : numpy.ndarray
        The numpy array with crops data with amplitude values refer to defect zone
    x_amp_non_def : numpy.ndarray
        The numpy array with crops data with amplitude values refer to non defect zone
    y_mask_def : numpy.ndarray
        The numpy array with crops data with defect depths values refer to defect zone
    y_mask_non_def : numpy.ndarray
        The numpy array with crops data with defect depths values refer to non defect zone
    
    """
    print('||||||||||||||||||')
    print('Defect and non defect data splitting')

    print('Orig x_time shape: ', x_time.shape)
    print('Orig x_amp shape: ', x_amp.shape)
    print('Orig y_mask shape: ', y_mask.shape)

    # удалим кропы не содержищие дефекты
    defects_nums = _calculate_crops_with_defects_positions(y_mask, crop_size)

    x_time_def = x_time[defects_nums]
    x_amp_def = x_amp[defects_nums]
    y_mask_def = y_mask[defects_nums]

    x_time_non_def = x_time[~defects_nums]
    x_amp_non_def = x_amp[~defects_nums]
    y_mask_non_def = y_mask[~defects_nums]


    print('x_time_def shape: ', x_time_def.shape)
    print('x_time_non_def shape: ', x_time_non_def.shape)
    print()

    print('x_amp_def shape: ', x_amp_def.shape)
    print('x_amp_non_def shape: ', x_amp_non_def.shape)
    print()

    print('y_mask_def shape: ', y_mask_def.shape)
    print('y_mask_non_def shape: ', y_mask_non_def.shape)

    print('||||||||||||||||||\n')

    return ((x_time_def, x_time_non_def),
            (x_amp_def, x_amp_non_def),
            (y_mask_def, y_mask_non_def))

    
@validate_call(config=dict(arbitrary_types_allowed=True))
def create_binary_arr_from_mask_arr(y_mask: np.ndarray) -> np.ndarray:
    """
    Create binary array from the y_mask array of shape 
    (batch, rows, cols, channel) that store crops data
    got from specialist about defect depths and locations.

    Parameters
    ----------
    y_mask : numpy.ndarray
        The numpy array with defect depths values of shape
        (batch, rows, cols, channel)

    Returns
    -------
    y_binary : numpy.ndarray
        The flat numpy array with binary values for each crop 
    
    """
    if not isinstance(y_mask, np.ndarray):
        raise TypeError('Mask array should be numpy array')
    if y_mask.ndim != 4:
        raise ValueError('Mask arr should have (batch, rows, cols, channel) shape')
    if not np.issubdtype(y_mask.dtype, np.number):
        raise ValueError('Mask arr shoul store numeric values')
    
    print('||||||||||||||||||')
    print('Y binary arr from Y mask arr creation')
    print('Y mask arr shape: ', y_mask.shape)
    # Найдем на каких картинках есть дефекты
    y_binary = list()
    for i in range(y_mask.shape[0]):
        if np.sum(y_mask[i] > 0) >= 1:
            y_binary.append(True)
        else:
            y_binary.append(False)

    y_binary = np.array(y_binary, dtype='bool')

    print('Y binary arr shape: ', y_binary.shape)
    print('||||||||||||||||||\n')

    return y_binary

    
@validate_call(config=dict(arbitrary_types_allowed=True))
def create_depth_arr_from_mask_arr(y_mask: np.ndarray) -> np.ndarray:
    """
    Create max depth array from the y_mask array of shape 
    (batch, rows, cols, channel) that store crops data
    got from specialist about defect depths and locations.

    Parameters
    ----------
    y_mask : numpy.ndarray
        The numpy array with defect depths values of shape
        (batch, rows, cols, channel)

    Returns
    -------
    y_binary : numpy.ndarray
        The flat numpy array with max depth values for each crop 
    
    """
    if not isinstance(y_mask, np.ndarray):
        raise TypeError('Mask array should be numpy array')
    if y_mask.ndim != 4:
        raise ValueError('Mask arr should have (batch, rows, cols, channel) shape')
    if not np.issubdtype(y_mask.dtype, np.number):
        raise ValueError('Mask arr shoul store numeric values')
    
    print('||||||||||||||||||')
    print('Y depth arr from Y mask arr creation')
    print('Y mask arr shape: ', y_mask.shape)
    # Найдем на каких картинках есть дефекты
    y_depth = list()
    for i in range(y_mask.shape[0]):
        y_depth.append(np.max(y_mask[i]))

    y_depth = np.array(y_depth)

    print('Y depth arr shape: ', y_depth.shape)
    print('||||||||||||||||||\n')

    return y_depth
    
@validate_call(config=dict(arbitrary_types_allowed=True))
def augment_data(arr: np.ndarray) -> np.ndarray:
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
    print('||||||||||||||||||')
    print('Data augmentation')

    print('Orig arr shape: ', arr.shape)

    arr = np.concatenate([arr,
                            np.rot90(arr,1,[1,2]),
                            np.rot90(arr,2,[1,2]),
                            np.rot90(arr,3,[1,2])],axis=0)


    print('||||||||||||\nAfter 4 steps of 90 degree rotate')
    print('arr shape: ', arr.shape)

    arr = np.concatenate([arr,np.flip(arr,2)],axis=0)

    print('||||||||||||\nAfter horizontal full mirroring')
    print('arr shape: ', arr.shape)

    arr = np.concatenate([arr,np.flip(arr,1)],axis=0)

    print('||||||||||||\nAfter vertical full mirroring')
    print('arr shape: ', arr.shape)

    '''arr = np.concatenate([arr,np.roll(arr,int(arr.shape[1]/2),axis=1)],axis=0)

    print('||||||||||||\nAfter vertical half shifting')
    print('arr shape: ', arr.shape)

    arr = np.concatenate([arr,np.roll(arr,int(arr.shape[2]/2),axis=2)],axis=0)

    print('||||||||||||\nAfter horizontal half shifting')
    print('X_time_arr shape: ', arr.shape)'''

    print('||||||||||||||||||\n')
    return arr

    
@validate_call(config=dict(arbitrary_types_allowed=True))
def split_data_to_train_val_datasets(arr: np.ndarray, 
                                     val_percent: PercentFloat) -> tuple[np.ndarray, np.ndarray]:
    """
    Split data of the arr of numpy arrays where each array store some part of
    defects filtered by some parameter. For example for numpy arrays of time
    values where the first stores only crops desctiptions refers to defect zones,
    and the second stores only crops desctiptions refers to non defect zones 
    the method returns 2 arrays where the first - stores the 1-val_percent of
    crops descriptions of each passed array and the second stores val_percent
    of each passed array.

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array of numpy arrays

    Returns
    -------
    arr_train : numpy.ndarray
        The numpy with train dataset part of data
    arr_val : numpy.ndarray
        he numpy with validation dataset part of data
    
    """
    print('||||||||||||||||||')
    print('Data spliting to test, val and train datasets')

    for item in arr:
        print('Orig item shape: ', item.shape)
    print('')

    arr_train = np.concatenate([item[int(item.shape[0] * val_percent):] for item in arr], axis=0)
    arr_val = np.concatenate([item[:int(item.shape[0] * val_percent)] for item in arr], axis=0)

    print('Result arr_train shape: ', arr_train.shape)
    print('Result arr_val shape: ', arr_val.shape)

    print('||||||||||||||||||\n')

    return arr_train, arr_val


# вернет бинарную 1D маску, где 1 - для кропов с дефектами
# 0 - для кропов без дефектов
# delete crop_size from arguments
@validate_call(config=dict(arbitrary_types_allowed=True))
def _calculate_crops_with_defects_positions(y_arr: np.ndarray, 
                                            crop_size: PositiveInt) -> np.ndarray:
    """
    Got the y_mask array that store crops data got from specialist 
    about defect depths and locations and return 1 dimension binary
    mask where True value descripres crop that refer to defect zone,
    False - not.

    Parameters
    ----------
    y_arr : numpy.ndarray
        The numpy array with defect depths values

    Returns
    -------
    defects_nums : numpy.ndarray
        The 1 dimension binary mask array
    
    """
    print('||||||||||||||||||')
    print('Defects nums calculating')
    # Найдем на каких картинках есть дефекты
    defects_nums = list()
    for i in range(y_arr.shape[0]):
        if np.sum(y_arr[i] > 0) >= 1:
            defects_nums.append(True)
        else:
            defects_nums.append(False)

    defects_nums = np.array(defects_nums, dtype='bool')

    print(f'For {y_arr.shape[0]} crops of size: {crop_size}',
          f'there are {np.sum(defects_nums)} defect crops', sep='\n')
    print('||||||||||||||||||\n')

    return defects_nums

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
    for item in num_pars:
        if not item[0] or not item[2]:
            print(f"""Got input df cell str with uncompleted time-amplitude value pars.
            The uncomplited pars replaced with zeros pars.
            Input str: {df_cell_value}""")
            break
    num_pars = np.array([[item[0], item[2]] if item[0] and item[2] 
                             else [0, 0] for item in num_pars]).astype(float)
    
    if num_pars.size == 0:
        #print(f'Got wrong input df cell str value:{df_cell_value}')
        return np.zeros((64))

    if num_pars.shape[0] > 32:
        raise ValueError(f'Too much time-amplitude values pars in a cell. Got: {num_pars.shape[0]}. Max: 32')
    
    time_vals = num_pars[:,0]
    amp_vals = num_pars[:,1]
    
    time_vals = np.pad(time_vals, (0, abs(time_vals.size-32)), constant_values=(0))
    amp_vals = np.pad(amp_vals, (0, abs(amp_vals.size-32)), constant_values=(0))
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
