"""
Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import logging
import pandas as pd
import numpy as np

from pydantic import validate_call, PositiveInt, NonNegativeInt 
from functools import wraps
from typing import Callable, Optional, Generator, Iterable

# create logger
logger = logging.getLogger('main.'+__name__)


@validate_call(config=dict(arbitrary_types_allowed=True))
def crop_df(df: pd.DataFrame, 
            xy: tuple[NonNegativeInt , NonNegativeInt ] = (0,0),
            width: Optional[PositiveInt] = None, 
            height: Optional[PositiveInt] = None) -> pd.DataFrame:
    """
    The func to crop df

        Orig df
        +-------------------columns count-------------------+
        |                                                   |
        |                Cropped df:                        |
        |                (x,y)------width------+            |
        |                 |                    |            |
    rows count            |                  height         |
        |                 |                    |            |
        |                 ---------------------+            |
        |                                                   |
        +---------------------------------------------------+

    Parameters
    ----------
    df: pandas.DataFrame
        The df to be cropped
    xy: tuple[pydantic.PositiveInt, pydantic.PositiveInt], default = (0,0)
        Anchor point.
    width: pydantic.PositiveInt, optional
        The width of crop df. 
    height: pydantic.PositiveInt, optional
        The height of crop df. 
        
    Returns
    -------
    out : pandas.DataFrame
        The cropped df
        
    """
    if xy[1] > df.shape[0]:
        raise ValueError(f"The y should be less or equal to df's rows count. Got y={xy[1]}. df.shape={df.shape}")
    if xy[0] > df.shape[1]:
        raise ValueError(f"The x should be less or equal to df's rows count. Got x={xy[0]}. df.shape={df.shape}")
    if xy == (0,0) and height is None and width is None:
        return df

    in_shape = df.shape

    end_row = None if height is None else xy[1]+height
    end_col = None if width is None else xy[0]+width
    
    df = df.iloc[xy[1]:end_row, xy[0]:end_col]

    if end_row > in_shape[0]:
        logger.warning(f'The demanded crop height is bigger that the df. Got params: xy={xy}, width={width},' + 
                       f' height={height}. But df has shape={in_shape}. Result crop shape={df.shape}')
    if end_col > in_shape[1]:
        logger.warning(f'The demanded crop width is bigger that the df. Got params: xy={xy}, width={width},' + 
                       f' height={height}. But df has shape={in_shape}. Result crop shape={df.shape}')

    logger.debug(f"""
    Cropped with (xy={xy},width={width},height={height}) detectors data shape: {df.shape}""")
    
    return df

@validate_call(config=dict(arbitrary_types_allowed=True))
def roll_df(df: pd.DataFrame, shift: int = 0, axis: int = 0) -> pd.DataFrame:
    """
    Roll a dataframe like numpy.roll method.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be rolled.
    shift : int, optional
        The number of places by which elements are shifted, default is 0.
    axis : int, {0 or 1}, optional
        Axis along which elements are shifted, default is 0 .

    Returns
    -------
    out : pandas.DataFrame
        Output dataframe, with the same shape as df.

    Raises
    ------
    TypeError
        1. If the df is not pandas.Dataframe type.
        2. If the shift is not int type.
        3. If the axis is not int type.
    ValueError
        1. If the axis is not 0 or 1 (because dataframes
        have only 2 dimensions).
        
    """
    if not axis in (0,1):
        raise ValueError("The axis should be 0 or 1 because only row " + 
                         "wise of col wise rolling supported")
    
    if shift == 0:
        return df
    
    df_values = df.to_numpy()
    df_indexes = df.index.to_numpy()
    df_columns = df.columns.to_numpy()

    df_values = np.roll(df_values, shift, axis)
    
    if axis == 0:
        df_indexes = np.roll(df.index.to_numpy(), shift)
    if axis == 1:
        df_columns = np.roll(df.columns.to_numpy(), shift)

    return pd.DataFrame(data=df_values, index=df_indexes, 
                        columns=df_columns)


@validate_call(config=dict(arbitrary_types_allowed=True))
def match_df_for_crops_dividing(df: pd.DataFrame, 
                                crop_size: PositiveInt, 
                                crop_step: PositiveInt,
                                mode: str = 'extend') -> pd.DataFrame:
    """
    Match the df for exact crops dividing by a determined crop window with
    a determined cropping step via extending or cropping it.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be extended.
    crop_size : int
        The size of crop sliding window (has equal sides).
    crop_step : int
        The step for df cropping by sliding window).
    mode: str
        The mode to use when preparing the numpy array for
        crop dividing. Can be "extend" or "crop".

    Returns
    -------
    out : pandas.DataFrame
        Output match dataframe.
        
    """
    message = f"""
    The input df shape: {df.shape}"""
    if crop_size > df.shape[0] or crop_size > df.shape[1]:
        raise ValueError("Crop size should be bigger or equal " + 
                         "than the given df less axis")

    if crop_step > df.shape[0] or crop_step > df.shape[1]:
        raise ValueError("Crop size should be bigger or equal " + 
                         "than the given df less axis")

    if not mode in ('crop','extend'):
        raise ValueError(f'The mode param should be one of the folowing: crop; extend. Got {mode}')
        
    new_rows = crop_step - ((df.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((df.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        if mode == 'extend':
            df = pd.concat([df, df.iloc[-2:-new_rows-2:-1]], axis=0)
        elif mode == 'crop':
            df = df.iloc[:-1*(crop_step-new_rows)]
    if new_cols != crop_step:
        if mode == 'extend':
            df = pd.concat([df, df.iloc[:,-2:-new_cols-2:-1]], axis=1)
        elif mode == 'crop':
            df = df.iloc[:,:-1*(crop_step-new_cols)]

    logger.debug(f"""{message}
    The crop size: {crop_size}
    The crop step: {crop_step}
    The output df shape: {df.shape}""")
    
    return df


@validate_call(config=dict(arbitrary_types_allowed=True))
def extend_df_for_prediction(df: pd.DataFrame, crop_size: PositiveInt, only_horizontal: bool=False) -> pd.DataFrame:
    """
    Extend dataframe for increasing network model prediction or
    training quantity. 
    To left side of dfs is added reverse right crop_size-1 
    columns of them. 
    To right side of dfs is added reverse left crop_size-1 
    columns of them and some for exact dividing by crops with
    crop_size and crop_step. 
    To top side is added reverse top crop_size-1 rows. 
    To bottom side is added reverse bottom crop_size-1 rows
    and some for exact dividing by crops with crop_size and 
    crop_step.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be extended.
    crop_size : int
        The size of crop sliding window (has equal sides).
    crop_step : int
        The step for df cropping by sliding window).
    only_horizontal: bool
        If true extends the df only horizontally

    Returns
    -------
    out : pd.DataFrame
        Output extended dataframe.
        
    """
    message = f"""
    The input df shape: {df.shape}"""
    if crop_size > df.shape[0] or crop_size > df.shape[1]:
        raise ValueError("Crop size should be bigger or equal " + 
                         "than the given df less axis")
    
    df_values = df.to_numpy()
    df_indexes = df.index.to_numpy()
    df_columns = df.columns.to_numpy()

    if not only_horizontal:
        df_values = np.pad(df_values, ((crop_size-1, crop_size-1),(0, 0)), 'reflect')
        df_indexes = np.pad(df_indexes, crop_size-1, 'reflect')
        
    df_values = np.pad(df_values, ((0, 0),(crop_size-1, crop_size-1)), 'wrap')
    df_columns = np.pad(df_columns, crop_size-1, 'wrap')
    
    df = pd.DataFrame(data=df_values, index=df_indexes, columns=df_columns)

    logger.debug(f"""{message}
    The crop size: {crop_size}
    The output df shape: {df.shape}""")
    
    return df


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


def _check_df_cell_is_correct_numpy_array(cell_value):
    """Check that every pandas dataframe cell is a flat numpy array of floats"""
    if not isinstance(cell_value, np.ndarray):
        raise TypeError(f'Every cell of the dataframe should store numpy array, but got: {type(cell_value)}')
    if cell_value.ndim > 1:
        raise ValueError(f'Every numpy array in the dataframe should be flat, but got shape: {cell_value.shape}')
    if not isinstance(cell_value[0].item(), float):
        raise TypeError(f'Every numpy array in the dataframe should store float values, but got: {cell_value.dtype}')