"""
Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import pandas as pd
import numpy as np

from typing_extensions import Annotated
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

PositiveInt = Annotated[int, Field(gt=0), AfterValidator(lambda x: int(x))]


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
        raise ValueError("""The axis should be 0 or 1 because only row 
        wise of col wise rolling supported""")
    
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

def _extend_df_decorator(func):
    """The decorator for pandas dataframe extending functions"""
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def wrapper(df: pd.DataFrame, **kwargs):
        """The wrapper for funcitons that checks input correctness
        and print some technical information"""
        print('|'*20)
        print(func.__name__)
        print(f'Input df.shape: {df.shape}')
        for key, value in kwargs.items():
            print(f'{key}: {value}')
        res_df = func(df, **kwargs)
        print(f'Output df.shape: {res_df.shape}')
        print('|'*20)
        return res_df
    return wrapper

@_extend_df_decorator
@validate_call(config=dict(arbitrary_types_allowed=True))
def extend_df_for_crops_dividing(df: pd.DataFrame, *, crop_size: PositiveInt, crop_step: PositiveInt) -> pd.DataFrame:
    """
    Extend the df for exact crops dividing by a determined crop window with
    a determined cropping step.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be extended.
    crop_size : int
        The size of crop sliding window (has equal sides).
    crop_step : int
        The step for df cropping by sliding window).

    Returns
    -------
    out : pandas.DataFrame
        Output extended dataframe.
        
    """
    if min([crop_size, crop_step]) > min(df.shape):
        raise ValueError("""Crop size and crop step should be bigger or equal
        than the given df less axis""")
        
    new_rows = crop_step - ((df.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((df.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        df = pd.concat([df, df.iloc[-2:-new_rows-2:-1]], axis=0)
    if new_cols != crop_step:
        df = pd.concat([df, df.iloc[:,-2:-new_cols-2:-1]], axis=1)
        
    return df

@_extend_df_decorator
@validate_call(config=dict(arbitrary_types_allowed=True))
def extend_df_for_prediction(df: pd.DataFrame, *, crop_size: PositiveInt) -> pd.DataFrame:
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

    Returns
    -------
    out : pd.DataFrame
        Output extended dataframe.
        
    """
    if crop_size > min(df.shape):
        raise ValueError("""Crop size should be bigger or equal
        than the given df less axis""")
    
    df_values = df.to_numpy()
    df_indexes = df.index.to_numpy()
    df_columns = df.columns.to_numpy()

    df_values = np.pad(df_values, ((crop_size-1, crop_size-1),(0, 0)), 'reflect')
    df_values = np.pad(df_values, ((0, 0),(crop_size-1, crop_size-1)), 'wrap')
    
    df_indexes = np.pad(df_indexes, crop_size-1, 'reflect')
    df_columns = np.pad(df_columns, crop_size-1, 'wrap')
    
    df = pd.DataFrame(data=df_values, index=df_indexes, columns=df_columns)
    
    return df