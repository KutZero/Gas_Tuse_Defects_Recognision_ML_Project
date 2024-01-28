import pandas as pd
import numpy as np

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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The df should be pandas.Dataframe")
    if not isinstance(shift, int):
        raise TypeError("The shift should be int")
    if not isinstance(axis, int):
        raise TypeError("The axis should be int")

    if shift == 0:
        return df

    if not axis in (0,1):
        raise ValueError("The axis should be 0 or 1")
    
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
    def wrapper(df, **kwargs):
        """The wrapper for funcitons that checks input correctness
        and print some technical information"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The df should be pandas.Dataframe")
        for key, value in kwargs.items():
            if not isinstance(value, int):
                raise TypeError(f"The {key} should be int")
            if value < 1:
                raise ValueError(f"The {key} must be positive")
            if value > df.shape[0] or value > df.shape[1]:
                raise ValueError(f"""The {key} should be less than 
                or equal to rows and cols of the input dataframe""")

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
def extend_df_for_crops_dividing(df: pd.DataFrame, *, crop_size: int, crop_step: int) -> pd.DataFrame:
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
    new_rows = crop_step - ((df.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((df.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        df = pd.concat([df, df.iloc[-2:-new_rows-2:-1]], axis=0)
    if new_cols != crop_step:
        df = pd.concat([df, df.iloc[:,-2:-new_cols-2:-1]], axis=1)
        
    return df

@_extend_df_decorator
def extend_df_for_prediction(df: pd.DataFrame, *, crop_size: int) -> pd.DataFrame:
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
    df_values = df.to_numpy()
    df_indexes = df.index.to_numpy()
    df_columns = df.columns.to_numpy()

    df_values = np.pad(df_values, ((crop_size-1, crop_size-1),(0, 0)), 'reflect')
    df_values = np.pad(df_values, ((0, 0),(crop_size-1, crop_size-1)), 'wrap')
    
    df_indexes = np.pad(df_indexes, crop_size-1, 'reflect')
    df_columns = np.pad(df_columns, crop_size-1, 'wrap')
    
    df = pd.DataFrame(data=df_values, index=df_indexes, columns=df_columns)
    
    return df