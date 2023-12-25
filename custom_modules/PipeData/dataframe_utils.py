import pandas as pd
import numpy as np

__all__ = ["roll_df", "extend_df_for_crops_dividing", "extend_df_for_prediction"]

def roll_df(df: pd.DataFrame, shift: int = 0, axis: int = 0):
    """
    Roll a dataframe like numpy.roll method

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be rolled
    shift : int, optional
        The number of places by which elements are shifted, default is 0
    axis : int, {0 or 1}, optional
        Axis along which elements are shifted, default is 0 

    Returns
    -------
    out : pd.DataFrame
        Output dataframe, with the same shape as df.
        
    """
    df_values = df.to_numpy()
    df_indexes = df.index.to_numpy()
    df_columns = df.columns.to_numpy()

    df_values = np.roll(df_values, shift, axis)
    if axis == 0:
        df_indexes = np.roll(df.index.to_numpy(), shift)
    if axis == 1:
        df_columns = np.roll(df.columns.to_numpy(), shift)

    out = pd.DataFrame(data=df_values, index=df_indexes, 
                        columns=df_columns)
    return out

def extend_df_for_crops_dividing(df: pd.DataFrame, crop_size: int, crop_step: int):
    """
    Extend the df for exact crops dividing where crop have <crop_size> 
    and <crop_step>

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be extended
    crop_size : int
        The size of crop sliding window (has equal sides)
    crop_step : int
        The step for df cropping by sliding window)

    Returns
    -------
    out : pd.DataFrame
        Output extended dataframe.
        
    """
    new_rows = crop_step - ((df.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((df.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        df = pd.concat([df, df.iloc[-2:-new_rows-2:-1]], axis=0)
    if new_cols != crop_step:
        df = pd.concat([df, df.iloc[:,-2:-new_cols-2:-1]], axis=1)
        
    return df

def extend_df_for_prediction(df, crop_size: int, crop_step: int):
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
    df : pd.DataFrame
        The dataframe to be extended
    crop_size : int
        The size of crop sliding window (has equal sides)
    crop_step : int
        The step for df cropping by sliding window)

    Returns
    -------
    out : pd.DataFrame
        Output extended dataframe.
        
    """
    extend_dims = crop_size - 1
    
    df = pd.concat([df.iloc[:,-1*extend_dims:], df, df.iloc[:,:extend_dims]],axis=1)
    df = pd.concat([df.iloc[extend_dims:0:-1,:], df, df.iloc[-2:-extend_dims-2:-1,:]],axis=0)
    
    return df