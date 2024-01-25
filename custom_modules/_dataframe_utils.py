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

def extend_df_for_crops_dividing(df: pd.DataFrame, crop_size: int, crop_step: int) -> pd.DataFrame:
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
        
    Raises
    ------
    TypeError
        1. If the df is not pandas.Dataframe type.
        2. If the crop_size is not int type.
        3. If the crop_step is not int type.
    ValueError
        1. If the crop_size is less than 1.
        2. If the crop_step is less than 1.
        
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The df should be pandas.Dataframe")
    if not isinstance(crop_size, int):
        raise TypeError("The crop_size should be int")
    if not isinstance(crop_step, int):
        raise TypeError("The crop_step should be int")

    if crop_size < 1:
        raise ValueError("The crop_size should be grater than or equal to 1")
    if crop_step < 1:
        raise ValueError("The crop_step should be grater than or equal to 1")

    print('||||||||||||||||||')
    print('extend_df_for_crops_dividing')
    print('input df shape: ', df.shape, end=' -> ')
    new_rows = crop_step - ((df.shape[0] - crop_size) % crop_step)
    new_cols = crop_step - ((df.shape[1] - crop_size) % crop_step)

    if new_rows != crop_step:
        df = pd.concat([df, df.iloc[-2:-new_rows-2:-1]], axis=0)
    if new_cols != crop_step:
        df = pd.concat([df, df.iloc[:,-2:-new_cols-2:-1]], axis=1)
    
    print('output shape: ', df.shape)
    print('||||||||||||||||||\n')
    return df

def extend_df_for_prediction(df, crop_size: int, crop_step: int) -> pd.DataFrame:
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
        
    Raises
    ------
    TypeError
        1. If the df is not pandas.Dataframe type.
        2. If the crop_size is not int type.
        3. If the crop_step is not int type.
    ValueError
        1. If the crop_size is less than 1.
        2. If the crop_step is less than 1.
        
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The df should be pandas.Dataframe")
    if not isinstance(crop_size, int):
        raise TypeError("The crop_size should be int")
    if not isinstance(crop_step, int):
        raise TypeError("The crop_step should be int")

    if crop_size < 1:
        raise ValueError("The crop_size should be grater than or equal to 1")
    if crop_step < 1:
        raise ValueError("The crop_step should be grater than or equal to 1")

    print('||||||||||||||||||')
    print('extend_df_for_prediction')
    print('input df shape: ', df.shape, end=' -> ')
    extend_dims = crop_size - 1
    
    df = pd.concat([df.iloc[:,-1*extend_dims:], df, df.iloc[:,:extend_dims]],axis=1)
    df = pd.concat([df.iloc[extend_dims:0:-1,:], df, df.iloc[-2:-extend_dims-2:-1,:]],axis=0)
    print('output shape: ', df.shape)
    print('||||||||||||||||||\n')
    return df