""" Модуль для всяческой обработки данных перед обучением модели.

Функции:

Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import re
import os
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from typing_extensions import Annotated
from pydantic import ValidationError, validate_call, PositiveInt, AfterValidator, Field

PositiveInt = Annotated[int, Field(gt=0), AfterValidator(lambda x: int(x))]
PercentFloat = Annotated[float, Field(ge=0,le=1), AfterValidator(lambda x: float(x))]

# create logger
logger = logging.getLogger('main.'+__name__)


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
    if np.all(arr==0):
        logger.debug('\nThe input array consists only from zeros so it was not changed')
        return arr
            
    max_val = arr.max()
    min_val = arr.min()
    
    if min_val < 0:
        arr -= min_val
    
    arr = arr / np.max(arr)

    logger.debug(f"""
    The arr max before normalization: {max_val}
    The arr min before normalization: {min_val}
    The arr max after normalization: {arr.max()}
    The arr min after normalization: {arr.min()}""")

    return arr

@validate_call(config=dict(arbitrary_types_allowed=True))
def standardize_data(arr: np.ndarray) -> np.ndarray:
    """
    Standartize the arr by dividing it element-wise by
    max in absolute value of the array.

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array with some float values

    Returns
    -------
    out : numpy.ndarray
        The numpy array with standartized some float values
    
    """
    if np.all(arr==0):
        logger.debug('\nThe input array consists only from zeros so it was not changed')
        return arr
    
    max_val = arr.max()
    min_val = arr.min()

    arr = arr / np.max(np.abs(arr))

    logger.debug(f"""
    The arr max before standardization: {max_val}
    The arr min before standardization: {min_val}
    The arr max after standardization: {arr.max()}
    The arr min after standardization: {arr.min()}""")
    
    return arr