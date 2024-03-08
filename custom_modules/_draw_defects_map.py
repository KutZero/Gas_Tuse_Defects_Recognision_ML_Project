import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pydantic import ValidationError, validate_call, PositiveInt

@validate_call(config=dict(arbitrary_types_allowed=True))
def draw_zeros_quantity_in_data_df(data_df: pd.DataFrame, **kwargs):
    data_df = data_df.map(lambda x: np.count_nonzero(x == 0))
    draw_defects_map(data_df, **kwargs)
    
@validate_call(config=dict(arbitrary_types_allowed=True))
def draw_defects_map(df: pd.DataFrame, /,
                        title: str = 'Развернутая карта дефектов',
                        xlabel: str = 'Номер датчика', 
                        ylabel: str = 'Номер измерения',
                        x_ticks_step: PositiveInt = 50,
                        y_ticks_step: PositiveInt = 20,
                        plt_style_context: str = 'default',
                        path_to_save: str = None):
    """
    Draw a defects map from the readed data.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe of size detectors num * rows
        with defect depth values in cells.
    title :str, optional
        The titile of the defects map.
    xlabel : str, optional
        The x label of the defects map.
    ylabel : str, optional
        The y label of the defects map.
    x_ticks_step : int, optional
        The x ticks step (approximate).
    y_ticks_step : int, optional
        The y ticks step (approximate).
    plt_style_context: str, optional
        The param for matplotlib.pyplot.style.context()
        
    Raises
    ------
    pydantic.ValidationError
        1. If the df is not pd.DataFrame type.
        2. If the title is not str type.
        3. If the xlabel is not str type.
        4. If the ylabel is not str type.
        5. If the x_ticks_step is not int type.
        6. If the y_ticks_step is not int type.
        7. If the plt_style_context is not str type.
    ValueError
        1. If the x_ticks_step is less than 1.
        2. If the y_ticks_step is less than 1.

    """
    
    with plt.style.context(plt_style_context):
        fig, ax = plt.subplots()
    
        fig.set_figwidth(18)
        fig.set_figheight(8)
        fig.patch.set_alpha(0.0)
    
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title(title, fontsize=25)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', labelsize = 20)
        
        # get columns and indexes list of int type
        cols = [re.search('[0-9]+', col)[0] for col in df.columns.to_numpy()]
        cols = np.array(cols).astype(int)
        indexes = df.index.to_numpy().astype(int)
        
        xlocs, xlabel_paddings, xlabels = _calc_labels_and_locs(cols, x_ticks_step)
        ylocs, ylabel_paddings, ylabels = _calc_labels_and_locs(indexes, y_ticks_step)
        
        map = ax.pcolormesh(df)

        cbar = fig.colorbar(map)
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_xticks(xlocs, xlabels)
        ax.set_yticks(ylocs, ylabels)

        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        
        xtext_labels = ax.get_xticklabels()
        ytext_labels = ax.get_yticklabels()

        for i in range(len(xtext_labels)):
            or_x, or_y = xtext_labels[i].get_position()
            xtext_labels[i].set(y=or_y+xlabel_paddings[i]*0.045)
            
        for i in range(len(ytext_labels)):
            or_x, or_y = ytext_labels[i].get_position()
            ytext_labels[i].set(x=or_x-ylabel_paddings[i]*0.05)

        ax.set_xticklabels(xtext_labels) 
        ax.set_yticklabels(ytext_labels) 
        
    plt.show()

def _add_index_fillers(arr: np.ndarray, step: int=1, i: int=0):
    """Add fillers between sorted (0 to max) numpy array of int or float"""
    if i > arr.shape[0]-2:
        return arr
    if arr[i+1] - arr[i] >= 1.5 * step:
        arr = np.insert(arr,i+1,arr[i]+step)
    return _add_index_fillers(arr, step, i+1)

@validate_call(config=dict(arbitrary_types_allowed=True))
def _cals_labels_paddings(locs: np.ndarray, step: int=1):
    """Calc labels paddings for avoid labels overlapping"""
    label_paddings = np.zeros(locs.shape)
    for i in range(locs.shape[0]-1):
        if locs[i+1] - locs[i] < step / 2:
            label_paddings[i] = 1
    return label_paddings

@validate_call(config=dict(arbitrary_types_allowed=True))
def _calc_labels_and_locs(arr: np.ndarray, step: int=1):
    """Calc locs and labels for matplotlib graph"""
    locs = np.sort(np.unique(np.concatenate(
                    [np.where((arr == min(arr)) | (arr == max(arr)))[0],
                           [0, arr.shape[0]-1]], axis=0)))

    locs = _add_index_fillers(locs, step)
    labels = arr[locs]
    # labels paddings for avoid labels overlapping
    label_paddings = _cals_labels_paddings(locs, step)
    return locs, label_paddings, labels