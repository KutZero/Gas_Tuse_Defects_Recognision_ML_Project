"""
Более подробнаую информацию можно получить так:

1) В Jupyter Notebook: "?[Имя модуля].[Имя функции]";
2) В общем виде: "print('[Имя модуля].[Имя функции].__doc__')";
3) В общем виде: "help([Имя модуля].[Имя функции])".
"""
import re
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from typing import Callable, Optional
from matplotlib import ticker
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon as shPolygon
from shapely.ops import unary_union
from typing_extensions import Annotated
from pydantic import validate_call, PositiveInt, AfterValidator, Field

PercentFloat = Annotated[float, Field(ge=0,le=1), AfterValidator(lambda x: float(x))]

# create logger
logger = logging.getLogger('main.'+__name__)


@validate_call(config=dict(arbitrary_types_allowed=True))
def draw_zeros_quantity_in_data_df(data_df: pd.DataFrame, *, 
                                   path_to_save: Optional[os.PathLike] = None, 
                                   dpi: int = 200, **kwargs):
    """
    Draw a zeros quantity in read x data.
    """
    data_df = data_df.map(lambda x: np.count_nonzero(x == 0))
    draw_defects_map(data_df, path_to_save, dpi, **kwargs)


@validate_call(config=dict(arbitrary_types_allowed=True))
def draw_defects_map(*args, path_to_save=None, dpi=200, **kwargs):
    """
    Draw a defects map from the readed data.

    Parameters
    ----------
    path_to_save: str or path-like or binary file-like
        The fname param for matplotlib.pyplot.savefig().
    dpi: int
        The dpi param for matplotlib.pyplot.savefig().
    """
    _build_defects_map(*args, **kwargs)
    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', dpi=dpi)
    plt.show()
    plt.close()

    
@validate_call(config=dict(arbitrary_types_allowed=True))
def draw_defects_map_with_reference_owerlap(df: pd.DataFrame, ref_df: pd.DataFrame, *, 
                                            pol_alfa: PercentFloat=1.0, 
                                            path_to_save: Optional[os.PathLike] = None, 
                                            dpi: int = 200, **kwargs):
    """
    Draw a defects map from the readed data with reference map owerlapption.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe of size detectors num * rows
        with defect depth values in cells.
    ref_df: pd.DataFrame
        The dataframe of size detectors num * rows
        with defect depth values in cells got by the experts.
    pol_alfa: PercentFloat
        
    path_to_save: str or path-like or binary file-like
        The fname param for matplotlib.pyplot.savefig().
    dpi: int
        The dpi param for matplotlib.pyplot.savefig().
    """
    fig, ax = _build_defects_map(df, **kwargs)
    
    # Get pixel position above the threshold
    Y, X = np.where(ref_df.to_numpy() > 0)
    positions = np.dstack((X, Y))[0]
    
    # Create a rectangle per position and merge them.
    rectangles = [shPolygon([xy, xy + [1, 0], xy + [1, 1], xy + [0, 1]]) for xy in positions]
    polygons = unary_union(rectangles)
    
    # Shapely will return either a Polygon or a MultiPolygon. 
    # Make sure the structure is the same in any case.
    if polygons.geom_type == "Polygon":
        polygons = [polygons]
    else:
        polygons = polygons.geoms
    
    # Add the matplotlib Polygon patches
    for polygon in polygons:
        ax.add_patch(mplPolygon(polygon.exterior.coords, fc=(0,1,0,1), alpha=pol_alfa)) #

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', dpi=dpi)
    plt.show()
    plt.close()


@validate_call(config=dict(arbitrary_types_allowed=True))
def draw_defects_map_with_rectangles_owerlap(df: pd.DataFrame, rectangles: list[Rectangle], *, 
                                             path_to_save: Optional[os.PathLike] = None, 
                                             dpi: int = 200, **kwargs):
    """
    Draw a defects map from the readed data with reference map owerlapption.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe of size detectors num * rows
        with defect depth values in cells.
    rectangles: list[Rectangle]
        The list of matplotlib.patches.Rectangle to be
        placed in the plot.
    path_to_save: str or path-like or binary file-like
        The fname param for matplotlib.pyplot.savefig().
    dpi: int
        The dpi param for matplotlib.pyplot.savefig().
    """
    fig, ax = _build_defects_map(df, **kwargs)
    
    # Add the matplotlib Polygon patches
    for rectangle in rectangles:
        ax.add_patch(rectangle)

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight', dpi=dpi)
    plt.show()
    plt.close()


@validate_call(config=dict(arbitrary_types_allowed=True))
def _build_defects_map(df: pd.DataFrame, 
                        /,
                        title: str = 'Развернутая карта дефектов',
                        xlabel: str = 'Номер датчика', 
                        ylabel: str = 'Номер измерения',
                        *,
                        x_ticks_step: PositiveInt = 50,
                        y_ticks_step: PositiveInt = 20,
                        plt_style_context = 'default',
                        pcolormesh_cmap = 'inferno',
                        add_color_bar: bool = False,
                        polygonize_data: int = 0,
                        bins: PositiveInt = 101):
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
    plt_style_context: optional
        The style param  for matplotlib.pyplot.style.context().
    pcolormesh_cmap: optional
        The cmap param for matplotlib.pyplot.pcolormesh.
    add_color_bar : bool, optional
        If true add color bar to the plot.
    polygonize_data : int, optional
        May be 0, 1, 2 or 4. 0 - do nothing. 1 - polygonize map.
        2 - polygonize map and add edges. 3 - polygonize map, add edges
        and label every polygon.
    bins: PositiveInt
        Used if approx_df is True. Defines possible
        range of values in the df. If 5 there are
        0. , 0.25, 0.5 , 0.75, 1. possible values. Polygon is a connected
        zone of values. Values connected to the polygon calculates on base
        of the step between values in bin. The step = 1 / (bins - 1). So if
        the bins=5, the step=0.25. If in that case the polygon named 0.25 it
        means than values within the one are bigger or equal than 0.25-step/2 
        and less than 0.25+step/2, or in other words if the polygon 
        named a 0.25 when the bins=5 the values within polygon >= 0.125 and < 0.375.
    """
    if not polygonize_data in [0,1,2,3]:
        raise ValueError('The polygonize_data must be on of [0,1,2,3]')
        
    with plt.style.context(plt_style_context):
        fig, ax = plt.subplots()
    
        fig.set_figwidth(32)
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

        if polygonize_data:
            df = _approx_df(df, bins)
        
        mapp = ax.pcolormesh(df, cmap=pcolormesh_cmap)

        if polygonize_data > 1:
           fig, ax = _polygonize(df, fig, ax, polygonize_data)
        
        if add_color_bar:
            cbar = fig.colorbar(mapp)
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
        ax.set_aspect('equal')
                
    return fig, ax


@validate_call(config=dict(arbitrary_types_allowed=True))
def _approx_df(df: pd.DataFrame, bins: PositiveInt) -> pd.DataFrame:
    """
    Approximate the df by given bins.
    If true approximated the df's cell values
    for range of max size equal to the bins param.
    The values approximates by the inner step value equals to 1 / (bins - 1). 
    Every valuSo if the bins=5, the step=0.25. If in that case the polygon named 0.25 it
    means than values within the one are bigger or equal than 0.25-step/2 
    and less than 0.25+step/2
    """
    step = 1/(bins-1)
    for value in np.arange(0,1+step/2,step):
        df = df.map(lambda x: value if (x >= value-step/2) and (x < value+step/2) else x)
    return df

    
@validate_call(config=dict(arbitrary_types_allowed=True))
def _polygonize(df: pd.DataFrame, fig, ax, polygonize_data):
    """
    Draw polygons over map by every unique value
    """
    for value in np.unique(df.to_numpy()):
        # Get pixel position above the threshold
        arr = df.to_numpy()
        Y, X = np.where(arr==value) # (arr >= value-step/2) & (arr < value+step/2)
        positions = np.dstack((X, Y))[0]
        
        # Create a rectangle per position and merge them.
        rectangles = [shPolygon([xy, xy + [1, 0], xy + [1, 1], xy + [0, 1]]) for xy in positions]
        polygons = unary_union(rectangles)
        
        # Shapely will return either a Polygon or a MultiPolygon. 
        # Make sure the structure is the same in any case.
        if polygons.geom_type == "Polygon":
            polygons = [polygons]
        else:
            polygons = polygons.geoms

        # Add the matplotlib Polygon patches
        if polygonize_data == 2:
            for polygon in polygons:
                ax.add_patch(mplPolygon(polygon.exterior.coords, fc=(0,0,0,0), ec='white'))
        elif polygonize_data == 3:
            for polygon in polygons:
                min_side = min([polygon.bounds[3] - polygon.bounds[1], 
                                polygon.bounds[2] - polygon.bounds[0]])
                text_coords = polygon.point_on_surface()
                ax.annotate(text=f'{value:.4f}', 
                            xy=[text_coords.x, text_coords.y], 
                            color=('white' if value <= 0.6 else 'black'), 
                            fontsize=(min_side if min_side <= 40 else 40), 
                            ha='center', va='center')
                ax.add_patch(mplPolygon(polygon.exterior.coords, fc=(0,0,0,0), ec='white'))
    return fig, ax
    

@validate_call(config=dict(arbitrary_types_allowed=True))
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