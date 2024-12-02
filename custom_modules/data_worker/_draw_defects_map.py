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
from functools import wraps

PercentFloat = Annotated[float, Field(ge=0,le=1), AfterValidator(lambda x: float(x))]

# create logger
logger = logging.getLogger('main.'+__name__)


def is_df_valid_decorator(func):
    @wraps(func)
    def wrapper(df,*args,**kvargs):
        if not df.dtypes.map(lambda x: x in ['int','float']).values.all():
            raise ValueError('The df should store only int and float values in every cell')
        if not list(df.index.names) == ['File', 'ScanNum']:
            raise ValueError(f'The df should have index with levels: "File", "ScanNum", but got: {df.index.names=}')
        if not list(df.columns.names) == ['DetectorNum']:
            raise ValueError(f'The df should have columns with levels: "DetectorNum", but got: {df.columns.names=}')
        if df.columns.values.dtype.name != 'int64':
            raise ValueError(f'The df should have columns with levels: "DetectorNum" and dtype "int64", but got: {df.columns.values.dtype=}')
        if df.index.get_level_values('ScanNum').dtype.name != 'int64':
            raise ValueError(f"The df's index level 'ScanNum' should have dtype 'int64', but got: {df.index.get_level_values('ScanNum').dtype=}")
        return func(df,*args,**kvargs)
    return wrapper

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
@is_df_valid_decorator
def _build_defects_map(df: pd.DataFrame, 
                        /,
                        title: str = 'auto',
                        xlabel: str = 'Номер датчика', 
                        ylabel: str = 'Номер сканирования',
                        *,
                        x_ticks_step: PositiveInt = 20,
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
        The titile of the defects map. If 'auto' the
        title will be generated from dataframe
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

        width = df.shape[1]//10
        
        fig.set_figwidth(width) # 32
        fig.set_figheight(df.shape[0]//10)
        fig.patch.set_alpha(0.0)

        width = 30
        
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_title(f"Развернутая карта девектов для файлов: {set(df.index.get_level_values('File').values)}" 
                     if title == 'auto' else title, fontsize=width)
        ax.set_xlabel(xlabel, fontsize=width)
        ax.set_ylabel(ylabel, fontsize=width, labelpad=90)
        ax.tick_params(axis='both', labelsize = width)

        if polygonize_data:
            df = _approx_df(df, bins)

        orig_x_ticks = df.columns.values
        orig_y_ticks = df.index.get_level_values('ScanNum').values

        # create base ticks (detectors and scans numbers)
        ax.set_xticks(*_get_ticks(orig_x_ticks, x_ticks_step))
        ax.set_yticks(*_get_ticks(orig_y_ticks, y_ticks_step))

        # create second level ticks (edges of mirror of padding results)
        sec_x = ax.secondary_xaxis(location=1)
        sec_x.set_xticks(*_get_extreme_ticks(orig_x_ticks))
        sec_x.tick_params('x', length=60, width=4, color='blue')

        sec_y = ax.secondary_yaxis(location=0)
        sec_y.set_yticks(*_get_extreme_ticks(orig_y_ticks))
        sec_y.tick_params('y', length=60, width=4, color='blue')

        # create third level ticks for every run
        # create edges lines
        third_y_locs, third_y_labels = _get_class_edges(df.index)
        third_y = ax.secondary_yaxis(location=0)
        third_y.set_yticks(third_y_locs, [])
        third_y.tick_params('y', length=120, width=4)
            
        third_y = ax.secondary_yaxis(location=0)
        third_y.set_yticks(third_y_locs[:-1], map(lambda x: f'<-{x}-> ', third_y_labels[:-1]))
        third_y.tick_params('y', length=0, width=0, labelsize=width, labelrotation=90, pad=90)
        
        if polygonize_data > 1:
           fig, ax = _polygonize(df, fig, ax, polygonize_data)
        
        if add_color_bar:
            mapp = ax.pcolormesh(df, cmap=pcolormesh_cmap)
            cbar = fig.colorbar(mapp, shrink=0.78, pad=0.01, ax=ax)
            cbar.ax.tick_params(labelsize=width)

        ax.set_aspect('equal')
        plt.tight_layout()
        
    return fig, ax


@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_class_edges(index: pd.MultiIndex) -> tuple[list[int],list[str]]:
    """Calc locs of class edges at y axis"""
    new_locs = []
    new_labels = []
    cur_class = ''
    for i in range(index.size-1):
        if cur_class != index[i][0]:
            new_locs.append(i)
            new_labels.append(index[i][0])
            cur_class = index[i][0]
    new_locs.append(index.size)
    new_labels.append(index[-1][0])
    return new_locs, new_labels


@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_extreme_ticks(ticks: np.ndarray) -> tuple[list[int],list]:
    """Calc extreme points locs in array of ticks"""
    new_locs = []
    for i in range(1, ticks.size-1):
        if (ticks[i-1] < ticks[i] > ticks[i+1]) or (ticks[i-1] > ticks[i] < ticks[i+1]):
            new_locs.append(i)
    return new_locs, []

    
@validate_call(config=dict(arbitrary_types_allowed=True))
def _get_ticks(ticks: np.ndarray, step: PositiveInt) -> tuple[list[int],list[str]]:
    """Filter ticks by applying the step. The first and last tick are used always"""
    new_locs = []
    new_labels = []
    for i in range(0, ticks.size-1, step):
        new_locs.append(i)
        new_labels.append(ticks[i])
    new_locs.append(len(ticks)-1)
    new_labels.append(ticks[-1])
    return new_locs, new_labels


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