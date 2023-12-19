import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from matplotlib.text import Text

class PipeData:
    _data_df = None
    _defects_df = None
    
    def draw_defects_map(self, 
                title: str = 'Развернутая карта дефектов',
                xlabel: str = 'Номер датчика', 
                ylabel: str = 'Номер измерения',
                x_ticks_step: int = 50,
                y_ticks_step: int = 20):
        """Draf a defects map from self._defects_df"""
        def calc_labels_and_locs(arr: np.ndarray, step=1):
            """Calc locs and labels for matplotlib graph"""
            def add_index_fillers(arr: np.ndarray, step: int=1, i: int=0):
                """Add fillers between sorted (0 to max) numpy array of int or float"""
                if i > arr.shape[0]-2:
                    return arr
                if arr[i+1] - arr[i] >= 1.5 * step:
                    arr = np.insert(arr,i+1,arr[i]+step)
                return add_index_fillers(arr, step, i+1)
                
            def cals_labels_paddings(locs: np.ndarray, step: int=1):
                """Calc labels paddings for avoid labels overlapping"""
                label_paddings = np.zeros(locs.shape)
                for i in range(locs.shape[0]-1):
                    if locs[i+1] - locs[i] < step / 2:
                        label_paddings[i] = 1
                return label_paddings
            
            locs = np.sort(np.unique(np.concatenate(
                            [np.where((arr == min(arr)) | (arr == max(arr)))[0],
                                   [0, arr.shape[0]-1]], axis=0)))
        
            locs = add_index_fillers(locs, step)
            labels = arr[locs]
            # labels paddings for avoid labels overlapping
            label_paddings = cals_labels_paddings(locs, step)
            return locs, label_paddings, labels
        
        assert not self._data_df is None, 'data_df is not initialized'
        assert not self._defects_df is None, 'defects_df is not initialized'

        with plt.style.context('dark_background'):
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
            
            df = self._defects_df
        
            # get columns and indexes list of int type
            cols = [int(re.search('[0-9]+', col)[0]) for col in df.columns.values]
            cols = np.array(cols)
            indexes = df.index.values
            
            xlocs, xlabel_paddings, xlabels = calc_labels_and_locs(cols, x_ticks_step)
            ylocs, ylabel_paddings, ylabels = calc_labels_and_locs(indexes, y_ticks_step)
            
            ax.pcolormesh(df)

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
                ytext_labels[i].set(x=or_x-ylabel_paddings[i]*0.02)

            ax.set_xticklabels(xtext_labels) 
            ax.set_yticklabels(ytext_labels) 
            
        plt.show()
    
    def get_data_df(self):
        assert not self._data_df is None, 'data_df is not initialized'
        return self._data_df

    def get_defects_df(self):
        assert not self._defects_df is None, 'defects_df is not initialized'
        return self._defects_df
        
    def roll_dfs_along_axis(self, shift: int, axis: int = 0):
        """Roll data_df and defects_df elements along a given axis"""
        def roll_df(df, shift, axis):
            """Roll any dataframe like numpy.roll method"""
            df_values = np.roll(df.to_numpy(), shift, axis)
            df_index = (df.index.to_numpy() if axis == 1 
                            else np.roll(df.index.to_numpy(), shift))
            df_columns = (df.columns.to_numpy() if axis == 0 
                              else np.roll(df.columns.to_numpy(), shift))
        
            return pd.DataFrame(data=df_values, index=df_index, 
                                columns=df_columns)

        assert not self._data_df is None, 'data_df is not initialized'
        assert not self._defects_df is None, 'defects_df is not initialized'

        self._data_df = roll_df(self._data_df, shift, axis)
        self._defects_df = roll_df(self._defects_df, shift, axis)