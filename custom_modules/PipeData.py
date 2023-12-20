import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from matplotlib.text import Text

class PipeData:
    """Base abstract class for ones to work with pipe data"""
    
    _data_df = None # pd.DataFrame with input network model data
    _defects_df = None # pd.DataFrame with output network model data
    # dict with 'left','right','top','bottom' keys. Each key store
    # a value where original (not extended) dfs are located
    _extend = None 
    # cumulative dfs shifting history
    _shift = None

    @staticmethod
    def _extend_df_for_crops_dividing(df, crop_size, crop_step):
        """Inner help func for extend one df for exact crops dividing
        where crop have <crop_size> and <crop_step>"""
        new_rows = crop_step - ((df.shape[0] - crop_size) % crop_step)
        new_cols = crop_step - ((df.shape[1] - crop_size) % crop_step)
    
        if new_rows != crop_step:
            df = pd.concat([df, df.iloc[-2:-new_rows-2:-1]], axis=0)
        if new_cols != crop_step:
            df = pd.concat([df, df.iloc[:,-2:-new_cols-2:-1]], axis=1)
            
        return df
    
    def reset_dfs_to_original(self):
        """Reset dfs extendings and rollings"""
        assert not self._extend is None, '_extend parameter is not initialized'
        self.roll_dfs_along_axis(axis=0, default=True)
        self.roll_dfs_along_axis(axis=1, default=True)
        self._data_df = self._data_df.iloc[self._extend['top']:self._extend['bottom'],
                                            self._extend['left']:self._extend['right']] 
        self._defects_df = self._defects_df.iloc[self._extend['top']:self._extend['bottom'],
                                            self._extend['left']:self._extend['right']]
        self._extend['left'] = 0
        self._extend['top'] = 0
        self._extend['right'] = self._data_df.shape[1]
        self._extend['bottom'] = self._data_df.shape[0]
    
    def extend_dfs_for_crops_dividing(self, crop_size, crop_step):
        """Extend dfs to be divided by crops of size <crop_size>
        exactly with crop divide step - <crop_step>"""
        assert not self._extend is None, '_extend parameter is not initialized'
        self.reset_dfs_to_original()
        self._data_df = self._extend_df_for_crops_dividing(self._data_df, crop_size, crop_step)
        self._defects_df = self._extend_df_for_crops_dividing(self._defects_df, crop_size, crop_step)
        
    
    
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
        
        assert not self._data_df is None, '_data_df parameter is not initialized'
        assert not self._defects_df is None, '_defects_df parameter is not initialized'

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
            cols = [re.search('[0-9]+', col)[0] for col in df.columns.to_numpy()]
            cols = np.array(cols).astype(int)
            indexes = df.index.to_numpy().astype(int)
            
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
        assert not self._data_df is None, '_data_df parameter is not initialized'
        return self._data_df

    def get_defects_df(self):
        assert not self._defects_df is None, '_defects_df parameter is not initialized'
        return self._defects_df
    
    def roll_dfs_along_axis(self, shift: int = 0, axis: int = 0, default: bool = False):
        """Roll data_df and defects_df elements along a given axis"""
        def roll_df(df, shift: int = 0, axis: int = 0):
            """Roll any dataframe like numpy.roll method"""
            df_values = df.to_numpy()
            df_indexes = df.index.to_numpy()
            df_columns = df.columns.to_numpy()

            df_values = np.roll(df_values, shift, axis)
            if axis == 1:
                df_indexes = np.roll(df.index.to_numpy(), shift)
            if axis == 0:
                df_columns = np.roll(df.columns.to_numpy(), shift)
            
            return pd.DataFrame(data=df_values, index=df_indexes, 
                                columns=df_columns)

        assert not self._data_df is None, '_data_df parameter is not initialized'
        assert not self._defects_df is None, '_defects_df parameter is not initialized'
        assert not self._shift is None, '_shift parameter is not initialized'
        
        # index - axis, value - shift
        shift_arr = [0,0]
        shift_arr[axis] = shift

        if default:
            shift_arr[axis] = -1 * self._shift[axis]
            self._shift[axis] = 0
        else:
            self._shift[axis] += shift_arr[axis]
        
        self._data_df = roll_df(self._data_df, shift_arr[axis], axis)
        self._defects_df = roll_df(self._defects_df, shift_arr[axis], axis)
        











        