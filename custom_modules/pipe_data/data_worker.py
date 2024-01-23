
class DataWorker:
    # The path to a data file like "*_data.csv"
    path_to_data_file = PathToCsv()
    # The path to a data file like "*_defects.csv"
    path_to_defects_file = PathToCsv()
    # The path to a data file like "*_pipe.csv"
    path_to_pipe_file = PathToCsv()
    
    def __init__(self, 
                 path_to_data_file: str,
                 path_to_defects_file: str,
                 path_to_pipe_file: str):
        self.path_to_data_file = path_to_data_file
        self.path_to_defects_file = path_to_defects_file
        self.path_to_pipe_file = path_to_pipe_file
        
        data_df = self._get_df_from_data_file()
        defects_df = self._get_df_from_defects_file()
        
        defects_arr = np.array((defects_df.shape[0],
                               defects_df.shape[1],
                               65))
        
        for i in range(defects_arr.shape[0]):
            for j in range(defects_arr.shape[1]):
                defects_arr[i,j,:32] = defects_df.iloc[i,j][:32]
                defects_arr[i,j,32:-1] = defects_df.iloc[i,j][32:]
                defects_arr[i,j,-1] = 0

        
        

    def _get_df_from_defects_file(self):
        """Read data file like "*_defects.csv" and returns
           pandas dataframe with preprocessed data from it"""
        using_columns = ['row_min', 'row_max', 'detector_min', 
                         'detector_max', 'fea_depth']
        df = pd.read_csv(self.path_to_defects_file, delimiter=';')
        return df[using_columns]

    @staticmethod
    def _split_cell_string_value_to_numpy_array_of_64_values(df_cell_value):
            """Converte all data cells values from given pandas dataframe from
               string (describes 2D values array) to 1D float numpy array of 64 items"""
            num_pars = re.findall(r'(-?[0-9]+(\.[0-9]+)*):(-?[0-9]+(\.[0-9]+)*)', df_cell_value)
            num_pars = np.array([[item[0], item[2]] for item in num_pars]).astype(float)
            
            if num_pars.size == 0:
                return np.zeros((64))
            
            time_vals = num_pars[:,0]
            amp_vals = num_pars[:,1]

            time_vals = np.pad(time_vals, (0, abs(time_vals.size-32)), constant_values=(0))
            amp_vals = np.pad(amp_vals, (0, abs(amp_vals.size-32)), constant_values=(0))
            
            return np.concatenate((time_vals, amp_vals), axis=0)
    
    def _get_df_from_data_file(self):
        """Read data file like "*_data.csv" and returns
           pandas dataframe with preprocessed data from it"""
    
        df = pd.read_csv(self.path_to_data_file, delimiter=';').astype(str)
        df = df.drop(['position'], axis=1)
        df = df.set_index('row')
        df = df.map(split_cell_string_value_to_numpy_array_of_64_values)
        return df
      