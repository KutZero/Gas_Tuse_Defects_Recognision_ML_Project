import os
from typing import Callable, Optional, Generator
from pydantic import BaseModel, field_validator, computed_field, PositiveInt
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated
import pathlib 

class DataPart(BaseModel): 
    """
    The dataclass to store describtion to read determine pipeline
    zone scanning result fully or partially with or without some
    preprocessing. A function could use this data to read 
    scanning data map and defect depths and locations map.

    The dataset consists of pipeline-zone-ultrasonic-scanning
    results. An each result being stored in a separate folder 
    and consists of 3 files: file like "*_data.csv", file like 
    "*_defects.csv" and file like "*_pipe.csv".
    
    The "*_data.csv" file stores raw scanning data. The file
    shape = (scans count, detectors count, points count, time+value 
    pair). The scans count could be from 0 to inf. The detectors
    count equals 400 at avaliable data. The dots count could be
    from 0 to 32 and represents received mirrored from a pipeline's 
    internal wall ultrasonic wave params pairs count. time+value - 
    are the params of each point.
    
    The  "*_defects.csv" file stores defects describtions found
    in the pipeline's scanned zone labeled by experts.

    The "*_pipe.csv" file stores common params of the inspected
    pipeline.

        "*_data.csv" file":
        +-------------------detectors count-----------------+
        |                                                   |
        |                DataPart:                          |
        |                (x,y)------width------+            |
        |                 |                    |            |
    scans count           |                  height         |
        |                 |                    |            |
        |                 ---------------------+            |
        |                                                   |
        +---------------------------------------------------+

    Parameters
    ----------
    path_to_run_folder: os.PathLike
        The path_to_run_folder leads to a determine pipeline zone scanning 
        result data files. The folder should store: 
        file like "*_data.csv"; 
        file like "*_defects.csv";
        file like "*_pipe.csv"
    xy: tuple[NonNegativeInt,NonNegativeInt]
        The xy stores coords of the DataPart anchor point.
    width: Optional[PositiveInt]
        The width allows to set the DataPart width
    height: Optional[PositiveInt]
         The height allows to set the DataPart height
    unify_func: Optional[Callable]
        The unify_func allows to apply unification function to raw 
        ultrasonic scanning data and defects depths data.
    unify_separatly: bool
        The unify_separatly allows to choose either unify time and
        amplitude values together (i.e. Divide all items by global
        absolute max for example) or separatly (i.e. Divite time items 
        by time item absolute max and divide amplitude items by amplitude
        item absolute max).
    """
    path_to_run_folder: os.PathLike
    xy: tuple[int,int] = (0,0)
    width: Optional[PositiveInt] = None
    height: Optional[PositiveInt] = None
    unify_func: Optional[Callable] = None
    unify_separatly: bool = True

    def _find_and_validata_data_file(self, rglob_pattern: str):
        res = set(pathlib.Path(self.path_to_run_folder).rglob(rglob_pattern))
        if len(res) != 1:
            raise ValueError(f'The path_to_run_folder should store one "{rglob_pattern}" file, but found: {res}')
        return res.pop()
    
    @computed_field
    @property 
    def data_path(self) -> os.PathLike:
        return self._find_and_validata_data_file('*_data.csv')
    
    @computed_field
    @property 
    def defects_path(self) -> os.PathLike:
        return self._find_and_validata_data_file('*_defects.csv')
    
    @computed_field
    @property 
    def pipe_path(self) -> os.PathLike:
        return self._find_and_validata_data_file('*_pipe.csv')
    
    @field_validator('path_to_run_folder')
    def is_path_dir(cls, value):
        if os.path.isdir(value):
            return value
        raise ValueError('The path_to_run_folder should be path to a dir')
