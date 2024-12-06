import os
from typing import Callable, Optional
from pydantic import BaseModel, field_validator, computed_field, PositiveInt
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated
import pathlib 

class DataPart(BaseModel): 
    """
    The dataclass to store describtion to preprocess data
    to sequance of crops.

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

    The result of reading is a pandas.DataFrame (df) with columns - 
    pandas.Index with names: 'DetectorNum' and dtype = 'int64'.
    The df's rows is pandas.MultiIndex with levels: 'File', 
    'ScanNun' and dtypes 'str' ('object' in pandas) and 'int64'.
    Every cell stores single dimensional numpy.ndarray of size
    65 where first 32 values - time ones, second 32 values - 
    amplitude ones, and the last value - defect depth.
    The df can store data for quantity of run scanning files
    and also can mix them. So the DataPart only describes the
    part of the df which is taken to make dataset of sequance
    of crops.

        The df:
    columns:['DetectorNum']
    index:['File', 'ScanNum']+-------------------detectors nums------------------+
    |            |           |                                                   |
    |            |           |                DataPart:                          |
    |            |           |                (x,y)------width------+            |
    |        run's names     |                |                     |            |
    |            |       scans nums           |                  height          |
    |            |           |                |                     |            |
    |            |           |                +---------------------+            |
    |            |           |                                                   |
    -------------|----------+----------------------------------------------------+

    Parameters
    ----------
    run_name: str, optional
        The run name to preprocess. If None the crop will
        be performed to all dataframe without dividing by 
        file name
    xy: tuple[NonNegativeInt,NonNegativeInt], default = (0,0)
        The xy stores coords of the DataPart anchor point.
    width: PositiveInt, optional
        The width allows to set the DataPart width
    height: PositiveInt, optional
         The height allows to set the DataPart height
    crop_size: PositiveInt, default = 1
        Crop size value
    crop_step: PositiveInt, default = 1
        Crop step value
    augmentations: bool, default = False
        Use augmentations or not
    """
    run_name: Optional[str] = None
    xy: tuple[int,int] = (0,0)
    width: Optional[PositiveInt] = None
    height: Optional[PositiveInt] = None
    crop_size: PositiveInt = 1
    crop_step: PositiveInt = 1
    augmentations: bool = False
