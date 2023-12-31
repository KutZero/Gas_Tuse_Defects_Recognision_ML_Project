import os

__all__ = ["PathToCsv"]

class PathToCsv:
    """Path to .csv files variables descriptor"""
    
    @classmethod
    def is_path_correct(cls, path: str) -> bool:
        """
        Check is the path correct
        
        Parameters
        ----------
        path: str
            The path to a .csv file.
            
        Returns
        -------
        out : bool
            True if the path is correct and locates to an existing file.
            
        Raises
        ------
        TypeError
            1. If the path is not str type.
        ValueError
            1. If the path is not exist or don't locate to a .csv file.

        """
        if not type(path) is str:
            raise TypeError("A path should be str")
        if not os.path.isfile(path) or not path.endswith('.csv'):
            raise ValueError("A path is not exist or don't not locate to a .csv file")
        return True
    
    def __set_name__(self, owner, name):
        self.name = '_' + name
        
    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if self.is_path_correct(value):
            setattr(instance, self.name, value)