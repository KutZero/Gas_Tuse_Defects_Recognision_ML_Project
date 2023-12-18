import os

# path variables descriptor
class PathToCsv:
    @classmethod
    def is_path_correct(cls, path):
        if not type(path) is str:
            raise TypeError("A path should be str")
        if not os.path.isfile(path):
            raise ValueError("A path is not exist or is not locate to a file")
        if not path.endswith('.csv'):
            raise ValueError("A path should locate to a csv file")
        return True
    
    def __set_name__(self, owner, name):
        self.name = '_' + name
        
    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if self.is_path_correct(value):
            setattr(instance, self.name, value)