import pandas as pd
import numpy as np

# TODO: Use to standardize the data format

# Try to make the class iterable

class TimeSeries():
    def __init__(self, data, columns=None):
        """
        Parameters:
        
        data: pandas.DataFrame or array-like
            The dataset to calculate features for.
        """
        if isinstance(data, pd.DataFrame):
            self.data, self.labels = self.parse_from_dataframe(data)
        elif isinstance(data, pd.Series):
            self.data, self.labels = self.parse_from_series(data)
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            self.data, self.labels = self.parse_from_array(data)
        else:
            raise ValueError("Data format not supported.")
        # self.data = data

    def parse_from_dataframe(self, data):
        pass
    
    def parse_from_array(self, data):
        pass

    def parse_from_series(self, data):
        pass
