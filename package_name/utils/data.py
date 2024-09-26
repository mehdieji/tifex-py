import pandas as pd
import numpy as np


class TimeSeries():
    def __init__(self, data, columns=None, name=None):
        """
        Parameters:
        ----------
        data: pandas.DataFrame or array-like
            The dataset to calculate features for.
        columns: list
            List of columns to parse. If None, all columns are parsed.
            If data is an array, this is the list of column names.
        name: str
            Name to prepend to the column names.
        """
        if isinstance(data, pd.DataFrame):
            self.data = self.parse_from_dataframe(data, columns=columns, name=name)
        elif isinstance(data, pd.Series):
            self.data = self.parse_from_series(data, name=name)
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = self.parse_from_array(data, columns=columns, name=name)
        else:
            raise ValueError("Data format not supported.")

    def __iter__(self):
        """
        Make the class iterable.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Get the next time series.
        """
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def parse_from_dataframe(self, data, columns=None, name=None):
        """
        Parse a pandas DataFrame into a list of univariate time series.

        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to parse.
        columns : list
            List of columns to parse. If None, all columns are parsed.
        name : str
            Name to prepend to the column names.
        
        Returns:
        --------
        ts_list : list
            List of tuples with the series name and the corresponding time series.
        """
        ts_list = []
        if columns is None:
            columns = data.columns
        for column in columns:
            if name is not None:
                label = f"{name}_{column}"
            else:
                label = column
            ts_list.append((label, data[column].values))
        return ts_list
    
    def parse_from_array(self, data, columns=None, name=None):
        """
        Parse an array into a list of univariate time series.

        Parameters:
        -----------
        data : array-like
            The dataset to parse. Should have shape (T) or (T, N).
        columns : list
            Names of data columns. If not None, should match the number of
            columns in the array.
        name: str
            Name to prepend to the column names.

        Returns:
        --------
        ts_list : list
            List of tuples with the series name and the corresponding time series.
        """
        ts_list = []
    
        if len(data.shape) == 1:
            if columns is None:
                columns = [0]
            if name is not None:
                label = f"{name}_{columns[0]}"
            else:
                label = columns[0]
            ts_list.append((label, data))
        elif len(data.shape) == 2:
            if columns is None:
                columns = list(range(data.shape[1]))
            for i in columns:
                if name is not None:
                    label = f"{name}_{i}"
                else:
                    label = i
                ts_list.append((label, data[:, i]))
        else:
            raise ValueError("Arrays with more than 2 dimensions are not supported.")
        return ts_list

    def parse_from_series(self, data, name=None):
        """
        Parse a pandas Series into a list of univariate time series.

        Parameters:
        -----------
        data : pandas.Series
            The dataset to parse.
        name : str
            Name to prepend to the column names.

        Returns:
        --------
        ts_list : list
            List of tuples with the series name and the corresponding time series.
        """
        if name is not None:
            name = f"{name}_{data.name}"
        else:
            name = data.name
        return [(name, data.values)]
