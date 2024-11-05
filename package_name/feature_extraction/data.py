import pandas as pd
import numpy as np
from scipy.signal import welch


class SignalFeatures():
    """
    Container class for the features of a time series and its labels.

    Attributes:
    -----------
    label : str
        The label of the time series.
    features : dict
        Dictionary with the features of the time series.
    """
    def __init__(self, label, features) -> None:
        self.label = label
        self.features = features

class TimeSeries():
    """
    Class to parse a dataset into a list of univariate time series. The class
    can parse data from a pandas DataFrame, a pandas Series or an array-like
    object. The class is iterable, so it can be used in a for loop to iterate
    over the time series.

    Attributes:
    -----------
    data : list
        List of tuples with the series name and the corresponding time series.
    columns : list
        List of columns to parse. If None, all columns are parsed.
    index : int
        Index to keep track of the current time series.
    """
    def __init__(self, data, columns=None, name=None):
        if isinstance(data, pd.DataFrame):
            self.data = self.parse_from_dataframe(data, columns=columns)
        elif isinstance(data, pd.Series):
            self.data = self.parse_from_series(data)
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = self.parse_from_array(data, columns=columns)
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

    def parse_from_dataframe(self, data, columns=None):
        """
        Parse a pandas DataFrame into a list of univariate time series.

        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset to parse.
        columns : list
            List of columns to parse. If None, all columns are parsed.
        
        Returns:
        --------
        ts_list : list
            List of tuples with the series name and the corresponding time series.
        """
        ts_list = []
        if columns is None:
            columns = data.columns
        for column in columns:
            ts_list.append(self.create_data_dict(data[column].values, column))
        return ts_list

    def parse_from_array(self, data, columns=None):
        """
        Parse an array into a list of univariate time series.

        Parameters:
        -----------
        data : array-like
            The dataset to parse. Should have shape (T) or (T, N).
        columns : list
            Names of data columns. If not None, should match the number of
            columns in the array.

        Returns:
        --------
        ts_list : list
            List of tuples with the series name and the corresponding time series.
        """
        ts_list = []

        if len(data.shape) == 1:
            if columns is None:
                columns = [0]
            label = columns[0]
            ts_list.append(self.create_data_dict(data, label))
        elif len(data.shape) == 2:
            if columns is None:
                columns = list(range(data.shape[1]))
            for i, c in enumerate(columns):
                ts_list.append(self.create_data_dict(data[:, i], c))
        else:
            raise ValueError("Arrays with more than 2 dimensions are not supported.")
        return ts_list

    def parse_from_series(self, data):
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

        name = data.name
        return [self.create_data_dict(data.values, name)]

    def create_data_dict(self, signal, name):
        """
        Create a dictionary with the signal and the label.

        Parameters:
        -----------
        signal : np.array
            The time series data.
        name : str
            The name of the time series.

        Returns:
        --------
        dict
            Dictionary with the signal and the label.
        """    
        return {"signal": signal, "label": name}


class SpectralTimeSeries(TimeSeries):
    def __init__(self, data, columns=None, fs=1.0):
        """
        Parameters:
        ----------
        data: pandas.DataFrame or array-like
            The dataset to calculate features for.
        columns: list
            List of columns to parse. If None, all columns are parsed.
            If data is an array, this is the list of column names.
        """
        self.fs = fs
        super().__init__(data, columns=columns)

    def create_data_dict(self, signal, name):
        """
        Creates a dictionary with the signal and its label as well as the
        frequency spectrum and the power spectral density of the signal.

        Parameters:
        -----------
        signal : np.array
            The time series data.
        name : str
            The name of the time series.
        
        Returns:
        --------
        dict
            Dictionary with the signal, the label, the frequency spectrum
            and the power spectral density.
        """
        # FFT (only positive frequencies)
        spectrum = np.fft.rfft(signal)  # Spectrum of positive frequencies
        spectrum_magnitudes = np.abs(spectrum)  # Magnitude of positive frequencies
        spectrum_magnitudes_normalized = spectrum_magnitudes / np.sum(spectrum_magnitudes)
        length = len(signal)
        freqs_spectrum = np.abs(np.fft.fftfreq(length, 1.0 / self.fs)[:length // 2 + 1])

        # Calculating the power spectral density using Welch's method.
        freqs_psd, psd = welch(signal, fs=self.fs)
        psd_normalized = psd / np.sum(psd)

        return {"signal": signal, "spectrum": spectrum, "magnitudes": spectrum_magnitudes,
                "magnitudes_normalized": spectrum_magnitudes_normalized, "freqs": freqs_spectrum,
                "psd": psd, "psd_normalized": psd_normalized, "freqs_psd": freqs_psd, "label": name}
