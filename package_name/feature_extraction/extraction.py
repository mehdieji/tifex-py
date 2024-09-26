import pandas as pd

import package_name.feature_extraction.statistical_feature_calculators as statistical_feature_calculators
import package_name.feature_extraction.spectral_features_calculators as spectral_features_calculators
from package_name.feature_extraction.settings import StatisticalFeatureParams
from package_name.utils.data import TimeSeries

def calculate_all_features(data, params=None, window_size=None, columns=None, signal_name=None, njobs=None):
    """
    Calculates statisctical, spectral, and time frequency features for the
    given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    params: BaseFeatureParams
        Parameters to use in feature extraction.
    window_size: int
        Window size to use for feature extraction.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    signal_name: str
        Name to prepend to the column names.
    njobs: int
        Number of worker processes to use. If None, the number returned by
        os.cpu_count() is used.
    
    Returns:
    -------

    """
    modules = [statistical_feature_calculators,
               spectral_features_calculators]
    calculators = get_calculators(modules)
    
    calculate_ts_features(data, calculators, params=params, window_size=window_size,
                          columns=columns, signal_name=signal_name, njobs=njobs)

def calculate_statistical_features(data, params=None, window_size=None, columns=None, signal_name=None, njobs=None):
    """
    Calculates all statistical features for the given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    params: BaseFeatureParams
        Parameters to use in feature extraction.
    window_size: int
        Window size to use for feature extraction.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    signal_name: str
        Name to prepend to the column names.
    njobs: int
        Number of worker processes to use. If None, the number returned by
        os.cpu_count() is used.

    Returns:
    -------
    features: pandas.DataFrame
        DataFrame of calculated features.
    """
    calculators = get_calculators([statistical_feature_calculators])
    features = calculate_ts_features(data, calculators, params=params, window_size=window_size,
                                     columns=columns, signal_name=signal_name, njobs=njobs)
    return features

def calculate_spectral_features():
    pass

def calculate_time_frequency_features():
    pass

def calculate_ts_features(data, calculators, params=None, window_size=None, columns=None, signal_name=None, njobs=None):
    """
    Calculate features for the given time series data.
    """
    features = []
    index = []

    # Standardize data format
    time_series = TimeSeries(data, columns=columns, name=signal_name)

    if params is None:
        params = StatisticalFeatureParams(window_size)

    param_dict = params.get_settings_as_dict()

    for series in time_series:
        features.append(calculate_features(series[1], calculators, param_dict))
        index.append(series[0])

    features_df = pd.DataFrame(features, index=index)
    return features_df

def calculate_features(data, calculators, param_dict):
    """
    Calculate features for the given univariate time series data.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    calculators: list
        List of feature calculators to use.
    param_dict: dict
        Dictionary of parameters to pass to the feature calculators.

    Returns:
    -------
    features: dict
        Dictionary of calculated features.
    """
    features = {}
    for calculate in calculators:
        feature = calculate(data, **param_dict)
        name = calculate.names

        if isinstance(name, list):
            for n, f in zip(name, feature):
                features[n] = f
        else:
            features[name] = feature

    return features

def get_calculators(modules):
    """
    Get all calculator functions from the given modules. Will exclude functions
    with the 'exclude' attribute.

    Parameters:
    ----------
    modules: list
        List of modules to get the calculators from.
    
    Returns:
    -------
    calculators: list
        List of calculator functions.
    """
    calculators = []
    for m in modules:
        module_calculators = [v for k, v in m.__dict__.items() if k.startswith("calculate_") and not hasattr(v, 'exclude')]
        calculators.extend(module_calculators)
    return calculators
