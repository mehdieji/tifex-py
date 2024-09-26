import pandas as pd
import multiprocessing as mp
from functools import partial

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

    calculate_ts_features(data, ["statistical", "spectral", "timefreq"], params=params, window_size=window_size,
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
    features = calculate_ts_features(data, ["statistical"], params=params, window_size=window_size,
                                     columns=columns, signal_name=signal_name, njobs=njobs)
    return features

def calculate_spectral_features():
    pass

def calculate_time_frequency_features():
    pass

def calculate_ts_features(data, modules, params=None, window_size=None, columns=None, signal_name=None, njobs=None):
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

    pool = mp.Pool(njobs)

    results = pool.imap(partial(calculate_features, modules=modules, param_dict=param_dict), time_series)

    for r in results:
        index.append(r[0])
        features.append(r[1])

    features_df = pd.DataFrame(features, index=index)
    return features_df

def get_modules(module_str):
    """
    Get a list of modules corresponding to the given module strings.

    Parameters:
    ----------
    module_str: list
        List of strings representing the modules to use.

    Returns:
    -------
    modules: list
        List of modules.
    """
    modules = []
    for name in module_str:
        if name=="statistical":
            modules.append(statistical_feature_calculators)
        elif name=="spectral":
            modules.append(spectral_features_calculators)
    return modules

def calculate_features(series, modules, param_dict):
    """
    Calculate features for the given univariate time series data.

    Parameters:
    ----------
    series: tuple of a string and pandas.DataFrame or array-like
        The name of the dataset to calculate features for and the data
        itself.
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
    calculators = get_calculators(get_modules(modules))
    for calculate in calculators:
        feature = calculate(series[1], **param_dict)
        name = getattr(calculate, "names")

        if isinstance(name, list):
            for n, f in zip(name, feature):
                features[n] = f
        else:
            features[name] = feature

    return series[0], features

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
