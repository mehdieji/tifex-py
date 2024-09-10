import package_name.feature_extraction.statistical_feature_calculators as statistical_feature_calculators
import package_name.feature_extraction.spectral_features_calculators as spectral_features_calculators
from package_name.feature_extraction.settings import StatisticalFeatureParams
from package_name.utils.data import TimeSeries

def calculate_all_features(data, params=None, window_size=None, columns=None, signal_name=None):
    """
    Calculates statisctical, spectral, and time frequency features for the
    given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    params: 
    
    """
    modules = [statistical_feature_calculators,
               spectral_features_calculators]
    calculators = get_calculators(modules)
    
    calculate_ts_features(data, calculators, params=params, window_size=window_size)

def calculate_statistical_features(data, params=None, window_size=None, columns=None, signal_name=None):
    calculators = get_calculators([statistical_feature_calculators])
    features = calculate_ts_features(data, calculators, params=params, window_size=window_size)
    return features

def calculate_spectral_features():
    pass

def calculate_time_frequency_features():
    pass

def calculate_ts_features(data, calculators, params=None, window_size=None, columns=None):
    # Standardize data format
    time_series = TimeSeries(data, columns=columns)

    if params is None:
        params = StatisticalFeatureParams(window_size)

    param_dict = params.get_settings_as_dict()

    # TODO: somehow iterate through time series
    features = calculate_features(time_series.data, calculators, param_dict)

    return features

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

        print(name, feature)
        for n, f in zip(name, feature):
            features[n] = f

    return features

# TODO: Change this to look at function attribute
def get_calculators(modules):
    calculators = []
    for m in modules:
        module_calculators = [v for k, v in m.__dict__.items() if k.startswith("calculate_") and not hasattr(v, 'exclude')]
        calculators.extend(module_calculators)
    return calculators
