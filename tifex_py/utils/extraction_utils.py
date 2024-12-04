# Description: Utility functions for the package.
import numpy as np

from tifex_py.feature_extraction.data import SignalFeatures


def get_calculators(module, calculator_list=None):
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
    for k, v in module.__dict__.items():
        if k.startswith("calculate_") and not hasattr(v, 'exclude'):
            if calculator_list is not None:
                if k.replace("calculate_", "") in calculator_list:
                    calculators.append(v)
            else:
                calculators.append(v)
    return calculators

def get_module(module_str):
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
    module = None
    if module_str=="statistical":
        from tifex_py.feature_extraction import statistical_feature_calculators
        module = statistical_feature_calculators
    elif module_str=="spectral":
        from tifex_py.feature_extraction import spectral_feature_calculators
        module = spectral_feature_calculators
    elif module_str=="time_frequency":
        from tifex_py.feature_extraction import time_frequency_feature_calculators
        module = time_frequency_feature_calculators
    return module

def extract_features(series, module, param_dict):
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
    calculators = get_calculators(get_module(module), param_dict["calculators"])
    for calculate in calculators:
        try:
            feature = calculate(**series, **param_dict)
        except Exception as e:
            name = getattr(calculate, "names")
            print(f"Error calculating feature(s) {name}: {e}")
            print(f"Feature(s) {name} will be set to Nan.")
            if isinstance(name, list):
                feature = [np.nan] * len(name)
            else:
                feature = np.nan

        name = getattr(calculate, "names")

        if isinstance(feature, SignalFeatures):
            for k, v in feature.features.items():
                features[f'{name}_{k}'] = v
        elif isinstance(name, list):
            if isinstance(feature[0], SignalFeatures):
                for n, f in zip(name, feature):
                    for k, v in f.features.items():
                        features[f'{n}_{k}'] = v
            else:
                if len(name) != len(feature):
                    print(f"Feature {name} has a different number of values than the feature itself.")

                for n, f in zip(name, feature):
                    features[n] = f
        else:
#            if hasattr(feature, "__iter__"):
#                print(f"Feature {name} has more than one value.")
            features[name] = feature

    if "label" in series:
        label = series["label"]
    else:
        label = ""

    return SignalFeatures(label, features)
