# Description: Utility functions for the package.
import numpy as np
import pandas as pd
from tifex_py.feature_extraction.data import SignalFeatures
import copy as cp


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


def extract_features(calculator, series, param_dict):
    """
    Calculate features for the given univariate time series data.

    Parameters:
    ----------
    series: tuple of a string and pandas.DataFrame or array-like
        The name of the dataset to calculate features for and the data
        itself.
    calculator: calculator function
        The calculator function to use for feature extraction.
    param_dict: dict
        Dictionary of parameters to pass to the feature calculators.

    Returns:
    -------
    features: dict
        Dictionary of calculated features.
    """
    features = []

    for i in range(len(series.data)):
        # for data in series:
        data = series.data[i]

        feature, name = calculator(**data, **param_dict)

        if feature is None:
            print(f"Feature(s) {name} will be set to Nan.")
            if isinstance(name, list):
                feature = [np.nan] * len(name)
            else:
                feature = np.nan
        feature = structure_features(feature, name)

        features.append(
            {
                "label": data["label"],
                "feature": feature,
                "idx": data.get("idx",None),
            }
        )
        
    return features


def extract_signals(calculator, series, param_dict):
    """
    Calculate signals for the given univariate time series data.

    Parameters:
    ----------
    series : Array-like
        The data to calculate signals for.
    calculator: calculator function
        The calculator function to use for feature extraction.
    param_dict: dict
        Dictionary of parameters to pass to the feature calculators.

    Returns:
    -------
    signals: array-like
        Array of calculated signals.
    """
    name = None
    # make a copy of the data to avoid modifying the original data
    series_data = series.data
    output_series_data = {}
    params = None
    for idx in range(len(series_data)):

        data = series_data[idx]
        result, name = calculator(**data, **param_dict)

        if result["signal"] is None:
            print(f"Error calculating signal {name}.")
        params = result["params"]

        if type(name) is str:
            if name not in output_series_data:
                output_series_data[name] = cp.deepcopy(series.data)
            output_series_data[name][idx]["signal"] = result["signal"]
        elif type(name) is list:
            for name_idx, n in enumerate(name):
                if n not in output_series_data:
                    output_series_data[n] = cp.deepcopy(series.data)

                output_series_data[n][idx]["signal"] = result["signal"][name_idx]
    if type(name) is str:
        name = [name]
    return output_series_data, params, name


def structure_features(feature, name):
    features = {}
    if isinstance(feature, SignalFeatures):
        for k, v in feature.features.items():
            features[f"{name}_{k}"] = v

    elif isinstance(name, list):
        if isinstance(feature[0], SignalFeatures):
            for n, f in zip(name, feature):
                for k, v in f.features.items():
                    features[f"{n}_{k}"] = v

        else:
            if len(name) != len(feature):
                print(
                    f"Feature {name} has a different number of values than the feature itself."
                )

            for n, f in zip(name, feature):
                features[n] = f

    else:
        features[name] = feature

    return features


def structure_results(results, group_name=None):
    all_features = {}
    for element in results:
        for item in element:
            label = item["label"]
            features = item["feature"]
            idx = item.get("idx", None)
            if all_features.get(idx) is None:
                all_features[idx] = {}
            for name, val in features.items():
                if all_features[idx].get(label) is None:
                    all_features[idx][label] = {}
                if group_name is not None:
                    all_features[idx][label][f"{group_name}_{name}"] = val
                else:
                    all_features[idx][label][name] = val

    all_outputs = []
    for idx, data in all_features.items():  # Per each sample
        sample_output = []
        for label, features in data.items():  # Per each label
            row = {"label": label}
            for name, value in features.items():  # Per each feature
                row[name] = value
            sample_output.append(row)
        df = pd.DataFrame(sample_output)
        # if idx==0:
        #     print(data)

        df.set_index("label", inplace=True)
        all_outputs.append(df)
    return all_outputs
