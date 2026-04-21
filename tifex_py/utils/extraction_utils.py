# Description: Utility functions for the package.
import numpy as np
import pandas as pd
from tifex_py.feature_extraction.data import SignalFeatures
import copy as cp
import os
import pickle

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
    """
    Structure the calculated features into a dictionary with appropriate keys.
    Parameters:
    ----------
    feature: SignalFeatures or list of SignalFeatures or array-like
        The calculated feature(s) to structure.
    name: str or list of str
        The name(s) of the feature(s) to use as keys in the structured output.
    Returns:
    -------
        features: dict
            Dictionary of structured features with appropriate keys.
    """
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


def structure_results(results, nan_mask=None, pos_inf_mask=None, neg_inf_mask=None, group_name=None):
    """
    Structure the calculated features into a list of DataFrames, one per sample, with labels as index and features as columns.
    Parameters:
    ----------
    results: list of dict
        List of dictionaries containing the calculated features for each sample.
    nan_mask: float, optional
        Value to replace NaN values with. If None, NaN values are not replaced.
    pos_inf_mask: float, optional
        Value to replace positive infinity values with. If None, positive infinity values are not replaced.
    neg_inf_mask: float, optional
        Value to replace negative infinity values with. If None, negative infinity values are not replaced.
    group_name: str, optional
        Optional prefix to add to feature names for grouping purposes. If None, no prefix is added.
    Returns:
    -------
    all_outputs: list of pandas.DataFrame
        List of DataFrames containing the structured features for each sample.
    """
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
        
        # Fill Nan, pos inf, neg inf values if specified
        if nan_mask is not None:
            df.fillna(nan_mask, inplace=True)
        if pos_inf_mask is not None:
            df.replace([np.inf], pos_inf_mask, inplace=True)
        if neg_inf_mask is not None:
            df.replace([-np.inf], neg_inf_mask, inplace=True)
            
        df.set_index("label", inplace=True)
        all_outputs.append(df)
    return all_outputs

def split_input_into_batches(data, samples_per_file=4000):
    """
    Split the input data into batches based on the specified number of samples per file.
    Parameters:
    ----------
    data: list
        The input data to be split into batches.
    samples_per_file: int
        The number of samples to include in each batch/file. Default is 4000.
    Returns:
    -------
    batch_start_end_indices: list of tuple
        List of tuples containing the start and end indices for each batch.
    """
    batch_start_end_indices = []
    for i in range(0, len(data), samples_per_file):
        batch_start_end_indices.append((i, min(i + samples_per_file, len(data))))
    return batch_start_end_indices

def save_features(features, batch_start_end_index, output_path):
    """
    Save the calculated features to a file in the specified output path. The file is named based on the batch start and end indices.
    Parameters:
    features: list of pandas.DataFrame
        List of DataFrames containing the calculated features to be saved.
    batch_start_end_index: tuple
        Tuple containing the start and end indices of the batch for which the features were calculated.
    output_path: str
        The directory path where the features should be saved.
    Returns:
    None
    """
    start, end = batch_start_end_index
    os.makedirs(f"{output_path}", exist_ok=True)
    with open(
        f"{output_path}/features_samples_{start}_{end}.pkl",
        "wb",
    ) as f:
        pickle.dump(features, f)