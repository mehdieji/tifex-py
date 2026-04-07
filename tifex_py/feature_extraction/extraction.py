import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

import tifex_py.feature_extraction.statistical_feature_calculators as statistical_feature_calculators
import tifex_py.feature_extraction.spectral_feature_calculators as spectral_feature_calculators
import tifex_py.feature_extraction.time_frequency_feature_calculators as time_frequency_feature_calculators
from tifex_py.feature_extraction.settings import (
    StatisticalFeatureParams,
    SpectralFeatureParams,
    TimeFrequencyFeatureParams,
)
from tifex_py.utils.extraction_utils import (
    get_calculators,
    extract_features,
    get_module,
    extract_signals,
    structure_results,
    split_input_into_batches,
    save_features,
)
from tifex_py.feature_extraction.data import TimeSeries, SpectralTimeSeries


def calculate_features(
    data,
    feature_type="statistical",
    params=None,
    window_size=None,
    columns=None,
    njobs=None,
    store_features=True,
):
    """
    Calculates the specified type of features for the given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    feature_type: str {"statistical", "spectral", "time_frequency"}
        The type of features to calculate.
    params: StatisticalFeatureParams or SpectralFeatureParams or TimeFrequencyFeatureParams
        Parameters to use in feature extraction.
    window_size: int
        Window size to use for feature extraction.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    njobs: int
        Number of worker processes to use. If None or -1, the number returned by
        os.cpu_count() is used.
    store_features: bool
        Whether to store the calculated features in a file. If True, the features are stored in the specified directory. If False, the features are returned as a list of DataFrames. Default is True.
    Returns:
    -------
    features: list of pandas.DataFrame
        List of DataFrames containing the calculated features.
    """
    if feature_type == "statistical":
        feature_function = calculate_statistical_features
        if params is not None and type(params).__name__ != "StatisticalFeatureParams":
            raise ValueError(
                "For statistical feature extraction, params should be an instance of StatisticalFeatureParams."
            )
    elif feature_type == "spectral":
        feature_function = calculate_spectral_features
        if params is not None and type(params).__name__ != "SpectralFeatureParams":
            raise ValueError(
                "For spectral feature extraction, params should be an instance of SpectralFeatureParams."
            )
    elif feature_type == "time_frequency":
        feature_function = calculate_time_frequency_features
        if params is not None and type(params).__name__ != "TimeFrequencyFeatureParams":
            raise ValueError(
                "For time frequency feature extraction, params should be an instance of TimeFrequencyFeatureParams."
            )
    else:
        raise ValueError(
            "Invalid feature type. Please choose from 'statistical', 'spectral', or 'time_frequency'."
        )

    batch_groups = split_input_into_batches(data, 4000)
    batch_data = []
    for batch in batch_groups:
        print(f"Processing batch {batch} for {feature_type} feature extraction.")
        input_data = data[batch[0] : batch[1]]
        features = feature_function(
            input_data, params=params, columns=columns, njobs=njobs
        )
        if store_features:
            save_features(
                features,
                batch,
                f"datasets_module/tifex_features/REALWORLD_all_param/{feature_type}_features",
            )
        else:
            batch_data.append(features)
    if not store_features:
        return batch_data


def calculate_all_features(
    data, stat_params, spec_params, tf_params, columns=None, njobs=None
):
    """
    Calculates statistical, spectral, and time frequency features for the
    given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    stat_params: StatisticalFeatureParams
        Parameters to use in statistical feature extraction.
    spec_params: SpectralFeatureParams
        Parameters to use in spectral feature extraction.
    tf_params: TimeFrequencyFeatureParams
        Parameters to use in time frequency feature extraction.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    njobs: int
        Number of worker processes to use. If None or -1, the number returned by
        os.cpu_count() is used.
    store_features: bool
        Whether to store the calculated features in a file. If True, the features are stored in the specified directory. If False, the features are returned as a list of DataFrames.

    Returns:
    -------
    features: list of pandas.DataFrame
        List of DataFrames containing the calculated features.
    """
    batch_groups = split_input_into_batches(data, 4000)
    batch_data = []
    for batch in batch_groups:
        print(f"Processing batch {batch} for all feature extraction.")
        input_data = data[batch[0] : batch[1]]
        stat_features = calculate_statistical_features(
            input_data, stat_params, columns=columns, njobs=njobs
        )
        spec_features = calculate_spectral_features(
            input_data, spec_params, columns=columns, njobs=njobs
        )
        tf_features = calculate_time_frequency_features(
            input_data, tf_params, columns=columns, njobs=njobs
        )

        output_features = []
        for idx in range(len(stat_features)):
            output_features.append(
                pd.concat(
                    [stat_features[idx], spec_features[idx], tf_features[idx]], axis=1
                )
            )

        if store_features:
            save_features(
                output_features,
                batch,
                f"datasets_module/tifex_features/REALWORLD_all_param/all_features",
            )
        else:
            batch_data.append(output_features)

    if not store_features:
        return batch_data


def calculate_statistical_features(
    data, params=None, window_size=None, columns=None, njobs=None
):
    """
    Calculates statistical features for the given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    params: StatisticalFeatureParams
        Parameters to use in feature extraction.
    window_size: int
        Window size to use for feature extraction.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    njobs: int
        Number of worker processes to use. If None or -1, the number returned by
        os.cpu_count() is used.

    Returns:
    -------
    features: pandas.DataFrame
        DataFrame of calculated features.
    """
    if params is None:
        params = StatisticalFeatureParams(window_size)

    time_series = TimeSeries(data, columns=columns)

    features = calculate_ts_features(time_series, "statistical", params, njobs=njobs)
    return features


def calculate_spectral_features(data, params=None, window_size=None, fs=None, columns=None, njobs=None):
    """
    Calculates spectral features for the given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    params: SpectralFeatureParams
        Parameters to use in feature extraction.
    window_size: int
        Window size to use for feature extraction.
    fs: float
        Sampling frequency of the data.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    njobs: int
        Number of worker processes to use. If None or -1, the number returned by
        os.cpu_count() is used.

    Returns:
    -------
    features: pandas.DataFrame
        DataFrame of calculated features.
    """
    if params is None:
        params = SpectralFeatureParams(fs,window_size=window_size)
    time_series = SpectralTimeSeries(
        data, columns=columns, fs=params.fs, nperseg=params.nperseg, n_fft=params.n_fft
    )
    features = calculate_ts_features(time_series, "spectral", params, njobs=njobs)
    return features


def calculate_time_frequency_features(
    data, params=None, window_size=None, columns=None, njobs=None
):
    """
    Calculates time frequency features for the given dataset.

    Parameters:
    ----------
    data: pandas.DataFrame or array-like
        The dataset to calculate features for.
    params: TimeFrequencyFeatureParams
        Parameters to use in feature extraction.
    window_size: int
        Window size to use for feature extraction.
    columns: list
        Columns to calculate features for or names of the np.array columns.
    njobs: int
        Number of worker processes to use. If None, the  or -1number returned by
        os.cpu_count() is used.

    Returns:
    -------
    features: pandas.DataFrame
        DataFrame of calculated features.
    """
    if params is None:
        params = TimeFrequencyFeatureParams(window_size)

    time_series = TimeSeries(data, columns=columns)
    features = calculate_time_frequency_ts_features(
        time_series, "time_frequency", params, njobs=njobs
    )
    return features


def calculate_ts_features(time_series, module, params, njobs=None):
    """
    Calculate features from the given module for the given time series data.

    Parameters:
    ----------
    time_series: TimeSeries
        The time series data to calculate features for.
    module: str
        The module with the feature calculators to use.
    params: BaseFeatureParams
        Parameters to use in feature extraction.
    njobs: int
        Number of worker processes to use. If None or -1, the number returned by
        os.cpu_count() is used.

    Returns:
    -------
    features_df: pandas.DataFrame
        DataFrame of calculated features.
    """
    if njobs is None or njobs == -1:
        njobs = os.cpu_count()

    features = []
    index = []

    pool = mp.Pool(njobs)

    param_dict = params.get_settings_as_dict()
    calculators = get_calculators(get_module(module), param_dict["calculators"])

    results = pool.imap(
        partial(
            extract_features,
            series=time_series,
            param_dict=param_dict,
        ),
        calculators,
    )

    all_features = structure_results(results)
    return all_features


def calculate_time_frequency_ts_features(time_series, module, params, njobs=None):
    """
    Calculate features from the given module for the given time series data.

    Parameters:
    ----------
    time_series: TimeSeries
        The time series data to calculate features for.
    module: str
        The module with the feature calculators to use.
    params: BaseFeatureParams
        Parameters to use in feature extraction.
    njobs: int
        Number of worker processes to use. If None or -1, the number returned by
        os.cpu_count() is used.

    Returns:
    -------
    features_df: pandas.DataFrame
        DataFrame of calculated features.
    """
    if njobs is None or njobs == -1:
        njobs = os.cpu_count()

    index = []

    pool = mp.Pool(njobs)

    param_dict = params.get_settings_as_dict()
    calculators = get_calculators(get_module(module), param_dict["calculators"])

    results = pool.imap(
        partial(
            extract_signals,
            series=time_series,
            param_dict=param_dict,
        ),
        calculators,
    )
    features = []
    for result in results:  # Per each time series calculator

        for name in result[2]:  # Per each signal name
            time_series.data = result[0][
                name
            ]  # Update the time series data with the calculated signal

            params = result[1]

            calculators = get_calculators(
                get_module("statistical"), params["calculators"]
            )  # Get the calculator function for the calculated signal
            # For each result -> get features from the signal
            feature_results = pool.imap(
                partial(
                    extract_features,
                    series=time_series,
                    param_dict=params,
                ),
                calculators,
            )

            structured_results = structure_results(feature_results, group_name=name)

            if len(features) == 0:
                features = structured_results
            else:
                for idx in range(len(structured_results)):
                    features[idx] = pd.concat(
                        [features[idx], structured_results[idx]], axis=1
                    )  # Concatenate features from different calculators

    return features
