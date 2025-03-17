# TIFEX-Py – Time-series Feature Extraction for Python

## Introduction
- `TIFEX-Py` is a Python 3 toolbox developed for time-series feature extraction. `TIFEX-Py` is a versatile tool designed for analyzing time series data and extracting features across statistical and amplitude, spectral, and time-frequency domains. `TIFEX-Py` is built upon widely used Python data analysis libraries such as `NumPy` and `Pandas`, ensuring compatibility with standard data processing workflows. The toolbox is open-source and publicly available.
- The available features are detailed in the README_Features.md

## Installation
`TIFEX-Py` can be installed with pip:
```
pip install tifex-py
```
or, if you want to madify the package and add new features, it can be cloned and installed with
```
git clone https://github.com/mehdieji/tifex-py.git
cd tifex-py
pip install -e .
```

## Getting Started with Feature Extraction
First, you need to load the time series data:
```
filename = "/home/scai9/feature_dataset/USCHAD_data.csv"
dataset = pd.read_csv(filename)

print(data.head)

       accx      accy      accz
0 -0.653698  0.711878 -0.423886
1 -0.643397  0.711878 -0.431206
2 -0.643397  0.715498 -0.431206
3 -0.643397  0.711878 -0.427546
4 -0.653698  0.708259 -0.427546

```

Then, initialize the default feature extraction parameters for all feature categories.

```
statistical_params = settings.StatisticalFeatureParams(25)
spectral_params = settings.SpectralFeatureParams(25)
time_freq_params = settings.TimeFrequencyFeatureParams(25)
```

Then, we can extract all available features for the `accx`, `accy`, and `accz` time series as shown below.

```
features = extraction.calculate_all_features(dataset, statistical_params, spectral_params, time_freq_params, columns=["accx", "accy", "accz"])
```


For more detailed information on how to use the package, please see the provided [Jupyter notebook](notebooks/basics.ipynb). Here, 

## Data Formats


### Input Formats
#### numpy Arrays
1 or 2 dimensional `np.ndarray` can be used as an input for feature extraction. In 2D arrays, the columns are treated as the different dimensions of the time series. The `columns` parameter can be used to provide the ordered names of the dimensions of the time series to be used as the index in the output DataFrame. Otherwise, they will be labelled by the integer index of the dimension.

#### pandas Series
Similarly, the `pandas.Series` is an acceptable data format for a univariate time series.

#### pandas DataFrame
The `pandas.DataFrame` is an acceptable input format for both univariate and multivariate time series. In this case, the `columns` parameter can be used to specify which columns of the DataFrame features will be extracted from.

### Output Format
The output from feature extraction is a `pandas.DataFrame` where the columns correspond to the different extracted features and the rows correspond to the dimension(s) of the time series the features were extracted from.

|          | mean    | mode    | ...     | number_cwt_peaks_5 |
| -------- | ------- | ------- | ------- | ------------------ |
| accx     | 0.505   | 0.409   |...      | 182                |
| accy     | 0.482   | 0.332   | ...     | 33                 |
| accz     | 0.234   | 0.123   | ...     | 928                |

## Feature Extraction Settings
The parameters used calculating the features as well as the selection of which of the available features to extract are configured using the `*FeatureParams` classes. There is a different class for each category of features.

The parameters can be modified fromtheir default values when the class is initialized or by loading a YAML or JSON configuration file. A base file for the custom configuration can be created by saving the default parameters to a YAML or JSON file as shown below.

```
# Initialization of the feature extraction parameters
statistical_params = settings.StatisticalFeatureParams(25)
spectral_params = settings.SpectralFeatureParams(25)
time_freq_params = settings.TimeFrequencyFeatureParams(25)

# Save and load the parameters to JSON file
statistical_params.to_json("statistical_params.json")
statistical_params_2 = settings.StatisticalFeatureParams.from_json("statistical_params.json")

# Save and load the parameters to YAML file
statistical_params.to_yaml("statistical_params.yaml")
statistical_params_2 = settings.StatisticalFeatureParams.from_yaml("statistical_params.yaml")
```

## Parallelization
Parallelization is configured via the `n_jobs` parameter of the `calculate_*_features` functions. By default, `os.cpu_count()` is used.

## Adding New Features
TODO reorder sectios

The features are divided into three subcategories with a `*_features_calculator.py` file corresponding to each one: [statistical features](tifex_py/feature_extraction/statistical_feature_calculators.py), [spectral features](tifex_py/feature_extraction/spectral_feature_calculators.py), and [time-frequency features](tifex_py/feature_extraction/time_frequency_feature_calculators.py). The new feature should be added the file corresponding to the category it best fits under.

Specific implementation details will be described in the following sections.

If you think the added feature would be useful to others and you would like to contribute to __Tifex-Py__, please make a pull request.

### Writing the Function (Statistical and Spectral Features)
The general structure of a statistical or spectral feature calculator function should be as follows:

```
@name("feature_name")
def calculate_geometric_mean(signal, param1, param2, **kwargs):
    """
    Function description

    Parameters:
    ----------
    signal : np.array
        An array of values corresponding to the input signal.
    param1 : param1 type
        Description of param1.
    param2 : param2 type
        Description of param2.

    Returns:
    -------
    float/array-like
        Description of the return value.
    
    References:
    ----------
        - ...
    """
    ...
    value = some_calculation(signal, param1, param2)
    ...
    return value
```
The parameter name of `signal` must always be the same. The calculator-specific parameter names must be unique from parameter names used in other calculator functions.

Each calculator function should return either a float or array-like feature with a limited number of values as they will all be included as individual elements in the resultant dataframe of features.

Writing a time-frequency feature calculator is similar, but will be discussed in a later section.

### Adding Time Frequency Features
Time frequency feature calculators extract a variety of statistical features from a given representation of the signal. For example, `calculate_tkeo_features` extracts statistical features from the for the Teager-Kaiser energy operator representation of the time-series. 

The general structure of the calculator should be as follows:
```
@name("representation_name")
def calculate_representation_features(signal, param1, representation_sf_params, **kwargs):
    """
    ...
    """
    # Compute the representation of the signal you want to calculate features for
    representation = calculate_representation(signal, param1)

    # Extract statistical features from the representation
    return extract_features({"signal": representation}, "statistical", spectogram_sf_params)
```
The parameter name of `signal` must always be the same and there should always be one input parameter with a `StatisticalFeaturesParam` which will specificy the parameters to usein the various statistical feature calculations for the representation.

### Adding the Name Decorator
As seen in the example feature calculator function above, there is an `@name(...)` decorator. This is used to specify the feature name used in the output dataframe. If the output of the calculator is a float, the decorator input is just the string of the desired name. If the output is array-like there are a few options for specifying a unique name for each of the entries.

1. If the calculator alway outputs an array of the same length, a unique name for each entry of the array can be specified in a list input to `@name(...)`.

For example `calculate_hjorth_mobility_and_complexity` returns a 2-entry array with the mobility and complexity respectively. Thus, the decorator is `@name(["hjorth_mobility", "hjorth_complexity"])`.

2. If there is an array-like parameter that is the same length as the output array and helps differentiate the entries, the feature names can be specified with a formattable string and a string with the parameter name to be used to differentiate the names. 

For example, `calculate_higuchi_fractal_dimensions` returns an array with the Higuchi Fractal Dimension for each k value in the given list `higuchi_k_values`. Thus, the names can be specified with `@name("higuchi_fractal_dimensions_k={}", "higuchi_k_values")` where the first string will be formatted with the corresponding value from the list `higuchi_k_values`.

3. If there is an integer parameter which determines the number of values in the return array, like the number of bins in a histogram, the input to the `@name(...)` should be similar to above. However, the names will instead be formatted with values from 0 to N-1, whenthe integer has value N.

For example, `calculate_histogram_bin_frequencies` returns an array with a histogram of the signal values with `hist_bins` number of bins. The name is specified as `@name("histogram_bin_frequencies_{}", "hist_bins")` so that the first string is formatted with `0` to `hist_bins - 1` for the corresponding array values.

### Adding the Parameters to the Settings
Settings for all feature calculators are stored in a `BaseFeatureParams` class. A separate settings class is created for each category of features (statistical, spectral, time-frequency) and can all be seen in [settings.py](tifex_py/feature_extraction/settings.py).

When a new calculator is added to `*_feature_calculators.py`, any calculator-specific parameters must be added to the corresponding `*FeatureParams` class. A description of the parameter should be added in addition to specifying a reasonable default value.

When adding a new time-frequency feature calculator, you must also add an attribute holding the `StatisticalFeatureParams` to use for that specific representation. This addition must also be accounted for in the methods used to load and save parameters from a file.

### Spectral Feature Specific Considerations
May of the spectral feature calculators using the frequency spectrum, power spectrum density, etc. To prevent repeated calculations, the following are computed from the signal to be used as inputs to the spectral feature calculators under the specified names:

1. `spectrum`: One-dimensional discrete Fourier Transform for real input computed with `np.fft.rfft`.
2. `magnitudes`: `abs(spectrum)`
3. `magnitudes_normalized`: `magnitudes / sum(magnitudes)`
4. `freqs`: TODO
5. `psd`: Estimate of power spectral density using Welch's method.
6. `psd_normalized`: `psd / np.sum(psd)`
7. `freqs_psd`: Sample frequencies from psd computation.

In the implementation of your spectral feature calculator, please use these precomputed values by including the above-specified name as an input parameter in the calculator. `signal` can also still be used as a parameter to get the original signal.

For example, `calculate_spectral_cumulative_frequency_below(freqs, magnitudes, thresholds_freq_below, **kwargs)` uses the precomputed `freqs` and `magnitudes` arrays in addition to a calculator-specific threshold parameter.

# Credits

This project was developed as part of Mehdi Ejtehadi's PhD research at SCAI Lab, with significant contributions from all authors and guidance from our esteemed supervisors.

## Authors
- **Mehdi Ejtehadi**
- **Gloria Edumaba Graham**
- **Cailin Ringstrom**

## Supervisors
- **Dr. Diego Paez-Granados**
- **Prof. Robert Riener**

## Lab
This research was conducted at **[SCAI Lab](https://scai.ethz.ch/)** led by Dr. Diego Paez-Granados.  

## Funding
This study was partially funded by the Schweizer Paraplegiker Stiftung (SPS) and the ETH Zürich Foundation under the 2021-HS-348 ETH-SPS Digital Transformation in Personalized Healthcare for SCI individuals.





