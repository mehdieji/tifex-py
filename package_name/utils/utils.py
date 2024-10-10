import package_name.feature_extraction as fe

# Description: Utility functions for the package.

class SignalFeatures():
    def __init__(self, label, features) -> None:
        self.label = label
        self.features = features

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
            modules.append(fe.statistical_feature_calculators)
        elif name=="spectral":
            modules.append(fe.spectral_features_calculators)
        elif name=="time_frequency":
            modules.append(fe.time_frequency_feature_calculators)
    return modules

def calculate_time_freq_features(series, params):
    features = {}
    param_dict = params.get_settings_as_dict()
    calculators = get_calculators(get_modules(["time_frequency"]))
    for calculate in calculators:
        feature = calculate(**series, **param_dict)
        name = getattr(calculate, "names")

        if isinstance(name, list):
            for n, f in zip(name, feature):
                features[n] = f
        else:
            features[name] = feature

    return [series["label"]], [features]

def extract_features(series, modules, params):
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
    param_dict = params.get_settings_as_dict()
    calculators = get_calculators(get_modules(modules))
    for calculate in calculators:
        feature = calculate(**series, **param_dict)
        name = getattr(calculate, "names")

        if isinstance(feature, SignalFeatures):
            for k, v in feature.features.items():
                features[f'{name}_{k}'] = v
        else:
            if isinstance(name, list):
                for n, f in zip(name, feature):
                    features[n] = f
            else:
                features[name] = feature

    if "label" in series:
        label = series["label"]
    else:
        label = ""

    return SignalFeatures(label, features)

