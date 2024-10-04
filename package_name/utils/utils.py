import package_name.feature_extraction.statistical_feature_calculators as statistical_feature_calculators
import package_name.feature_extraction.spectral_features_calculators as spectral_features_calculators

# Description: Utility functions for the package.

def name(strname, argname=None):
    """
    Decorator to add names to a function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            names = []
            if argname:
                label = kwargs[argname]
                if not isinstance(label, list):
                    label = list(range(label))
                for param in label:
                    names.append(strname.format(param))
            else:
                if type(strname) is list:
                    for n in strname:
                        names.append(n)
                else:
                    names.append(strname)
            if len(names) == 1:
                names = names[0]
            wrapper.names = names
            return func(*args, **kwargs)
        return wrapper
    return decorator

def exclude():
    """
    Decorator to exclude a function from being included in a list of functions.
    """
    def wrapper(f):
        setattr(f, 'exclude', True)
        return f
    return wrapper

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
