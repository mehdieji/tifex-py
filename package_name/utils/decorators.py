import numpy as np
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
                if not isinstance(label, list) and not isinstance(label, np.ndarray):
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
