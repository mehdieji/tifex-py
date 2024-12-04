import numpy as np
# Description: Utility functions for the package.

def name(strname, argname=None, offset=0):
    """
    Decorator to add names to a function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            names = []
            if isinstance(argname, list):
                i = 0
                for param in argname:
                    if isinstance(param, str):
                        label = kwargs[param]
                        if not isinstance(label, list) and not isinstance(label, np.ndarray):
                            label = list(range(label+offset))
                        for p in label:
                            names.append(strname[i].format(p))
                    else:
                        names.append(strname[i].format(param))
                    i += 1
            elif argname and argname in kwargs:
                label = kwargs[argname]
                if not isinstance(label, list) and not isinstance(label, np.ndarray):
                    label = list(range(label+offset))
                for param in label:
                    names.append(strname.format(param))
            elif type(strname) is list:
                    for n in strname:
                        names.append(n)
            else:
                names = strname

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
