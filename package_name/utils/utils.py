# Think if theres a smarter place to put this stuff
def name(strname, argname=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            names = []
            print(strname)
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
            # print(names)
            wrapper.names = names
            return func(*args, **kwargs)
        # wrapper.names = func.names
        return wrapper
    return decorator

def exclude():
    def wrapper(f):
        setattr(f, 'exclude', True)
        return f
    return wrapper
