import warnings

def warn(msg, ask=False) :
    warnings.warn(msg)
    if ask :
        try:
           input = raw_input
        except NameError:
           pass
        if not input('continue? (y/n)').lower() == 'y' :
            raise KeyboardInterrupt