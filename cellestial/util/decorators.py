from lets_plot import (
    ggtb,
)

"""
This file is deprecated...
Preserved for reference...
"""

# ------------------------------ EXAMPLE STRUCTURE ------------------------------
def example(func):
    """Not a real decorator, just an example."""
    def wrapper(*args, **kwargs):
        # MUST DO: merge the default kwargs with the user-provided kwargs
        all_kwargs = func.__kwdefaults__
        all_kwargs.update(kwargs)
        # ------------------------------------------------------

        """ modify the output
        result = func(*args, **kwargs)

        # handle the case
        if all_kwargs.get("example"):
            result += something 
        else:
            pass

        return result
        """

    # MUST DO: inherit the default kwargs
    wrapper.__kwdefaults__ = func.__kwdefaults__  # inherit the default kwargs
    return wrapper


# ------------------------------ INTERACTIVE ------------------------------
def interactive(func):
    """Make the plot interactive."""
    def wrapper(*args, **kwargs):
        # merge the default kwargs with the user-provided kwargs
        all_kwargs = func.__kwdefaults__
        all_kwargs.update(kwargs)
        # ------------------------------------------------------

        # get the value of the `interactive` kwarg
        inter = all_kwargs.get("interactive")
        if inter is True:
            return func(*args, **kwargs) + ggtb()
        elif inter is False:
            return func(*args, **kwargs)
        else:
            msg = f"expected True or False for 'interactive' argument, but received {inter}"
            raise ValueError(msg)

    wrapper.__kwdefaults__ = func.__kwdefaults__  # inherit the default kwargs
    return wrapper
