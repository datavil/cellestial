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

