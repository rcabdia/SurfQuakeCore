import functools
import sys
import warnings


def get_python_major_version() -> int:

    return int(sys.version.split(".")[1])


def deprecated(msg=""):
    """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""
    def decorator(func):

        @functools.wraps(func)
        def wrap(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(f"Call to deprecated function {func.__name__}. {msg}",
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return wrap
    return decorator
