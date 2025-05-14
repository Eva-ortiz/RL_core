# libraries imports
import datetime as dt
import logging
import sys
import time
import warnings
from io import StringIO
from math import floor, log10
from pathlib import Path

import pandas as pd
import tomli
from matplotlib import pyplot

# custom dtypes
Timestamp = dt.date | dt.datetime | pd.Timestamp


def find_exp(number) -> int:
    """Obtain the exponent from scientific notation.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation
    """
    base10 = log10(abs(number))
    return floor(base10)


def scientific_notation_label(number: float) -> str:
    """Return scientific notation label of input number.

    Parameters
    ----------
    number : float
        Number for which get scientific notation.

    Returns
    -------
    str
        String with scientific notation.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/21226868/superscript-in-python-plots
    .. [2] https://matplotlib.org/2.0.2/users/mathtext.html
    """
    exponent = find_exp(number)
    base = number / 10**exponent

    label = (
        f"$10^{int(exponent)}$" if base == 1 else rf"${base} \cdot 10^{int(exponent)}$"
    )  # [1], [2]
    return label


def timer(start: time, end: time, label: str = "Execution"):
    """Print elapsed time.

    Parameters
    ----------
    start : time
        Start time.
    end : time
        End time.
    label : str, optional
        Optional label for output message.
        By default, "Execution".

    References
    ----------
    .. [1] https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco

    Examples
    --------
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    output = main()
    end = time.time()

    timer(start, end)
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"------ {label} time: {{:0>2}}:{{:0>2}}:{{:05.2f}} ------".format(
            int(hours), int(minutes), seconds
        )
    )


def set_logging(level: str = "debug"):
    """Set root logger with specific logging level. Capture warnings by warnings module.

    Parameters
    ----------
    level : str, optional
        logging level, by default "debug"
    """

    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return f"{filename}:{lineno}: {category.__name__}:{message}"

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    warnings.formatwarning = warning_on_one_line
    logging.captureWarnings(True)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-4.4s] %(message)s", level=numeric_level
    )

    # A large number of DEBUG warnings appear on the log due to matplotlib
    # https://stackoverflow.com/questions/65728037/matplotlib-debug-turn-off-when-python-debug-is-on-to-debug-rest-of-program
    pyplot.set_loglevel(level="info")

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)


class Capturing(list):
    """Context manager for capturing stdout.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """

    def __enter__(self):
        """__enter__ method."""
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        """__exit__ method."""
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def pretty_series_print(series: pd.Series):
    """Print series in terminal."""
    for key, val in series.items():
        print(key, val)


def check_bool_or_int(value: bool | int):
    """Return input value as bool or int.

    Parameters
    ----------
    value : bool | int
        Value to check

    Returns
    -------
    bool_val : bool
        Input value if bool value, else False
    int_val : int
        Input value if int value, else 1
    """
    if isinstance(value, bool):
        bool_val = value
        int_val = 1
    elif isinstance(value, int):
        bool_val = False
        int_val = value
    else:
        raise NotImplementedError("Type of input value not implemented.")
    return bool_val, int_val


def check_bool_or_float(value: bool | float):
    """Return input value as bool or float.

    Parameters
    ----------
    value : bool | float
        Value to check

    Returns
    -------
    bool_val : bool
        Input value if bool value, else False
    float_val : float | None
        Input value if float value, else None
    """
    if isinstance(value, bool):
        bool_val = value
        float_val = None
    elif isinstance(value, float):
        bool_val = False
        float_val = value
    else:
        raise NotImplementedError("Type of input value not implemented.")
    return bool_val, float_val


# --------------- config utils --------------- By Komorebi AI Technologies
class Conf(dict):
    """Allow for keys not in the original dict, defaulting to None.

    Sub-class of dict that overrides `__getitem__` to allow for keys not in
    the original dict, defaulting to None.

    Author
    ------
    Komorebi AI Technologies
    """

    def __init__(self, *args, **kwargs):
        """Update dict with all keys from dict."""
        self.update(*args, **kwargs)
        # Parse the config (specifically, change "None" to None and "int" to int)
        self.update(parse_dict(self))

    def __getitem__(self, key):
        """Get key from dict. If not present, return None and raise warning.

        Parameters
        ----------
        key : Hashable
            key to get from original dict

        Returns
        -------
            original value in the dict or None if not present
        """
        if key not in self:
            warnings.warn(f"Key '{key}' not in conf. Defaulting to None", stacklevel=1)
            val = None
        else:
            val = dict.__getitem__(self, key)
        return val


def load_conf(path: str | Path, key: str = None) -> Conf:
    """Load TOML config as dict-like.

    Parameters
    ----------
    path : str
        Path to TOML config file
    key : str, optional
        Section of the conf file to load

    Returns
    -------
    Conf
        Config dictionary

    Author
    ------
    Komorebi AI Technologies
    """
    with open(path, "rb") as f:
        config = tomli.load(f)
    return Conf(config) if key is None else Conf(config[key])


def parse_str(x: str):
    """Parse a string value x.

    - If x is "none", returns None
    - If x is numeric, returns int(x) or float(x)
    - In any other case, returns the original x

    Author
    ------
    Komorebi AI Technologies
    """
    if not isinstance(x, str):  # Only True when value is not str
        return x
    elif x.lower() == "none":
        return None
    elif x.isnumeric():  # Only True when value is int
        return int(x)
    elif isfloat(x):  # Only True when value is float
        return float(x)
    else:
        return x


def parse_list(ser: list) -> list:
    """Parse the elements of a list.

    Author
    ------
    Komorebi AI Technologies
    """
    return [parse_str(x) for x in ser]


def parse_dict(d: dict) -> dict:
    """Parse of the elements of a dictionary.

    Author
    ------
    Komorebi AI Technologies
    """
    out_d = d.copy()
    for key, value in d.items():
        if isinstance(value, dict):
            out_d.update({key: parse_dict(value)})
        elif isinstance(value, list):
            out_d.update({key: parse_list(value)})
        else:
            out_d.update({key: parse_str(value)})
    return out_d


def isfloat(value):
    """Return True when the value can be converted to float.

    Author
    ------
    Komorebi AI Technologies
    """
    try:
        float(value)
        return True
    except ValueError:
        return False
