import tomllib
from pathlib import Path


def _read_toml(path: Path | str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_conf(path: Path | str, second_path: Path | str | None = None) -> dict:
    """Load a TOML configuration file and, optionally, combine it with a second one.

    The two files are combined section by section: for a section present in both,
    the keys are joined and `second_path` wins on repeated keys. Sections or
    top-level keys present in only one file are kept as they are.

    Parameters
    ----------
    path : Path | str
        Path to the main configuration file.
    second_path : Path | str | None, optional
        Path to the second configuration file, by default None.

    Returns
    -------
    dict
        Configuration.
    """
    conf = _read_toml(path)
    if second_path is None:
        return conf

    second = _read_toml(second_path)
    merged = conf | second
    for section in conf.keys() & second.keys():
        if isinstance(conf[section], dict) and isinstance(second[section], dict):
            merged[section] = conf[section] | second[section]
    return merged
