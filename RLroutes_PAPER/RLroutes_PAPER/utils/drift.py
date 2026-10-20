import datetime as dt
import logging
import sys
import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager, suppress
from enum import StrEnum
from functools import partial
from io import StringIO
from pathlib import Path
from timeit import default_timer
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# custom dtypes
Timestamp = dt.date | dt.datetime | pd.Timestamp

DAY_SECS = 60 * 60 * 24
HOUR_SECS = 60 * 60
MINUTE_SECS = 60

EARTH_RADIUS = 6378.137  # [km]
LAT_DEGREE = 111.120  # [km]

MAX_LAT, MIN_LAT = 90, -90
MAX_LON, MIN_LON = 180, -180

KNOTS_2_KM_H = 1.852  # km/h is 1 knot
CM_S_2_KM_H = 0.036  # km/h is 1 cm/s
M_S_2_KM_H = 3.6  # km/h is 1 m/s

OCEAN_LIMITS = {
    "atlantic": {"lat_min": -60, "lat_max": 60, "lon_min": -70, "lon_max": 30},
    "indian": {"lat_min": -60, "lat_max": 60, "lon_min": 30, "lon_max": 140},
    "pacific": {
        "lat_min": -60,
        "lat_max": 60,
        "boundaries_1": {"lon_min": 140, "lon_max": 180},
        "boundaries_2": {"lon_min": -180, "lon_max": -70},
    },
}


class Coord(NamedTuple):
    """Represents a geographic coordinate."""

    lat: float
    lon: float


# custom errors
class PositionError(Exception):
    """Custom error for positions data."""

    pass


class Ocean(StrEnum):
    """Available oceans to locate our agent.

    Attributes
    ----------
    PACIFIC : str
        Pacific ocean.
    ATLANTIC : str
        Atlantic ocean.
    INDIAN : str
        Indian ocean.
    """

    PACIFIC = "pacific"
    ATLANTIC = "atlantic"
    INDIAN = "indian"


class Drift(StrEnum):
    """Drift style."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    DIRECTION = "direction"
    REAL_CURRENT = "real_current"


@contextmanager
def timer(tag=None):
    """Timer contextmanager."""
    start = default_timer()
    elapsed = None
    try:
        yield lambda: elapsed
    finally:
        end = default_timer()
        elapsed = end - start
        header = (
            "Elapsed time (D:H:M)" if tag is None else f"[{tag}] Elapsed time (D:H:M)"
        )
        days, hours, minutes = sec_2_day_hour_min(elapsed)
        print(f"{header}: {days}:{hours}:{minutes:.3f}")


def sec_2_day_hour_min(seconds: float) -> tuple[int, int, float]:
    """Input seconds to return days/hours/mins."""
    days = int(seconds // DAY_SECS)
    hours = int(seconds % DAY_SECS // HOUR_SECS)
    minutes = seconds % DAY_SECS % HOUR_SECS / MINUTE_SECS
    return days, hours, minutes


def lat_lon_to_km(coord_deg: Coord) -> Coord:
    """Transform degrees of lat and lon to kilometers in a coordinate point.

    Parameters
    ----------
    coord_deg : Coord
        `Coord` object which represents [latitude, longitude] point. Latitude
        and longitude given in degrees of latitude and longitude, respectively.

    Returns
    -------
    Coord
        `Coord` object which represents the [latitude, longitude] point.
        Latitude and longitude are given in kilometers (from the reference point
        of each coordinate).
    """
    lon_km = (
        coord_deg.lon * EARTH_RADIUS * np.cos(np.radians(coord_deg.lat)) * 2 * np.pi
    ) / 360
    lat_km = coord_deg.lat * LAT_DEGREE
    return Coord(lon=lon_km, lat=lat_km)


def km_to_lat_lon(coord_km: Coord) -> Coord:
    """Transform kilometers to lat and lon degrees in a coordinate point.

    Parameters
    ----------
    coord_km : Coord
        `Coord` object which represents a [latitude, longitude] point. Latitude
        and longitude are given in km (from reference point of each coordinate).

    Returns
    -------
    Coord
        `Coord` object which represents [latitude, longitude] point. Latitude
        and longitude given in degrees of latitude and longitude, respectively.
    """
    lat_deg = coord_km.lat / LAT_DEGREE
    lon_deg = (
        coord_km.lon * 360 / (EARTH_RADIUS * np.cos(np.radians(lat_deg)) * 2 * np.pi)
    )
    return Coord(lon=lon_deg, lat=lat_deg)


def lon_to_360(lon: pd.Series | xr.DataArray | np.ndarray):
    """Map longitudes from [-180, 180] to [0, 360)."""
    return lon % 360


def lon_to_180(lon: pd.Series | xr.DataArray | np.ndarray):
    """Map longitudes from [0, 360] to [-180, 180)."""
    return (lon + 180) % 360 - 180


def cartesian_to_polars(
    x: float | np.typing.ArrayLike, y: float | np.typing.ArrayLike
) -> tuple[float | np.typing.ArrayLike, float | np.typing.ArrayLike]:
    """Convert cartesian vector components to direction and modulus.

    Parameters
    ----------
    x : float | np.typing.ArrayLike
        "x" component of the vector.
    y : float | np.typing.ArrayLike
        "y" component of the vector.

    Returns
    -------
    dir : float | np.typing.ArrayLike
        Direction of the vector, in degrees within [0, 360), measured
        counterclockwise from the positive "x" axis.
    modu : float | np.typing.ArrayLike
        Modulus of the vector, in the units of `x` and `y`.
    """
    dir = np.degrees(np.arctan2(y, x)) % 360
    modu = np.hypot(x, y)
    return dir, modu


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


def filter_buoy_nd_date(
    df: pd.DataFrame,
    buoy_key: str | int,
    date_min: Timestamp,
    date_max: Timestamp,
    buoy_nd_date_cols: dict[str, str],
    columns: Iterable[str] | str | None = None,
    **kwargs: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Filter and order records by buoyKEY and date range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter.
    buoy_key : str | int
        Buoy KEY to filter. Type determined by type of buoy_key to filter in df.
    date_min : Timestamp
        Min date of the filtered data.
    date_max : Timestamp
        Max date of the filtered data.
    buoy_nd_date_cols : dict[str, str]
        Name of the columns storing buoyKEYs and dates, with "buoy_col" and
        "date_col" as keys.
    columns : Iterable[str] | str | None, optional
        Selected columns to return, by default None.
    kwargs : Callable[[pd.Series], pd.Series]
        Any additional condition, with keys as the column to filter and values
        as a lambda function of the condition.

    Return
    ------
    pd.DataFrame
        Filtered and orderder DataFrame by buoyKEY, date range with selected
        columns. Records with all NaN are excluded.

    Raises
    ------
    KeyError
        If indicated buoyKEY and date columns are not in data.
    KeyError
        If selected columns are not in data.
    """
    # ask for required columns
    if not set(buoy_nd_date_cols.values()) <= set(df.columns):
        raise KeyError("Data must contain indicated buoyKEY and date columns.")
    if columns is not None and not set(columns) <= set(df.columns):
        raise KeyError(f"Data must contain {columns} columns.")

    # filter by buoyKEY, date and additional kwargs and sort by buoy and date cols
    df = df.where(
        (df[buoy_nd_date_cols["buoy_col"]].str.lower() == str(buoy_key).lower())
        & (df[buoy_nd_date_cols["date_col"]] >= date_min)
        & (df[buoy_nd_date_cols["date_col"]] <= date_max)
    )
    if any(kwargs):
        for col, condition in kwargs.items():
            df = df[condition(df[col])]

    df = df.sort_values(
        [buoy_nd_date_cols["buoy_col"], buoy_nd_date_cols["date_col"]]
    ).dropna(axis=0, how="all")

    # filter selected columns
    return df if columns is None else df[columns]


def time_delta_between_records(
    df: pd.DataFrame, datetime_col: str, group_col: str | None = None
) -> tuple[pd.Timedelta, bool]:
    """Calculate timedelta between consecutive records and check if all are equal.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime records.
    datetime_col : str
        Name of the datetime column.
    group_col : str | None, optional
        Name of the column to group by (e.g., buoy KEY) before calculating deltas.
        If None, calculate deltas for the entire DataFrame, by default None.

    Returns
    -------
    timedelta : pd.Timedelta
        The timedelta between records. If `group_col` is provided, returns
        timedelta consistent across all groups.
    all_equal : bool
        True if all timedeltas are equal, False otherwise.

    Raises
    ------
    ValueError
        If timedeltas are not consistent across groups when `group_col` is provided.
    """
    if group_col is None:
        if df.shape[0] < 2:
            raise ValueError("DataFrame must have at least 2 records.")

        # Calculate deltas for entire DataFrame
        df_sorted = df.sort_values(datetime_col)
        series_delta_time = df_sorted[datetime_col].iloc[1:].reset_index(
            drop=True
        ) - df_sorted[datetime_col].iloc[:-1].reset_index(drop=True)

        delta_time = series_delta_time.iloc[0]
        all_equal = (series_delta_time == delta_time).all()
    else:
        # Calculate deltas for each group
        df_sorted = df.sort_values([group_col, datetime_col])
        delta_time = None
        all_equal = True

        for group_id in df_sorted[group_col].unique():
            group_data = df_sorted[df_sorted[group_col] == group_id]

            if group_data.shape[0] < 2:
                continue

            # Calculate delta time for this group
            series_delta_time = group_data[datetime_col].iloc[1:].reset_index(
                drop=True
            ) - group_data[datetime_col].iloc[:-1].reset_index(drop=True)

            # Check if all delta times within this group are the same
            group_all_equal = (series_delta_time == series_delta_time.iloc[0]).all()
            all_equal = all_equal and group_all_equal

            # Check consistency across groups
            if delta_time is None:
                delta_time = series_delta_time.iloc[0]
            elif delta_time != series_delta_time.iloc[0]:
                raise ValueError(
                    f"Inconsistent timedeltas across groups. "
                    f"Expected {delta_time}, got {series_delta_time.iloc[0]} "
                    f"for group {group_id}."
                )

        if delta_time is None:
            raise ValueError("No groups with at least 2 records found.")

    return delta_time, all_equal


def add_drift_from_uv(
    datetime: dt.datetime,
    df_pos: pd.DataFrame,
    ds_uv: xr.Dataset,
    delta_hours: float,
    no_drift_idxs: Iterable,
    dict_df_cols: dict[str, str],
):
    """Add drift to positions with uv currents data.

    Parameters
    ----------
        datetime: dt.datetime
            Datetime of positions and currents to work with.
        df_pos : pd.DataFrame
            Positions dataframe to add the drift. Drift will be sum to next record
            positions in km and later transformed into deg.
        ds_uv : xr.Dataset
            Currents dataset, with Eastward velocity (uo) and Northward velocity (vo)
            in m/s. NaN velocities already filled with 0
        delta_hours : float
            Number of hours of drift between records. It is recommended to just indicate
            timedelta between records.
        no_drift_idxs : Iterable
            Indexes of 'df_pos' to not apply drift.
        dict_df_cols : dict[str, str], optional
            Dictionary with input column / variable names. Expected:
                * lat_col_pos: `df_pos` latitude col, in degrees. By default "LATITUD".
                * lon_col_pos: `df_pos` longitude col, in degrees.
                    By default "LONGITUD".
                * lat_km_col_pos: `df_pos` latitude col, in km. By default "LATITUD_KM".
                * lon_km_col_pos: `df_pos` longitude col, in km.
                    By default "LONGITUD_KM".
                * dt_col_pos: `df_pos` datetime col. By default "FECHA_HORA".
                * buoykey_col_pos: `df_pos` buoy key col.
                    By default "KEY_BALIZA".
                * idx_col_pos: `df_pos` index col. By default "df_idx".
                * lat_col_uv: `ds_uv` latitude coord. By default "latitude".
                * lon_col_uv: `ds_uv` longitude coord. By default "longitude".
                * u_col_uv: `ds_uv` eastward velocity (uo) var, in m/s. By default "uo".
                * v_col_uv: `ds_uv` northward velocity (vo) var, in m/s.
                    By default "vo".
                * dt_col_uv: `ds_uv` time coord. By default "time".

    Returns
    -------
    df_pos : pd:DataFrame
        Updated 'df_pos' with added products during the function and drifted
        lat/lon/lat_km/lon_km to following 'datetime'.
    """
    logging.info(f"Computing {datetime}...")
    # get kwargs
    lat_col_pos = dict_df_cols.get("lat_col_pos", "LATITUD")
    lon_col_pos = dict_df_cols.get("lon_col_pos", "LONGITUD")
    lat_km_col_pos = dict_df_cols.get("lat_km_col_pos", "LATITUD_KM")
    lon_km_col_pos = dict_df_cols.get("lon_km_col_pos", "LONGITUD_KM")
    dt_col_pos = dict_df_cols.get("dt_col_pos", "FECHA_HORA")
    buoykey_col_pos = dict_df_cols.get("buoykey_col_pos", "KEY_BALIZA")
    idx_col_pos = dict_df_cols.get("idx_col_pos", "df_idx")

    lat_col_uv = dict_df_cols.get("lat_col_uv", "latitude")
    lon_col_uv = dict_df_cols.get("lon_col_uv", "longitude")
    u_col_uv = dict_df_cols.get("u_col_uv", "uo")
    v_col_uv = dict_df_cols.get("v_col_uv", "vo")
    dt_col_uv = dict_df_cols.get("dt_col_uv", "time")

    # make sure `df_pos` is properly ordered,
    # so we can safely iterate based on its indexes
    # DISCLAIMER: essential assert, this function use is heavily based on this
    # assumption for input dataframe
    assert df_pos.equals(
        df_pos.sort_values([buoykey_col_pos, dt_col_pos], axis=0, inplace=False)
    )

    df_pos_timedelta, consistent_timedelta = time_delta_between_records(
        df=df_pos, datetime_col=dt_col_pos, group_col=buoykey_col_pos
    )
    df_pos_delta_h = df_pos_timedelta.total_seconds() / HOUR_SECS
    if not consistent_timedelta:
        raise ValueError("Inconsistent timedeltas within `df_pos`.")
    elif df_pos_delta_h != delta_hours:
        warnings.warn(
            f"`delta_hours` differs from {df_pos_delta_h} timedelta between records.",
            stacklevel=1,
        )

    # select, for each buoy position, the currents of the nearest grid cell
    df_pos_slice = df_pos[df_pos[dt_col_pos] == datetime]
    ds_uv_slice = ds_uv.sel({dt_col_uv: pd.Timestamp(datetime).normalize()})
    nearest = ds_uv_slice.sel(
        {
            lat_col_uv: xr.DataArray(
                df_pos_slice[lat_col_pos].to_numpy(), dims="points"
            ),
            lon_col_uv: xr.DataArray(
                df_pos_slice[lon_col_pos].to_numpy(), dims="points"
            ),
        },
        method="nearest",
    ).compute()

    # Save retrieved copernicus lat-lon and velocities for this datetime.
    # DISCLAIMER: we work over `df_pos` original indexes `idx_col_pos`
    slice_idxs = df_pos_slice[idx_col_pos]
    df_pos.loc[
        slice_idxs,
        [
            f"{lat_col_uv}_copernicus",
            f"{lon_col_uv}_copernicus",
            f"{v_col_uv}_copernicus",
            f"{u_col_uv}_copernicus",
        ],
    ] = np.column_stack(
        [nearest[c].values for c in (lat_col_uv, lon_col_uv, v_col_uv, u_col_uv)]
    )
    # When saved, transform displacement to km, CAUTION WITH uo AND vo ORDER
    df_pos.loc[slice_idxs, [f"{v_col_uv}_displ_km", f"{u_col_uv}_displ_km"]] = (
        np.column_stack([nearest[v_col_uv].values, nearest[u_col_uv].values])
        * M_S_2_KM_H
        * delta_hours
    )  # m/s * km/h/(m/s) * h

    # obtain locations with retrieved displacement at each datetime
    # DISCLAIMER: we work over `df_pos` original indexes
    for df_idx in set(slice_idxs.values) - set(no_drift_idxs):
        # add buoy drift in km distance
        df_pos.loc[df_idx + 1, [lat_km_col_pos, lon_km_col_pos]] = (
            df_pos.loc[df_idx, [lat_km_col_pos, lon_km_col_pos]].values
            + df_pos.loc[
                df_idx, [f"{v_col_uv}_displ_km", f"{u_col_uv}_displ_km"]
            ].values
        )
        # check do not add a drift to final records of each buoy,
        # including last record
        assert (
            df_idx not in no_drift_idxs
        ), "Displacement added to selected not drift buoy."
        assert (
            df_pos.loc[df_idx, buoykey_col_pos]
            == df_pos.loc[df_idx + 1, buoykey_col_pos]
        ), "Displacement added to a different buoy."

        # overwrite next position in degrees
        next_lat_lon_deg = km_to_lat_lon(
            Coord(
                lat=df_pos.loc[df_idx + 1, lat_km_col_pos],
                lon=df_pos.loc[df_idx + 1, lon_km_col_pos],
            )
        )
        df_pos.loc[df_idx + 1, [lat_col_pos, lon_col_pos]] = (
            next_lat_lon_deg.lat,
            next_lat_lon_deg.lon,
        )

    return df_pos


def df_drift_buoys(
    df: pd.DataFrame,
    drift: Drift,
    no_drift_buoys_keys: list[int] | None = None,
    save_vel_plot_dir: Path | None = Path("./data"),
    dict_df_cols: dict[str, str] | None = None,
    currents_dir: Path = Path("./ds_uv.nc"),
    currents_file_per_day: bool = False,
    dict_uv_cols: dict[str, str] | None = None,
    ocean: Ocean | None = None,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Add buoys drifting to input STATIC latitudes and longitudes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to change lats/lons so drift is added to buoys.
    drift : Drift
        Drift modality:
            * Drift.UNIFORM : uniform random drift of +-50cm/s (~1 knot,
              ~1.80km/h).
            * Drift.GAUSSIAN : drift retrieved from gaussian distribution,
              centered in 0 and with sigma of 1 knots.
            * Drift.DIRECTION : gaussian distribution for random velocity,
              modulus centered in 50cm/s with sigma of 25cm/s. Consistent
              direction with gaussian change centered in 0.
            * Drift.REAL_CURRENT : drift of Copernicus currents [m/s]. Stored in
              `currents_dir`. Periodicity of one current map per day.
    no_drift_buoys_keys : list[int] | None, optional
        Keys of buoys to which drift should not be applied, by default None.
    save_vel_plot_dir : Path | None, optional
        Folder to save velocity distribution used for buoys drifting,
        by default Path("./data").
    dict_df_cols : dict[str,str] | None, optional
        Name of input dataframe columns. If None, `dict_df_cols` is set to:
        {"lat_col":"LATITUD", "lon_col":"LONGITUD",
        "buoyKEY_col":"KEY_BALIZA", "datetime_col":"FECHA_HORA"}.
    currents_dir: Path, optional
        [Only used if `Drift.REAL_CURRENT`] Path where to load ocean currents for drift.
        By default, Path("./ds_uv.nc").
    currents_file_per_day : bool, optional
        Indicate if currents are given in a file per day or in a single file.
        By default False, so all currents are assumed to be in a single file.
    dict_uv_cols : dict[str,str] | None, optional
        [Only used if `Drift.REAL_CURRENT`] Name of currents dataset variables.
        If None, `dict_uv_cols` is set to:
        {"lat_col": "latitude", "lon_col": "longitude", "v_col": "vo", "u_col": "uo"}.
    ocean : Ocean | None, optional
        [Only used if `Drift.REAL_CURRENT`] Ocean where to restrict currents,
        by default None.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with drift buoys lats/lons.
    ds : xr.Dataset | None
        Dataset with u/v currents. Only if `Drift.REAL_CURRENT`.

    Warnings
    --------
    * In input `df`, it is assumed datetime has been extended along all records.

    References
    ----------
    .. [1] https://essd.copernicus.org/articles/15/5135/2023/#:~:text=Free%2Ddrifting%20buoys%20tend%20to,as%20a%20mean%20drift%20velocity.
    """
    # retrieve col names
    dict_df_cols = dict_df_cols or {}
    lat_col = dict_df_cols.get("lat_col", "LATITUD")
    lon_col = dict_df_cols.get("lon_col", "LONGITUD")
    buoyKEY_col = dict_df_cols.get("buoyKEY_col", "KEY_BALIZA")
    datetime_col = dict_df_cols.get("datetime_col", "FECHA_HORA")

    # in general we consider a drift of 50cm/s (~1 knot, ~1.80km/h) [1]
    UNIFORM_VEL = 1 * KNOTS_2_KM_H  # [km/h]
    GAUSS_VEL_MEAN_ND_SIGMA = 0, 1  # knots
    DIRECTION_MIN_VEL, DIRECTION_MAX_VEL = 0, 4
    DIRECTION_VEL_MEAN_ND_SIGMA = 1, 0.5  # knots
    DIRECTION_VEL_DELTA_MEAN_ND_SIGMA = 0, 1e-3  # knots
    DIRECTION_DIR_DELTA_MEAN_ND_SIGMA = 0, 0.36  # degree

    ds = None  # initialize if not used

    # make sure df is ordered by buoy and datetime, crucial for following computations
    df = df.sort_values([buoyKEY_col, datetime_col])

    delta_time, all_delta_equal = time_delta_between_records(
        df=df, datetime_col=datetime_col, group_col=buoyKEY_col
    )
    assert all_delta_equal, "Inconsistent timedeltas between records."
    delta_hours = delta_time.total_seconds() / HOUR_SECS

    # ---------------- obtain buoy drift in km ----------------
    if drift in [Drift.UNIFORM, Drift.GAUSSIAN, Drift.DIRECTION]:
        if drift == Drift.UNIFORM:
            synthetic_vel_delta = UNIFORM_VEL  # [km/h]
            delta_km = synthetic_vel_delta * delta_hours  # km\h * h

            # generate random numbers to determine to which extend the buoy drifts at
            # each time step
            latlon_km_drift = np.random.uniform(
                low=-delta_km, high=delta_km, size=(df.shape[0], 2)
            )

            # determine the velocities to plot
            vel_to_plot = synthetic_vel_delta

        elif drift == Drift.GAUSSIAN:
            # represent the normal distribution to use for drifting
            mu, sigma = GAUSS_VEL_MEAN_ND_SIGMA  # knots

            # generate random numbers to determine to which extend the buoy drifts at
            # each time step
            random_knots_drift = np.random.normal(
                loc=mu, scale=sigma, size=(df.shape[0], 2)
            )  # knots
            latlon_km_drift = (
                random_knots_drift * KNOTS_2_KM_H * delta_hours
            )  # knots * (km\h)\knots * h

            # determine the velocities to plot
            vel_to_plot = random_knots_drift

        elif drift == Drift.DIRECTION:
            n_buoys = df[buoyKEY_col].unique().shape[0]
            n_datetimes_per_buoy = df.groupby(buoyKEY_col)[datetime_col].nunique()

            # check all buoys have the same number of time records
            assert (
                n_datetimes_per_buoy.iloc[0] == n_datetimes_per_buoy
            ).all(), "Not all buoys have same number of datetime records"

            # normal distribution to use for velocity
            mu, sigma = DIRECTION_VEL_MEAN_ND_SIGMA  # knots

            # normal distribution to use for velocity and direction variation
            mu_delta_vel, sigma_delta_vel = DIRECTION_VEL_DELTA_MEAN_ND_SIGMA  # knots
            mu_delta_deg, sigma_delta_deg = DIRECTION_DIR_DELTA_MEAN_ND_SIGMA  # degrees

            # generate random numbers to determine the initial drift velocity of each
            # buoy
            first_knots_drift = np.random.normal(
                loc=mu, scale=sigma, size=n_buoys
            ).clip(min=DIRECTION_MIN_VEL, max=DIRECTION_MAX_VEL)  # knots
            first_knots_drift_list = list(first_knots_drift)  # knots

            # generate random numbers to determine the initial preferred drift direction
            # of each buoy
            first_deg_drift = list(
                np.random.uniform(low=0, high=360, size=n_buoys)
            )  # degrees

            # iteratively generate variying velocity and direction, slightly
            # modifying the previous ones
            first_record_buoy_idxs = (
                df.reset_index().groupby(buoyKEY_col).first()["index"]
            )
            assert (
                first_record_buoy_idxs % n_datetimes_per_buoy == 0
            ).all(), (
                "If ordered, first record per buoy should be multiple of n datetimes."
            )

            assert (
                df.index == range(df.shape[0])
            ).all(), "df.index must be 0-based contiguous for numpy array indexing"

            random_knots_drift, random_deg_drift = (
                np.zeros(shape=df.shape[0]),
                np.zeros(shape=df.shape[0]),
            )
            for idx in df.index:
                if idx in first_record_buoy_idxs:
                    random_knots_drift[idx] = first_knots_drift_list.pop()
                    random_deg_drift[idx] = first_deg_drift.pop()
                else:
                    random_knots_drift[idx] = (
                        random_knots_drift[idx - 1]
                        + np.random.normal(loc=mu_delta_vel, scale=sigma_delta_vel)
                    ).clip(min=DIRECTION_MIN_VEL, max=DIRECTION_MAX_VEL)
                    random_deg_drift[idx] = random_deg_drift[
                        idx - 1
                    ] + np.random.normal(loc=mu_delta_deg, scale=sigma_delta_deg)
            assert (
                not first_knots_drift_list and not first_deg_drift
            ), "Incoherent number of buoys"

            # transform from modulus and angle to lat lon delta in kilometers
            latlon_km_drift = (
                np.array(
                    [
                        random_knots_drift * np.sin(np.radians(random_deg_drift)),
                        random_knots_drift * np.cos(np.radians(random_deg_drift)),
                    ]
                ).T  # knots
                * KNOTS_2_KM_H  # (km\h)\knots
                * delta_hours  # h
            )

            # determine the velocities to plot
            vel_to_plot = first_knots_drift

        # ---------------- transform lat/lon coords to km ----------------
        df_latlon_km = list(
            map(
                lat_lon_to_km,
                [Coord(lat=lat, lon=lon) for lat, lon in df[[lat_col, lon_col]].values],
            )
        )

        # records must be ordered by buoy and time so the following logic can be
        # performed
        assert df.equals(
            df.sort_values([buoyKEY_col, datetime_col], axis=0, inplace=False)
        )

        no_drift_buoys_idx = df[df[buoyKEY_col].isin(no_drift_buoys_keys or [])].index
        for idx, km_drift in enumerate(latlon_km_drift):
            # do not add a drift to the first record of each buoy or selected buoys
            if (
                idx == 0
                or (df.loc[idx - 1, buoyKEY_col] != df.loc[idx, buoyKEY_col])
                or idx in no_drift_buoys_idx
            ):
                continue
            # add buoy drift in km
            df_latlon_km[idx] = df_latlon_km[idx - 1] + km_drift

        # ---------------- transform km coords to lat/lon ----------------

        # retrieve coords in deg once drift has been applied
        df_latlon_deg = list(
            map(
                km_to_lat_lon,
                [Coord(lat=lat, lon=lon) for lat, lon in df_latlon_km],
            )
        )
        # overwrite lat and lon data with drifted values
        df[lat_col] = [coord.lat for coord in df_latlon_deg]
        df[lon_col] = [coord.lon for coord in df_latlon_deg]

    elif drift == Drift.REAL_CURRENT:
        dict_uv_cols = dict_uv_cols or {}
        uv_lat_col = dict_uv_cols.get("lat_col", "latitude")
        uv_lon_col = dict_uv_cols.get("lon_col", "longitude")
        uv_v_col = dict_uv_cols.get("v_col", "vo")
        uv_u_col = dict_uv_cols.get("u_col", "uo")

        # ------------- retrieve ocean currents for selected dates -------------
        # slice of data to retrieve
        def _preprocess(
            ds: xr.Dataset, lon_bnds: tuple[float, float], lat_bnds: tuple[float, float]
        ):
            return ds.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))

        # retrieve dates
        df["FECHA"] = df[datetime_col].dt.date
        df_dates = df["FECHA"].unique()
        df_datetimes = df[datetime_col].unique()

        # retrieve ocean currents all at once or date by date
        ds_routes = (
            currents_dir
            if not currents_file_per_day
            else [currents_dir / f"{date}.nc" for date in pd.Series(df_dates)]
        )

        # ...and lat-lon slices
        if ocean is not None:
            lat_bnds = (OCEAN_LIMITS[ocean]["lat_min"], OCEAN_LIMITS[ocean]["lat_max"])
        # pacific ocean has two slices as it has -180 - 180 transition
        if ocean == Ocean.PACIFIC:
            ds_list = []
            for bounds in ["boundaries_1", "boundaries_2"]:
                lon_bnds = (
                    OCEAN_LIMITS[ocean][bounds]["lon_min"],
                    OCEAN_LIMITS[ocean][bounds]["lon_max"],
                )
                partial_func = partial(
                    _preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds
                )
                ds_list.append(xr.open_mfdataset(ds_routes, preprocess=partial_func))
            # concat values, same latitudes but additional longitudes
            ds = xr.concat(ds_list, dim=uv_lon_col, join="exact")

            # displace longitudes from [-180, 180] to [0, 360] for pacific continuity.
            # sortby is required: concat leaves lon as 140..180, -180..-70
            # (not monotonic)
            assert (df[lon_col].max() <= 180) and (df[lon_col].min() >= -180)
            assert (ds[uv_lon_col].max() <= 180) and (ds[uv_lon_col].min() >= -180)

            df[lon_col] = lon_to_360(df[lon_col])
            ds_uv = (
                ds.assign_coords({uv_lon_col: lon_to_360(ds[uv_lon_col])})
                .sortby(uv_lon_col)
                .drop_duplicates(uv_lon_col)  # 180 and -180 would collapse to 180
            )

            assert (df[lon_col].max() <= 360) and (df[lon_col].min() >= 0)
            assert (ds_uv[uv_lon_col].max() <= 360) and (ds_uv[uv_lon_col].min() >= 0)

        elif ocean in [Ocean.ATLANTIC, Ocean.INDIAN]:
            lon_bnds = (OCEAN_LIMITS[ocean]["lon_min"], OCEAN_LIMITS[ocean]["lon_max"])
            partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
            ds = xr.open_mfdataset(ds_routes, preprocess=partial_func)
        elif ocean is None:
            ds = xr.open_mfdataset(ds_routes)

        # ------------- prepare initial data -------------
        ds_uv = ds_uv if ocean == Ocean.PACIFIC else ds
        ds_uv = ds_uv[[uv_u_col, uv_v_col]]
        # keep only the first (surface) depth level
        if "depth" in ds_uv.dims:
            ds_uv = ds_uv.isel(depth=0)
        # fill nan in velocity with zeros
        ds_uv = ds_uv.fillna(0).load()

        # obtain lat lon positions in km to later add retrieved velocities
        df[[f"{lat_col}_KM", f"{lon_col}_KM"]] = list(
            map(
                lat_lon_to_km,
                [Coord(lat=lat, lon=lon) for lat, lon in df[[lat_col, lon_col]].values],
            )
        )
        # store dataframe index in column for later calculus
        df = df.reset_index().rename(columns={"index": "df_idx"})

        # add a new columns to store following product, such as copernicus lat, lon and
        # uo, vo displacements in km
        df[[f"{uv_u_col}_displ_km", f"{uv_v_col}_displ_km"]] = pd.DataFrame(
            np.zeros(shape=(df.shape[0], 2))
        )

        # store df indexes of indicated no drift buoys
        no_drift_buoys_idx = df[df[buoyKEY_col].isin(no_drift_buoys_keys or [])].index

        # ------------- add time and position dependen drift -------------
        # Assign Eastward velocity (uo) and Northward velocity (vo) in m/s
        # (Copernicus data) of the nearest grid cell to each lat, lon and datetime.
        # With df_datetimes[:-1], avoid use velocity of last record for the
        # (inexistent) next record.
        drift_from_uv_cols = {
            "lat_col_pos": lat_col,
            "lon_col_pos": lon_col,
            "lat_km_col_pos": f"{lat_col}_KM",
            "lon_km_col_pos": f"{lon_col}_KM",
            "dt_col_pos": datetime_col,
            "buoykey_col_pos": buoyKEY_col,
            "idx_col_pos": "df_idx",
            "lat_col_uv": uv_lat_col,
            "lon_col_uv": uv_lon_col,
            "v_col_uv": uv_v_col,
            "u_col_uv": uv_u_col,
            "dt_col_uv": "time",
        }
        for datetime in df_datetimes[:-1]:
            df = add_drift_from_uv(
                datetime=datetime,
                df_pos=df,
                ds_uv=ds_uv,
                delta_hours=delta_hours,
                no_drift_idxs=no_drift_buoys_idx,
                dict_df_cols=drift_from_uv_cols,
            )

        # ------------- final checks and storgae of product -------------
        # check all lat/lons are closer than the Copernicus grid size to their
        # assigned Copernicus cell
        delta_grid_size = 1e-5
        gap_farther = abs(
            df[[lat_col, lon_col]].values
            - df[[f"{uv_lat_col}_copernicus", f"{uv_lon_col}_copernicus"]].values
        ) > (1 / 12 + delta_grid_size)  # Copernicus grid size

        # reduce 2D boolean (n_rows x 2) to 1D (True if any col is True)
        rows_farther = gap_farther.any(axis=1)
        cop_coords = [
            f"{uv_lat_col}_copernicus",
            f"{uv_lon_col}_copernicus",
        ]

        assert ~rows_farther.any(), (
            f"Merge of data/Copernicus went wrong. "
            f"{rows_farther.sum()} lat/lon are farther than "
            f"maximum distance in Copernicus grid:\n"
            f"Lat/lon:\n{df.loc[rows_farther, [lat_col, lon_col]]}"
            f"\nCopernicus lat/lon:\n"
            f"{df.loc[rows_farther, cop_coords]}"
        )

        # determine the velocities to plot (in m/s)
        vel_to_plot = df[
            [f"{uv_v_col}_copernicus", f"{uv_u_col}_copernicus"]
        ].to_numpy()

        if ocean == Ocean.PACIFIC:
            # displace from [0,360] to [-180, 180]
            df[lon_col] = lon_to_180(df[lon_col])
            df[f"{uv_lon_col}_copernicus"] = lon_to_180(df[f"{uv_lon_col}_copernicus"])

            # check longitudes are again between [-180, 180]
            assert (df[lon_col].max() <= 180) and (df[lon_col].min() >= -180)
            assert (df[f"{uv_lon_col}_copernicus"].max() <= 180) and (
                df[f"{uv_lon_col}_copernicus"].min() >= -180
            )

    # -------------- plot velocities --------------
    if save_vel_plot_dir is not None:
        # plot obtained velocities distribution (check min/max is what we desire)
        plt.figure()
        plt.title("Distribution of velocities")
        _, bins, _ = plt.hist(vel_to_plot.flatten(), 50, density=True)
        with suppress(UnboundLocalError):
            plt.plot(
                bins,
                1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-((bins - mu) ** 2) / (2 * sigma**2)),
                linewidth=2,
                color="r",
            )
        plt.savefig(save_vel_plot_dir / "velocities_distribution.png")

    # -------------- check coords in deg once drift has been applied --------------
    assert df[lat_col].max() <= MAX_LAT
    assert df[lat_col].min() >= MIN_LAT
    assert df[lon_col].max() <= MAX_LON
    assert df[lon_col].min() >= MIN_LON

    return df, ds
