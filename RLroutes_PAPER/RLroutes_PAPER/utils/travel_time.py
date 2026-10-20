import numpy as np
from pyproj import Geod

from RLroutes_PAPER.utils.drift import DAY_SECS

RADIUS_EARTH_M = 6371000
LAT_DEG_TO_M = 111120
N_PATH_POINTS = 100

# sphere of radius RADIUS_EARTH_M: great circles
SPHERE = Geod(a=RADIUS_EARTH_M, b=RADIUS_EARTH_M)


def great_circle_travel_time(
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    velocity: float,
) -> tuple[np.ndarray, float, float, bool]:
    """Compute the travel time of a boat at constant speed along the great circle.

    * If the great circle crosses the ±180 meridian, the time is the haversine
      distance divided by `velocity`.
    * Otherwise, the great circle is sampled at `N_PATH_POINTS` points, each point
      is projected to meters with ``x = lon * R * cos(lat)`` and
      ``y = lat * LAT_DEG_TO_M``, and the time is the length of the resulting
      polyline divided by `velocity`.

    Parameters
    ----------
    lat_start : float
        Latitude of the start of the route, in degrees.
    lon_start : float
        Longitude of the start of the route, in degrees within [-180, 180].
    lat_end : float
        Latitude of the end of the route, in degrees.
    lon_end : float
        Longitude of the end of the route, in degrees within [-180, 180].
    velocity : float
        Constant speed of the boat, in m/s. Must be higher than 0.

    Returns
    -------
    geodesic_path : np.ndarray
        Array of shape (N_PATH_POINTS, 2) with the [longitude, latitude] of the
        great circle, in degrees within [-180, 180].
    travel_days : float
        Travel time, in days. NaN if start and end are the same point.
    distance_m : float
        Great-circle distance, in meters.
    crosses_antimeridian : bool
        True if the great circle crosses the ±180 meridian.

    Warnings
    --------
    * `travel_days` is NaN when start and end are the same point; callers rely on
      it to detect a stay in the same position.
    """
    assert (
        180 >= lon_start >= -180 and 180 >= lon_end >= -180
    ), "Longitude range expected to be [-180, 180]."
    if velocity <= 0:
        raise ValueError("`velocity` must be higher than 0.")

    distance_m = compute_haversine_distance(
        lat1=lat_start, lon1=lon_start, lat2=lat_end, lon2=lon_end
    )
    crosses_antimeridian = abs(lon_end - lon_start) > 180

    geodesic_path = np.array(
        [(lon_start, lat_start)]
        + SPHERE.npts(lon_start, lat_start, lon_end, lat_end, N_PATH_POINTS - 2)
        + [(lon_end, lat_end)]
    )

    if distance_m == 0:
        travel_days = np.nan
    elif crosses_antimeridian:
        travel_days = distance_m / velocity / DAY_SECS
    else:
        lon, lat = geodesic_path[:, 0], geodesic_path[:, 1]
        x_m = lon * RADIUS_EARTH_M * np.cos(np.radians(lat)) * 2 * np.pi / 360
        y_m = lat * LAT_DEG_TO_M
        travel_days = np.hypot(np.diff(x_m), np.diff(y_m)).sum() / velocity / DAY_SECS

    return geodesic_path, travel_days, distance_m, crosses_antimeridian


def compute_haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance between pairs of points on the sphere `SPHERE`.

    Parameters
    ----------
    lat1, lon1 : np.ndarray
        Latitudes and longitudes of the first set of points, in degrees.
    lat2, lon2 : np.ndarray
        Latitudes and longitudes of the second set of points, in degrees.

    Returns
    -------
    np.ndarray
        Distance between each pair of points, in meters.
    """
    return SPHERE.inv(lon1, lat1, lon2, lat2)[2]
