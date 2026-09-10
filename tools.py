import numpy as np


def list_to_num(arr):
    """
    Convert a binary list to an integer.

    Parameters
    ----------
    arr : list of int
        Binary list containing only 0 and 1.

    Returns
    -------
    int
        Integer represented by the binary list.
    """
    n = len(arr)

    # Automatically generate binary weights.
    # For example, n=5 -> [16, 8, 4, 2, 1]
    weights = [2 ** (n - 1 - i) for i in range(n)]

    return sum(a * w for a, w in zip(arr, weights))


def num_to_list(num, N):
    """
    Convert an integer to a binary list of length N.

    Parameters
    ----------
    num : int
        Integer to be converted.
    N : int
        Desired length of the binary list.

    Returns
    -------
    list of int
        Binary list of length N containing 0s and 1s.
    """
    if num < 0 or num >= 2 ** N:
        raise ValueError(
            f"The input integer must be between 0 and {2**N - 1}"
        )

    binary_list = [(num >> i) & 1 for i in range(N - 1, -1, -1)]

    return binary_list


def cell_area_latlon_grid(
    res_deg=0.25,
    lat_max=23.5,
    lon_min=-180.0,
    lon_max=180.0,
    R=6371000.0,
    unit="km2"
):
    """
    Calculate the actual area of each cell in a regular latitude-longitude
    grid within the tropical region (±lat_max).

    Parameters
    ----------
    res_deg : float, optional
        Spatial resolution of the grid in degrees. Default is 0.25°.
    lat_max : float, optional
        Maximum latitude of the study region. The grid extends from
        -lat_max to +lat_max. Default is 23.5°.
    lon_min : float, optional
        Minimum longitude of the grid. Default is -180°.
    lon_max : float, optional
        Maximum longitude of the grid. Default is 180°.
    R : float, optional
        Earth's radius in meters. Default is 6,371,000 m.
    unit : str, optional
        Output area unit. Options are "m2" and "km2".
        Default is "km2".

    Returns
    -------
    numpy.ndarray
        Two-dimensional array of grid-cell areas with shape (nlat, nlon).
    """

    # Latitude boundaries from north to south, including both endpoints.
    # The number of boundaries is nlat + 1.
    lat_edges = np.arange(
        lat_max,
        -lat_max - res_deg,
        -res_deg
    )

    # Longitude boundaries from west to east, including both endpoints.
    # The number of boundaries is nlon + 1.
    lon_edges = np.arange(
        lon_min,
        lon_max + res_deg,
        res_deg
    )

    nlat = len(lat_edges) - 1
    nlon = len(lon_edges) - 1

    # Grid-cell width in longitude, converted from degrees to radians.
    dlon = np.deg2rad(res_deg)

    # Northern and southern boundaries of each latitude band.
    lat_n = np.deg2rad(lat_edges[:-1])
    lat_s = np.deg2rad(lat_edges[1:])

    # Area factor for each latitude band.
    # This accounts for the variation in cell area with latitude.
    lat_factor = np.sin(lat_n) - np.sin(lat_s)

    # Expand the latitude-dependent area factor to a 2D grid.
    area = (
        (R**2)
        * dlon
        * lat_factor[:, None]
        * np.ones((1, nlon), dtype=float)
    )

    # Convert square meters to square kilometers if requested.
    if unit == "km2":
        area = area / 1e6

    return area