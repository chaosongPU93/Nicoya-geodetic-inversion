import numpy as np

def LL2ckmd(lon, lat, lon0, lat0, rot):
    """
    Transforms geographic coordinates (lon, lat) to local Cartesian (x, y) in meters,
    rotated by a given angle from the standard east-north system.

    Parameters:
        lon, lat : float or array-like
            Input longitude and latitude in degrees.
        lon0, lat0 : float
            Origin longitude and latitude in degrees.
        rot : float
            Rotation angle in degrees (CCW positive).

    Returns:
        x_rot, y_rot : ndarray
            Rotated Cartesian coordinates in meters.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape.")

    R = 6378137  # Earth's equatorial radius in meters
    ff = 1 / 298.257  # flattening factor
    r = R * (1 - ff * np.sin(np.radians(lat0))**2)  # radius at lat0
    mpd = r * np.pi / 180  # meters per degree at lat0

    cos_rot = np.cos(np.radians(rot))
    sin_rot = np.sin(np.radians(rot))

    # Convert (lon, lat) to local Cartesian coordinates
    yy = (lat - lat0) * mpd
    xx = (lon - lon0) * mpd * np.cos(np.radians(lat0))

    # Apply rotation
    x_rot = xx * cos_rot + yy * sin_rot
    y_rot = -xx * sin_rot + yy * cos_rot

    return x_rot, y_rot