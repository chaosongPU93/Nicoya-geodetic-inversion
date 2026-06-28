import numpy as np

def ckm2LLd(xx, yy, lon0, lat0, rot):
    """
    Convert local Cartesian coordinates (xx, yy) in meters to geographic coordinates (lon, lat)
    using a rotated system defined by rot and referenced to origin (lon0, lat0).

    Parameters:
        xx, yy : array-like or scalar
            Local Cartesian coordinates in meters.
        lon0, lat0 : float
            Origin of the local coordinate system in degrees.
        rot : float
            Rotation angle in degrees. Positive is counterclockwise.

    Returns:
        lon, lat : ndarray
            Geographic coordinates in degrees.
    """

    # Ensure arrays
    xx = np.asarray(xx)
    yy = np.asarray(yy)

    if xx.shape != yy.shape:
        raise ValueError("xx and yy must have the same shape.")

    # WGS-84 parameters
    R = 6378137.0  # Equatorial radius in meters
    ff = 1 / 298.257  # Flattening factor

    # Radius at latitude
    r = R * (1 - ff * np.sin(np.radians(lat0))**2)

    mpd = r * np.pi / 180  # meters per degree

    # Rotation matrix components
    cos_rot = np.cos(np.radians(rot))
    sin_rot = np.sin(np.radians(rot))

    # Rotate the coordinate system
    x_rot = xx * cos_rot + yy * sin_rot
    y_rot = -xx * sin_rot + yy * cos_rot

    # Convert to lat/lon
    lat = lat0 + y_rot / mpd
    lon = lon0 + x_rot / (mpd * np.cos(np.radians(lat0)))

    return lon, lat