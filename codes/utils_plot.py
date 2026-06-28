import pandas as pd
import numpy as np
import re
import pygmt

# Formula from https://mathworld.wolfram.com/AzimuthalEquidistantProjection.html
def azi_equidist_proj(lon, lat, lon0, lat0, R=6371.0):
    """
    Convert longitude and latitude to x, y coordinates (in km) using 
    the Azimuthal Equidistant Projection with a fixed center.
    
    Parameters:
    - lon, lat : arrays or scalars of longitude and latitude (in degrees)
    - lon0, lat0 : center of projection (in degrees)
    - R : Earth's radius (default: 6371 km)

    Returns:
    - x, y : Cartesian coordinates in km
    """
    # Convert degrees to radians
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    lon0_rad, lat0_rad = np.radians(lon0), np.radians(lat0)

    # Compute angular distance c
    cos_c = np.sin(lat0_rad) * np.sin(lat_rad) + np.cos(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad)
    c = np.arccos(np.clip(cos_c, -1, 1))  # Clip to handle numerical precision errors

    # Compute projected coordinates
    # Handle singularity when c is near 0 (point near projection center)
    # Using Taylor series: lim(c->0) c/sin(c) = 1
    sin_c = np.sin(c)
    with np.errstate(divide='ignore', invalid='ignore'):
        kprime = np.where(np.abs(sin_c) > 1e-10, c / sin_c, 1.0)

    x = R * kprime * (np.cos(lat_rad) * np.sin(lon_rad - lon0_rad))
    y = R * kprime * (np.cos(lat0_rad) * np.sin(lat_rad) - np.sin(lat0_rad) * np.cos(lat_rad) * np.cos(lon_rad - lon0_rad))

    return x, y


# Formula from https://mathworld.wolfram.com/AzimuthalEquidistantProjection.html
def inverse_azi_equidist_proj(x, y, lon0, lat0, R=6371.0):
    """
    Convert x, y coordinates (in km) back to longitude and latitude using 
    the inverse Azimuthal Equidistant Projection.

    Parameters:
    - x, y : arrays or scalars of Cartesian coordinates (in km)
    - lon0, lat0 : center of projection (in degrees)
    - R : Earth's radius (default: 6371 km)

    Returns:
    - lon, lat : Longitude and Latitude (in degrees)
    """

    # if x == 0 and y == 0:
    #     lon, lat = lon0, lat0
    
    # else:    
    #     # Convert center lon/lat to radians
    #     lon0_rad, lat0_rad = np.radians(lon0), np.radians(lat0)

    #     # Compute angular distance c
    #     c = np.sqrt(x**2 + y**2) / R
        
    #     # Compute latitude
    #     lat_rad = np.arcsin(np.cos(c) * np.sin(lat0_rad) + y * np.sin(c) * np.cos(lat0_rad) / c / R)
    #     print("lat_rad:", lat_rad)

    #     # Compute longitude
    #     if lat0 == 90:
    #         lon_rad = lon0_rad + np.arctan2(-x, y)
    #     elif lat0 == -90:
    #         lon_rad = lon0_rad + np.arctan2(x, y)
    #     else:
    #         lon_rad = lon0_rad + np.arctan2(x * np.sin(c) / R, 
    #                                         c * np.cos(lat0_rad) * np.cos(c) - y * np.sin(lat0_rad) * np.sin(c) / R)

    #     # Convert radians to degrees
    #     lon, lat = np.degrees(lon_rad), np.degrees(lat_rad)

    # return lon, lat


    # Detect original types
    is_scalar = np.isscalar(x) and np.isscalar(y)
    is_series = isinstance(x, pd.Series) or isinstance(y, pd.Series)
    
    # Convert to NumPy arrays for computation
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    # Prepare output arrays
    lon = np.empty_like(x_arr, dtype=float)
    lat = np.empty_like(y_arr, dtype=float)
    
    # Mask for origin points
    mask_origin = (x_arr == 0) & (y_arr == 0)
    
    # Handle origin points directly
    lon[mask_origin] = lon0
    lat[mask_origin] = lat0
    
    # For non-origin points
    mask_nonorigin = ~mask_origin
    if np.any(mask_nonorigin):
        x_ = x_arr[mask_nonorigin]
        y_ = y_arr[mask_nonorigin]

        lon0_rad = np.radians(lon0)
        lat0_rad = np.radians(lat0)
        c = np.sqrt(x_**2 + y_**2) / R

        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat0_rad) +
            y_ * np.sin(c) * np.cos(lat0_rad) / c / R
        )

        # Longitude calculation depending on pole cases
        if lat0 == 90:
            lon_rad = lon0_rad + np.arctan2(-x_, y_)
        elif lat0 == -90:
            lon_rad = lon0_rad + np.arctan2(x_, y_)
        else:
            lon_rad = lon0_rad + np.arctan2(
                x_ * np.sin(c) / R,
                c * np.cos(lat0_rad) * np.cos(c) -
                y_ * np.sin(lat0_rad) * np.sin(c) / R
            )

        lon[mask_nonorigin] = np.degrees(lon_rad)
        lat[mask_nonorigin] = np.degrees(lat_rad)
    
    # Convert back to original type
    if is_scalar:
        return float(lon), float(lat)
    elif is_series:
        return pd.Series(lon, index=x.index if isinstance(x, pd.Series) else y.index), \
               pd.Series(lat, index=x.index if isinstance(x, pd.Series) else y.index)
    else:
        return lon, lat
    

def parse_trench_file(filename, lon_range=None, lat_range=None):
    """
    Parse a trench data file and optionally filter by geographic bounds.
    
    Args:
        filename (str): Path to the trench data file
        lon_range (tuple): (min_lon, max_lon) to filter longitude, optional
        lat_range (tuple): (min_lat, max_lat) to filter latitude, optional
    
    Returns:
        list: List of dictionaries, each containing segment info and coordinates
    """
    segments = []
    current_segment = None
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this is a segment header line (starts with >)
            if line.startswith('>'):
                # Save previous segment if it exists and has coordinates
                if current_segment and current_segment['coordinates']:
                    segments.append(current_segment)
                
                # Start new segment
                # Extract segment ID and name
                match = re.match(r'>\s*(\d+)\s+(.*)', line)
                if match:
                    segment_id = match.group(1)
                    segment_name = match.group(2).strip()
                else:
                    segment_id = "unknown"
                    segment_name = line[1:].strip()
                
                current_segment = {
                    'segment_id': segment_id,
                    'segment_name': segment_name,
                    'coordinates': []
                }
                continue
            
            # Try to parse coordinate line
            if current_segment is not None:
                try:
                    coords = line.split()
                    if len(coords) >= 2:
                        longitude = float(coords[0])
                        latitude = float(coords[1])
                        
                        # Apply geographic filtering if specified
                        if lon_range and not (lon_range[0] <= longitude <= lon_range[1]):
                            continue
                        if lat_range and not (lat_range[0] <= latitude <= lat_range[1]):
                            continue
                        
                        current_segment['coordinates'].append({
                            'lon': longitude,
                            'lat': latitude
                        })
                except (ValueError, IndexError):
                    # Skip lines that can't be parsed as coordinates
                    continue
    
    # Don't forget the last segment
    if current_segment and current_segment['coordinates']:
        segments.append(current_segment)
    
    return segments

def segments_to_dataframe(segments):
    """
    Convert segments list to a pandas DataFrame.
    
    Args:
        segments (list): List of segment dictionaries from parse functions
    
    Returns:
        pandas.DataFrame: DataFrame with columns for segment info and coordinates
    """
    data = []
    for segment in segments:
        for coord in segment['coordinates']:
            data.append({
                'segment_id': segment['segment_id'],
                'segment_name': segment['segment_name'],
                'lon': coord['lon'],
                'lat': coord['lat']
            })
    
    return pd.DataFrame(data)

def filter_segments_by_bounds(segments, lon_range=None, lat_range=None):
    """
    Filter existing segments by geographic bounds.
    
    Args:
        segments (list): List of segment dictionaries
        lon_range (tuple): (min_lon, max_lon) to filter longitude
        lat_range (tuple): (min_lat, max_lat) to filter latitude
    
    Returns:
        list: Filtered segments
    """
    filtered_segments = []
    
    for segment in segments:
        filtered_coords = []
        for coord in segment['coordinates']:
            lon = coord['lon']
            lat = coord['lat']
            
            # Check if coordinate is within bounds
            if lon_range and not (lon_range[0] <= lon <= lon_range[1]):
                continue
            if lat_range and not (lat_range[0] <= lat <= lat_range[1]):
                continue
            
            filtered_coords.append(coord)
        
        # Only include segment if it has coordinates within bounds
        if filtered_coords:
            filtered_segment = segment.copy()
            filtered_segment['coordinates'] = filtered_coords
            filtered_segments.append(filtered_segment)
    
    return filtered_segments


def parse_plate_interface_file(filename):
    """
    Parse a text file with plate interface data segmented by depth contour lines.
    
    Args:
        filename (str): Path to the input file
    
    Returns:
        pandas.DataFrame: DataFrame with columns ['longitude', 'latitude', 'depth']
    """
    data = []
    current_depth = None
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this is a contour line (depth marker)
            if line.startswith('>') and 'contour' in line and '-Z' in line:
                # Extract depth value using regex
                # Pattern matches "-Z" followed by digits (with optional decimal)
                depth_match = re.search(r'-Z(-?\d+(?:\.\d+)?)', line)
                if depth_match:
                    current_depth = float(depth_match.group(1))
                continue
            
            # If we have a current depth and the line contains coordinate data
            if current_depth is not None:
                # Try to parse as longitude, latitude coordinates
                try:
                    # Split by whitespace and convert to floats
                    coords = line.split()
                    if len(coords) >= 2:
                        longitude = float(coords[0])
                        latitude = float(coords[1])
                        data.append({
                            'lon': longitude,
                            'lat': latitude,
                            'dep': current_depth
                        })
                except (ValueError, IndexError):
                    # Skip lines that can't be parsed as coordinates
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by absolute depth from min to max
    df = df.sort_values('dep', key=lambda x: abs(x))
    
    # Reset index after sorting
    df = df.reset_index(drop=True)

    return df


def project_vector_2d_matrix(E, N, azimuth):
    """
    Alternative implementation using rotation matrix approach
    
    Parameters:
    -----------
    E : float or array-like
        East component of the vector
    N : float or array-like
        North component of the vector
    azimuth : float
        Azimuth angle in degrees
    
    Returns:
    --------
    normal_comp : float or array
        Component normal to the azimuth direction
    parallel_comp : float or array
        Component parallel to the azimuth direction
    """
    # Convert inputs to numpy arrays
    E = np.asarray(E)
    N = np.asarray(N)
    
    # Convert azimuth to radians
    theta = np.radians(azimuth)
    
    # Rotation matrix to align with azimuth direction
    # This rotates the coordinate system so that the azimuth direction
    # becomes the new x-axis (parallel direction)
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    
    # Stack N and E components
    vectors = np.vstack([E.flatten(), N.flatten()])
    
    # Apply rotation
    rotated = R @ vectors
    
    # Extract parallel and normal components
    normal_comp = rotated[0].reshape(N.shape)
    parallel_comp = rotated[1].reshape(N.shape)
    
    return normal_comp, parallel_comp


def rmse_3d_dataframe(predicted_df, true_df):
    """
    Per-component (scalar) RMSE: sqrt( sum(rx^2+ry^2+rz^2) / (3N) ).
    Treats each scalar component as an independent observation.
    Equals 1/sqrt(3) of the per-vector RMSE.

    predicted_df: DataFrame with columns ['ux', 'uy', 'uz'] for predicted vectors
    true_df: DataFrame with columns ['ux', 'uy', 'uz'] for true vectors
    """
    predicted = predicted_df[['ux', 'uy', 'uz']].values
    true = true_df[['ux', 'uy', 'uz']].values

    return np.sqrt(np.mean((predicted - true)**2))


def rmse_3d_vec_dataframe(predicted_df, true_df):
    """
    Per-vector RMSE: sqrt( sum(|r_i|^2) / N ), units same as input vector.
    "Typical residual vector magnitude per station" — the convention more
    commonly used in geodesy/GPS for 3D position/displacement misfit.

    predicted_df: DataFrame with columns ['ux', 'uy', 'uz'] for predicted vectors
    true_df: DataFrame with columns ['ux', 'uy', 'uz'] for true vectors
    """
    predicted = predicted_df[['ux', 'uy', 'uz']].values
    true = true_df[['ux', 'uy', 'uz']].values
    diff = predicted - true                       # shape (N, 3)
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


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



from scipy.stats import pearsonr

class SlipRecoveryMetrics:
    def __init__(self, mtrue_s, m_s, loc_f):
        """
        mtrue_s, m_s: pandas DataFrames with slip cols (e.g. ['s_dip'])
        loc_f: pandas DataFrame with fault node geometry (must include 'zf')
        """
        self.mtrue_s = mtrue_s
        self.m_s = m_s
        self.loc_f = loc_f

    # -------------------------
    # RMS Error
    # -------------------------
    def rmse(self, mask, col='s_dip'):
        true_vals = self.mtrue_s.loc[mask, col].values
        inv_vals  = self.m_s.loc[mask, col].values

        rmse = np.sqrt(np.mean((true_vals - inv_vals)**2))

        return rmse
    
    # -------------------------
    # Normalized RMS Error
    # -------------------------
    def nrmse(self, mask, col='s_dip'):
        true_vals = self.mtrue_s.loc[mask, col].values
        inv_vals  = self.m_s.loc[mask, col].values

        numerator = np.sqrt(np.sum((true_vals - inv_vals)**2))
        denominator = np.sqrt(np.sum(true_vals**2))

        return numerator / denominator if denominator > 0 else np.nan

    # -------------------------
    # relative L2 norm error
    # -------------------------
    def rel_l2_error(self, mask, col='s_dip'):
        true_vals = self.mtrue_s.loc[mask, col].to_numpy()
        inv_vals  = self.m_s.loc[mask, col].to_numpy()

        num = np.linalg.norm(inv_vals - true_vals)
        den = np.linalg.norm(true_vals)

        return num / den if den > 0 else np.nan
    
    # -------------------------
    # Stripe Detection Rate
    # -------------------------
    def detection_rate(self, mask, col='s_dip', threshold=0.5, min_fraction=0.5):
        true_vals = self.mtrue_s.loc[mask, col].values
        inv_vals  = self.m_s.loc[mask, col].values
        # print(inv_vals.shape, true_vals.shape)

        detected_nodes = inv_vals >= threshold * true_vals
        # print(detected_nodes.shape, detected_nodes.sum())
        fraction = detected_nodes.mean()*100
        
        return fraction, fraction >= min_fraction

    # -------------------------
    # Peak Slip Recovery
    # -------------------------
    def peak_slip_recovery(self, mask, col='s_dip'):
        true_vals = self.mtrue_s.loc[mask, col].values
        inv_vals  = self.m_s.loc[mask, col].values

        peak_true = true_vals.max()
        peak_inv  = inv_vals.max()

        return peak_inv / peak_true if peak_true > 0 else np.nan
    
    # -------------------------
    # Pattern Correlation
    # -------------------------
    def pattern_correlation(self, mask, col='s_dip'):
        true_vals = self.mtrue_s.loc[mask, col].values
        inv_vals  = self.m_s.loc[mask, col].values

        # mu_true = np.mean(true_vals)
        # mu_inv  = np.mean(inv_vals)

        # numerator = np.sum((true_vals - mu_true) * (inv_vals - mu_inv))
        # denominator = np.sqrt(
        #     np.sum((true_vals - mu_true)**2) *
        #     np.sum((inv_vals - mu_inv)**2)
        # )
        # return numerator / denominator if denominator > 0 else np.nan

        if len(true_vals) < 2 or np.all(true_vals == true_vals[0]) or np.all(inv_vals == inv_vals[0]):
            return np.nan  # correlation undefined for constant arrays 

        corr, _ = pearsonr(true_vals, inv_vals)
        return corr
  
    # -------------------------
    # Summary Table
    # -------------------------
    def summary(self, mask, col='s_dip', type='pattern', threshold=0.5):
        detection_frac, detection_pass = self.detection_rate(mask, col=col, threshold=threshold)
        zf_masked = self.loc_f.loc[mask, 'zf']

        if type=='pattern':
            metrics = {
                "Type": type,
                "NumNodes": mask.sum(),
                "DepthRange": f"{zf_masked.min():.2f} - {zf_masked.max():.2f}",
                "RMSE": self.rmse(mask, col=col),
                # "NRMSE": self.nrmse(mask, col=col),
                # "RL2norm": self.rel_l2_error(mask, col=col),
                "DetectionRate": detection_frac,
                "DetectionPass": detection_pass,
                "PeakRecovery": self.peak_slip_recovery(mask, col=col),
                # "Correlation": self.pattern_correlation(mask, col=col),
            }
        elif type=='gap':
            metrics = {
                "Type": type,
                "NumNodes": mask.sum(),
                "DepthRange": f"{zf_masked.min():.2f} - {zf_masked.max():.2f}",
                "RMSE": self.rmse(mask, col=col),
                # "NRMSE": self.nrmse(mask, col=col),
                # "RL2norm": self.rel_l2_error(mask, col=col),
            }
            
        return pd.DataFrame(metrics, index=[0]).T.rename(columns={0: "Value"})    


class GlobalSlipMetrics:
    def __init__(self, mtrue_s, m_s, loc_f):
        """
        mtrue_s, m_s: pandas DataFrames with slip cols (e.g. ['s_dip'])
        loc_f: pandas DataFrame with fault node geometry (must include 'zf')
        """
        self.mtrue_s = mtrue_s
        self.m_s = m_s
        self.loc_f = loc_f

    def rmse(self, col='s_dip'):
        t = self.mtrue_s[col].to_numpy()
        p = self.m_s[col].to_numpy()
        return np.sqrt(np.mean((t - p) ** 2))

    def detection_rate(self, col='s_dip', threshold=0.5, min_fraction=0.5):
        t = self.mtrue_s[col].to_numpy()
        p = self.m_s[col].to_numpy()

        # exclude zero-slip nodes from evaluation
        mask = t > 0
        if mask.sum() == 0:
            return np.nan, False
        # print(mask.sum())
        
        detected = p[mask] >= threshold * t[mask]
        frac = detected.mean()*100
        return frac, frac >= min_fraction

    def summary(self, col='s_dip', threshold=0.5):
        zf = self.loc_f['zf'].to_numpy()
        det_frac, det_pass = self.detection_rate(col=col, threshold=threshold)

        metrics = {
            "NumNodes": len(self.loc_f),
            "DepthRange": f"{zf.min():.2f} - {zf.max():.2f}",
            "RMSE": self.rmse(col=col),
            "DetectionRate": det_frac,
            "DetectionPass": det_pass
        }

        return pd.DataFrame(metrics, index=[0]).T.rename(columns={0: "Value"})
    

from scipy.stats import binned_statistic

def extract_directional_profile(x_coords, y_coords, values, azimuth_deg, 
                               profile_center_x=None, profile_center_y=None, 
                               profile_length=None, thickness_km=None, 
                               bin_width_km=1):
    """
    Extract profile along a specified direction
    
    Parameters:
    - azimuth_deg: direction in degrees CW from North
    - profile_center_x, profile_center_y: center point (if None, use data center)
    - profile_length: total length of profile in km (if None, use full data extent)
    - thickness_km: thickness of profile in km (if None, use ALL data - stack/average)
    - bin_width_km: spacing between bins along profile
    
    Returns:
    - profile_data: dict with binned results and metadata
    """
    
    # Set defaults if not provided
    if profile_center_x is None:
        profile_center_x = np.mean(x_coords)
    if profile_center_y is None:
        profile_center_y = np.mean(y_coords)
    
    az_rad = np.radians(azimuth_deg)
    
    # Direction vectors
    dir_along = np.array([np.sin(az_rad), np.cos(az_rad)])      # Along profile
    dir_perp = np.array([-np.cos(az_rad), np.sin(az_rad)])     # Perpendicular to profile
    
    # Translate coordinates relative to profile center
    coords_rel = np.column_stack([x_coords - profile_center_x, 
                                  y_coords - profile_center_y])
    
    # Project onto profile coordinate system
    proj_along = np.dot(coords_rel, dir_along)     # Distance along profile
    proj_perp = np.dot(coords_rel, dir_perp)       # Distance perpendicular to profile
    
    # Determine profile extent
    if profile_length is None:
        half_length = max(np.abs(np.min(proj_along)), np.abs(np.max(proj_along)))
        profile_length = 2 * half_length
    else:
        half_length = profile_length / 2
    
    # Filter points based on thickness
    if thickness_km is None:
        # Use ALL data - stack/average everything
        mask = (proj_along >= -half_length) & (proj_along <= half_length)
        mode = "stack_all"
        print(f"Mode: Stacking/averaging ALL data along {azimuth_deg}° direction")
    else:
        # Use thin slice
        half_thickness = thickness_km / 2
        mask = ((proj_along >= -half_length) & (proj_along <= half_length) & 
                (np.abs(proj_perp) <= half_thickness))
        mode = "thin_slice"
        print(f"Mode: Thin slice with {thickness_km} km thickness along {azimuth_deg}° direction")
    
    if np.sum(mask) == 0:
        print("No points found within the specified profile!")
        return None
    
    # Extract points within the profile
    prof_along = proj_along[mask]
    prof_perp = proj_perp[mask]
    prof_values = values[mask]
    prof_x = x_coords[mask]
    prof_y = y_coords[mask]
    
    print(f"Using {len(prof_values)} points out of {len(values)} total points")
    
    # Create bins along the profile
    bin_edges = np.arange(-half_length, half_length + bin_width_km, bin_width_km)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Bin the data using scipy for robustness
    bin_means, _, _ = binned_statistic(prof_along, prof_values, 
                                      statistic='mean', bins=bin_edges)
    bin_stds, _, _ = binned_statistic(prof_along, prof_values, 
                                     statistic='std', bins=bin_edges)
    bin_counts, _, _ = binned_statistic(prof_along, prof_values, 
                                       statistic='count', bins=bin_edges)
    
    # Calculate additional statistics for thin slice mode
    if mode == "thin_slice":
        # Perpendicular spread statistics
        perp_extent = np.max(prof_perp) - np.min(prof_perp)
        perp_std = np.std(prof_perp)
    else:
        perp_extent = np.max(proj_perp) - np.min(proj_perp)  # Full extent
        perp_std = np.std(proj_perp)
    
    profile_data = {
        'bin_centers': bin_centers,
        'values': bin_means,
        'std': bin_stds,
        'counts': bin_counts,
        'raw_along': prof_along,
        'raw_perp': prof_perp,
        'raw_values': prof_values,
        'raw_x': prof_x,
        'raw_y': prof_y,
        'mask': mask,
        'mode': mode,
        'azimuth_deg': azimuth_deg,
        'thickness_km': thickness_km,
        'profile_length': profile_length,
        'perp_extent': perp_extent,
        'perp_std': perp_std,
        'total_points': len(prof_values),
        'center_x': profile_center_x,
        'center_y': profile_center_y
    }
    
    return profile_data    


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_profile_enhanced(profile_sets, all_x, all_y, all_values_method1, all_values_method2, all_values_method3, mtrue_s, show_profile_boundary=True):
    """
    Enhanced plotting for multiple sets of directional profiles with 5 panels including true model
    
    Parameters:
    - profile_sets: list of lists, where each inner list contains profiles:
                   [prof_true, prof_hom, prof_sw, prof_3d] for each location
    - all_x, all_y: arrays of ALL coordinates (full dataset)
    - all_values_method1, all_values_method2, all_values_method3: inferred value arrays for each method
    - mtrue_s: true model values array
    - show_profile_boundary: whether to show profile boundaries on the map
    """
    
    # Flatten and validate profiles
    all_profiles = []
    set_info = []
    
    for set_idx, profile_set in enumerate(profile_sets):
        valid_profiles_in_set = [p for p in profile_set if p is not None]
        if valid_profiles_in_set:
            all_profiles.extend(valid_profiles_in_set)
            set_info.extend([set_idx] * len(valid_profiles_in_set))
    
    if not all_profiles:
        print("No valid profiles to plot")
        return
    
    # Create figure with 5 panels (2x3, but we'll use subplot2grid for better control)
    fig = plt.figure(figsize=(20, 12), dpi=300)
    
    # Create subplots with custom layout
    ax1 = plt.subplot2grid((2, 3), (0, 0))  # True model
    ax2 = plt.subplot2grid((2, 3), (0, 1))  # Hom
    ax3 = plt.subplot2grid((2, 3), (0, 2))  # SW
    ax4 = plt.subplot2grid((2, 3), (1, 0))  # 3D
    ax5 = plt.subplot2grid((2, 3), (1, 1), colspan=2)  # Profile comparisons (spans 2 columns)
    
    axes_scatter = [ax1, ax2, ax3, ax4]
    
    # Create discrete colormap with 10 bins
    # n_bins = 10
    n_bins = 20
    viridis_r = plt.get_cmap('viridis_r')
    colors = viridis_r(np.linspace(0, 1, n_bins))
    discrete_cmap = ListedColormap(colors)
    
    # Colorblind-friendly colors for different sets (locations)
    set_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green - colorblind friendly
    line_styles = ['--', '-.', ':', '-']  # Four line styles for four methods
    
    # Data arrays for each method (including true model)
    value_data = [all_values_method1, all_values_method2, all_values_method3, mtrue_s]
    method_names = ['Hom', 'SW', '3D', 'True']
    
    # Find global min/max for consistent color scaling
    all_values = np.concatenate(value_data)
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    
    # Panels 1-4: Scatter plots for each method (including true model)
    for method_idx in range(4):
        ax = axes_scatter[method_idx]
        
        # Scatter plot
        scatter = ax.scatter(all_x, all_y, c=value_data[method_idx], cmap=discrete_cmap, 
                            s=40, edgecolor='none', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Value')
        
        # Plot profile boundaries if requested
        if show_profile_boundary:
            for set_idx, profile_set in enumerate(profile_sets):
                # For true model (method_idx=0), use the first profile's geometry
                # For other methods, use the corresponding method's profile
                profile_data = None
                if method_idx == 0:
                    # Use first non-None profile for geometry (should be true model)
                    profile_data = profile_set[0] if len(profile_set) > 0 and profile_set[0] is not None else None
                else:
                    # Use corresponding method profile
                    if method_idx < len(profile_set) and profile_set[method_idx] is not None:
                        profile_data = profile_set[method_idx]
                
                if profile_data is not None:
                    if profile_data['mode'] == 'thin_slice':
                        # Draw thin slice boundary
                        thickness_km = profile_data['thickness_km']
                        profile_length = profile_data['profile_length']
                        azimuth_deg = profile_data['azimuth_deg']
                        center_x = profile_data['center_x']
                        center_y = profile_data['center_y']
                        
                        # Calculate boundary corners
                        half_length = profile_length / 2
                        half_thickness = thickness_km / 2
                        az_rad = np.radians(azimuth_deg)
                        
                        corners = np.array([
                            [-half_length, -half_thickness],
                            [half_length, -half_thickness], 
                            [half_length, half_thickness],
                            [-half_length, half_thickness],
                            [-half_length, -half_thickness]
                        ])
                        
                        # Rotation matrix
                        cos_az = np.cos(az_rad)
                        sin_az = np.sin(az_rad)
                        rotation_matrix = np.array([[sin_az, -cos_az],
                                                   [cos_az, sin_az]])
                        
                        # Rotate and translate
                        rotated_corners = corners @ rotation_matrix.T
                        rotated_corners[:, 0] += center_x
                        rotated_corners[:, 1] += center_y
                        
                        # Plot boundary
                        ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 
                               color=set_colors[set_idx], linewidth=2,
                               label=f'Profile {set_idx+1}' if method_idx == 0 else "")
            
            # Add legend only for first panel
            if method_idx == 0:
                ax.legend()
        
        # Configure subplot
        ax.set_xlabel('X coordinate (km)')
        ax.set_ylabel('Y coordinate (km)')
        ax.set_title(f'{method_names[method_idx]}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    # Panel 5: Profile comparisons (all methods including true model)
    ax = ax5
    
    # Get reference azimuth from first valid profile
    reference_profile = None
    for profile_set in profile_sets:
        for profile in profile_set:
            if profile is not None:
                reference_profile = profile
                break
        if reference_profile:
            break
    
    if reference_profile is None:
        print("No valid reference profile found")
        return
    
    reference_azimuth = reference_profile['azimuth_deg']
    az_rad = np.radians(reference_azimuth)
    dir_along = np.array([np.sin(az_rad), np.cos(az_rad)])
    
    # Plot all profiles for all methods
    for set_idx, profile_set in enumerate(profile_sets):
        for method_idx, profile_data in enumerate(profile_set):
            if profile_data is None:
                continue
                
            valid_mask = ~np.isnan(profile_data['values'])
            
            # Transform bin centers to common reference system
            profile_center = np.array([profile_data['center_x'], profile_data['center_y']])
            bin_centers_local = profile_data['bin_centers']
            
            common_distances = []
            for bin_center in bin_centers_local:
                global_pos = profile_center + bin_center * dir_along
                distance_from_origin = np.dot(global_pos, dir_along)
                common_distances.append(distance_from_origin)
            
            common_distances = np.array(common_distances)
            
            # Plot styling
            color = set_colors[set_idx]
            linestyle = line_styles[method_idx % 4]  # Cycle through line styles
            
            # Make true model line thicker for emphasis
            linewidth = 3 if method_idx == 3 else 2  # True model (method_idx=3) gets thicker line
            label = f'Profile {set_idx+1} {method_names[method_idx]}'
            
            # Line plot - no markers
            ax.plot(common_distances[valid_mask], 
                   profile_data['values'][valid_mask],
                   linestyle=linestyle, linewidth=linewidth, 
                   color=color, label=label)
    
    # Configure profile panel
    ax.set_xlabel(f'Distance along {reference_azimuth}° CW from North from (0,0) (km)')
    ax.set_ylabel('Average value')
    ax.set_title('Profile Comparisons (All Methods)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12) #bbox_to_anchor=(1.05, 1), loc='upper left'
    
    # Final layout
    plt.tight_layout()
    plt.show()


def rot_xy(x, y, rot):
    x = np.asarray(x)
    y = np.asarray(y)
    cos_rot = np.cos(np.radians(rot))
    sin_rot = np.sin(np.radians(rot))
    x_rot = x * cos_rot + y * sin_rot
    y_rot = -x * sin_rot + y * cos_rot

    return x_rot, y_rot    

def moment2mag(M_0):
    """
    Convert scalar moment in Nm to moment magnitude (Mw) using different conventions.
    
    Parameters:
    -----------
    M_0 : float or array_like
        Scalar seismic moment in Newton-meters (Nm)
    
    Returns:
    --------
    M_w1 : float or ndarray
        Moment magnitude using Hanks & Kanamori (1979): Mw = (2/3)*log10(M0*1.0e7) - 10.7

    M_w2 : float or ndarray
        Moment magnitude using Kanamori (1977): Mw = (2/3)*(log10(M0*1.0e7) - 16.1)

    M_w3 : float or ndarray
        In GCMT project, page bottom of https://www.globalcmt.org/CMTsearch.html
        "Note on calculation of moment magnitude: The moment 
        magnitude is calculated by this software using the formula of Kanamori (1977),
        MW = (2/3)*(log M0 - 16.1), where M0 is given in units of dyne-cm. Prior to 
        February 1, 2006, the quantity (2/3)*16.1 was rounded to the value 10.73. 
        For a small number of earthquakes, searches conducted after 2006/02/01 will 
        give values for MW that differ by 0.1 magnitude unit from values given by 
        searches prior to 2006/02/01."
        So essentially they are using below:
          Mw = (2/3)*log10(M0*1.0e7)-10.73
    """
    logM = np.log10(np.asarray(M_0) * 1.0e7)
    
    M_w1 = (2.0 / 3.0) * logM - 10.7
    M_w2 = (2.0 / 3.0) * (logM - 16.1)
    M_w3 = (2.0 / 3.0) * logM - 10.73

    return M_w1, M_w2, M_w3


