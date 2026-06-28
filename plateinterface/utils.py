import pandas as pd
import numpy as np
import re

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
    kprime = c / np.sin(c)
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
    # Convert center lon/lat to radians
    lon0_rad, lat0_rad = np.radians(lon0), np.radians(lat0)

    # Compute angular distance c
    c = np.sqrt(x**2 + y**2) / R
    
    # Compute latitude
    lat_rad = np.arcsin(np.cos(c) * np.sin(lat0_rad) + y * np.sin(c) * np.cos(lat0_rad) / c / R)

    # Compute longitude
    if lat0 == 90:
        lon_rad = lon0_rad + np.arctan2(-x, y)
    elif lat0 == -90:
        lon_rad = lon0_rad + np.arctan2(x, y)
    else:
        lon_rad = lon0_rad + np.arctan2(x * np.sin(c) / R, 
                                        c * np.cos(lat0_rad) * np.cos(c) - y * np.sin(lat0_rad) * np.sin(c) / R)

    # Convert radians to degrees
    lon, lat = np.degrees(lon_rad), np.degrees(lat_rad)

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