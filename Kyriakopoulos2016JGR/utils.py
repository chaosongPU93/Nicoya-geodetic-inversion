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

def remove_parallel_component(E, N, azimuth):
    """
    Remove the parallel component from a vector, keeping only the perpendicular part
    
    This is a convenience function that combines projection and decomposition
    to directly give you the vector with the parallel component removed.
    
    Parameters:
    -----------
    N : float or array-like
        North component of the original vector
    E : float or array-like
        East component of the original vector
    azimuth : float
        Azimuth angle in degrees for the direction to remove
    
    Returns:
    --------
    N_perpendicular : float or array
        North component after removing parallel component
    E_perpendicular : float or array
        East component after removing parallel component
    
    Examples:
    ---------
    # Remove N45E component from vector
    N_orig, E_orig = 10, 5
    N_perp, E_perp = remove_parallel_component(N_orig, E_orig, 45)
    """
    # Get the normal and parallel components along the azimuth direction
    normal, parallel = project_vector_2d_matrix(E, N, azimuth)
    # parallel, normal = project_vector_2d(E, N, azimuth)
    
    # set the parallel component to 0, then rotate back to N, E
    E_normal, N_normal = project_vector_2d_matrix(normal, 0*parallel, -azimuth)
    
    return E_normal, N_normal


def normal_to_NE(normal_component, azimuth):
    """
    Convert normal component back to North and East components
    
    This function takes the normal (perpendicular) component from a projection
    and converts it back to N and E components. This is useful for removing
    the parallel component and keeping only the perpendicular part.
    
    Parameters:
    -----------
    normal_component : float or array-like
        The normal component from the projection
    azimuth : float
        Azimuth angle in degrees (same as used in original projection)
    
    Returns:
    --------
    N_normal : float or array
        North component of the normal vector
    E_normal : float or array
        East component of the normal vector
    
    Examples:
    ---------
    # Remove parallel component to N45E, keep only perpendicular part
    N_orig, E_orig = 10, 5
    parallel, normal = project_vector_2d(N_orig, E_orig, 45)
    N_perp, E_perp = normal_to_NE(normal, 45)
    
    # Verify: N_perp, E_perp should be the original vector minus parallel component
    """
    # Convert inputs to numpy arrays
    normal_component = np.asarray(normal_component)
    
    # Convert azimuth to radians
    azimuth_rad = np.radians(azimuth)
    
    # Normal direction unit vector (90° counterclockwise from azimuth)
    normal_unit_N = np.sin(azimuth_rad)  # North component of normal unit vector
    normal_unit_E = np.cos(azimuth_rad)   # East component of normal unit vector
    
    # Scale the unit vector by the normal component magnitude
    N_normal = normal_component * normal_unit_N
    E_normal = normal_component * normal_unit_E
    
    return E_normal, N_normal 


def parallel_to_NE(parallel_component, azimuth):
    """
    Convert parallel component back to North and East components
    
    This function takes the parallel component from a projection
    and converts it back to N and E components. This is useful for
    reconstructing only the parallel part of the original vector.
    
    Parameters:
    -----------
    parallel_component : float or array-like
        The parallel component from the projection
    azimuth : float
        Azimuth angle in degrees (same as used in original projection)
    
    Returns:
    --------
    N_parallel : float or array
        North component of the parallel vector
    E_parallel : float or array
        East component of the parallel vector
    
    Examples:
    ---------
    # Get only the parallel component to N45E
    N_orig, E_orig = 10, 5
    parallel, normal = project_vector_2d(N_orig, E_orig, 45)
    N_par, E_par = parallel_to_NE(parallel, 45)
    
    # Verify: N_par, E_par should be the projection of original vector onto N45E
    """
    # Convert inputs to numpy arrays
    parallel_component = np.asarray(parallel_component)
    
    # Convert azimuth to radians
    azimuth_rad = np.radians(azimuth)
    
    # Parallel direction unit vector (along azimuth)
    parallel_unit_N = np.cos(azimuth_rad)  # North component of parallel unit vector
    parallel_unit_E = -np.sin(azimuth_rad)  # East component of parallel unit vector
    
    # Scale the unit vector by the parallel component magnitude
    N_parallel = parallel_component * parallel_unit_N
    E_parallel = parallel_component * parallel_unit_E
    
    return E_parallel, N_parallel 


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


import meshio
def rmse_3d_dataframe(predicted_df, true_df):
    """
    predicted_df: DataFrame with columns ['ux', 'uy', 'uz'] for predicted vectors
    true_df: DataFrame with columns ['ux', 'uy', 'uz'] for true vectors
    """
    # Convert to numpy arrays
    predicted = predicted_df[['ux', 'uy', 'uz']].values
    true = true_df[['ux', 'uy', 'uz']].values
    
    return np.sqrt(np.mean((predicted - true)**2))


def exo_to_msh(infile, outfile=None, gmsh_format="gmsh22"):
    """
    Convert an Exodus (.exo / .e) mesh to Gmsh .msh format.

    Parameters
    ----------
    infile : str
        Path to the input Exodus file (.exo or .e).
    outfile : str, optional
        Path to the output Gmsh file (.msh). If None, uses same basename as infile.
    gmsh_format : str
        Output Gmsh format. Default is "gmsh22" (ASCII v2.2, widely compatible).

    Returns
    -------
    outfile : str
        Path to the saved Gmsh mesh.
    """
    if outfile is None:
        outfile = infile.rsplit(".", 1)[0] + ".msh"

    print(f"Reading Exodus file: {infile}")
    mesh = meshio.read(infile)

    print("Cell types:", mesh.cells_dict.keys())
    print("Cell data keys:", mesh.cell_data.keys())
    print("Point data keys:", mesh.point_data.keys())

    meshio.write(outfile, mesh, file_format=gmsh_format)
    print(f"✅ Wrote Gmsh mesh: {outfile}")

    return outfile


import meshio
from pathlib import Path
import numpy as np

def exo_to_fenics_xml(infile, outdir=None, facet_map=None, volume_map=None):
    """
    Convert Exodus mesh to FEniCS XML files with custom ID mapping:
      - meshname.xml
      - meshname_facet_region.xml
      - meshname_physical_region.xml

    Parameters
    ----------
    infile : str
        Input Exodus mesh (.exo or .e)
    outdir : str, optional
        Output directory (default: same as infile)
    facet_map : dict, optional
        Mapping {old_id: new_id} for boundary facets
    volume_map : dict, optional
        Mapping {old_id: new_id} for volume cells
    """
    infile_path = Path(infile)
    if outdir is None:
        outdir = infile_path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meshname = infile_path.stem

    print(f"Reading Exodus file: {infile}")
    mesh = meshio.read(infile)

    # Detect volume element type
    if "tetra" in mesh.cells_dict:
        dim = 3
        main_type = "tetra"
    elif "triangle" in mesh.cells_dict:
        dim = 2
        main_type = "triangle"
    else:
        raise ValueError("Mesh must contain tetra or triangle elements.")

    # Detect facet element type
    if dim == 3 and "triangle" in mesh.cells_dict:
        facet_type = "triangle"
    elif dim == 2 and "line" in mesh.cells_dict:
        facet_type = "line"
    else:
        facet_type = None

    # Helper: find matching key
    def find_key(possible_keys):
        for k in mesh.cell_data.keys():
            if k.lower() in possible_keys:
                return k
        return None

    # Volume markers
    vol_key = find_key(["block_id", "element_block_id", "cell_marker", "physical", "region_id"])
    if vol_key:
        vol_idx = next(i for i, c in enumerate(mesh.cells) if c.type == main_type)
        vol_markers = mesh.cell_data[vol_key][vol_idx].copy()
        if volume_map:
            vol_markers = np.array([volume_map.get(v, v) for v in vol_markers])
    else:
        vol_markers = None

    # Facet markers
    if facet_type:
        facet_key = find_key(["surface_id", "face_id", "ss_id", "sideset", "boundary_id", "line_id"])
        if facet_key:
            facet_idx = next(i for i, c in enumerate(mesh.cells) if c.type == facet_type)
            facet_markers = mesh.cell_data[facet_key][facet_idx].copy()
            if facet_map:
                facet_markers = np.array([facet_map.get(v, v) for v in facet_markers])
        else:
            facet_markers = None
    else:
        facet_markers = None

    # Write mesh geometry XML
    meshio.write(
        outdir / f"{meshname}.xml",
        meshio.Mesh(points=mesh.points, cells=[(main_type, mesh.cells_dict[main_type])])
    )
    print(f"✅ Wrote {meshname}.xml")

    # Write volume XML
    if vol_markers is not None:
        meshio.write(
            outdir / f"{meshname}_physical_region.xml",
            meshio.Mesh(points=mesh.points, cells=[(main_type, mesh.cells_dict[main_type])],
                        cell_data={"physical_region": [vol_markers]})
        )
        print(f"✅ Wrote {meshname}_physical_region.xml")

    # Write facet XML
    if facet_type and facet_markers is not None:
        meshio.write(
            outdir / f"{meshname}_facet_region.xml",
            meshio.Mesh(points=mesh.points, cells=[(facet_type, mesh.cells_dict[facet_type])],
                        cell_data={"facet_region": [facet_markers]})
        )
        print(f"✅ Wrote {meshname}_facet_region.xml")

    return {
        "mesh": str(outdir / f"{meshname}.xml"),
        "physical_region": str(outdir / f"{meshname}_physical_region.xml") if vol_markers is not None else None,
        "facet_region": str(outdir / f"{meshname}_facet_region.xml") if facet_markers is not None else None
    }


import meshio
from pathlib import Path
import numpy as np

def exo_fault_to_points(infile, geo_file=None, dx_str="dx_fault_coarse"):
    mesh = meshio.read(infile)
    points = mesh.points

    if geo_file is None:
        geo_file = str(Path(infile).with_suffix(".geo"))

    with open(geo_file, "a") as f:
        # Points
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"Point({i}) = {{{x}, {y}, {z}, {dx_str}}};\n")

    print(f"Wrote {geo_file} with {len(points)} points")


def exo_fault_to_geo(infile, geo_file=None, lc=1.0, fault_id=7, triangulate_quads=True):
    mesh = meshio.read(infile)
    points = mesh.points

    # Detect triangles or quads
    surf_cells = None
    for key, data in mesh.cells_dict.items():
        if key == "triangle":
            surf_cells = data
            break
        elif key == "quad" and triangulate_quads:
            # Split quads into triangles
            surf_cells = []
            for q in data:
                surf_cells.append([q[0], q[1], q[2]])
                surf_cells.append([q[0], q[2], q[3]])
            surf_cells = np.array(surf_cells)
            break
    if surf_cells is None:
        raise ValueError("No surface elements found!")

    if geo_file is None:
        geo_file = str(Path(infile).with_suffix(".geo"))

    with open(geo_file, "w") as f:
        # Points
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"Point({i}) = {{{x}, {y}, {z}, {lc}}};\n")

        # # Lines with orientation storage
        # edge_dict = {}  # key=(min,max), value=(line_id, direction_from_min)
        # line_id = 1
        # for tri in surf_cells:
        #     edges = [(tri[0]+1, tri[1]+1), (tri[1]+1, tri[2]+1), (tri[2]+1, tri[0]+1)]
        #     for e in edges:
        #         e_sorted = tuple(sorted(e))
        #         if e_sorted not in edge_dict:
        #             edge_dict[e_sorted] = (line_id, e[0] == e_sorted[0])
        #             f.write(f"Line({line_id}) = {{{e_sorted[0]}, {e_sorted[1]}}};\n")
        #             line_id += 1

        # # Surfaces with proper line orientation
        # loop_id = 1
        # surf_id = 1
        # for tri in surf_cells:
        #     tri_edges = [
        #         (tri[0]+1, tri[1]+1),
        #         (tri[1]+1, tri[2]+1),
        #         (tri[2]+1, tri[0]+1)
        #     ]
        #     loop_lines = []
        #     for e in tri_edges:
        #         e_sorted = tuple(sorted(e))
        #         lid, forward = edge_dict[e_sorted]
        #         if forward == (e[0] == e_sorted[0]):
        #             loop_lines.append(f"{lid}")
        #         else:
        #             loop_lines.append(f"-{lid}")
        #     f.write(f"Line Loop({loop_id}) = {{{', '.join(loop_lines)}}};\n")
        #     f.write(f"Plane Surface({surf_id}) = {{{loop_id}}};\n")
        #     loop_id += 1
        #     surf_id += 1

        # # Physical surface for the fault
        # surf_ids_str = ", ".join(str(i) for i in range(1, surf_id))
        # f.write(f"Physical Surface({fault_id}) = {{{surf_ids_str}}};\n")

    print(f"✅ Wrote {geo_file} with {len(points)} points and {len(surf_cells)} triangles")


from pathlib import Path

def expand_gmsh_points(expr):
    """
    Expand a Gmsh-like point expression into an explicit list of integers.

    Example:
        "1, 2, 5:2:287" -> [1, 2, 5, 7, 9, ..., 287]
    """
    expr = expr.replace(" ", "")
    parts = expr.split(",")
    expanded = []

    for part in parts:
        if ":" in part:
            nums = list(map(int, part.split(":")))
            if len(nums) == 2:
                start, end = nums
                expanded.extend(range(start, end+1))
            elif len(nums) == 3:
                start, step, end = nums
                expanded.extend(range(start, end+1, step))
            else:
                raise ValueError(f"Invalid range syntax: {part}")
        else:
            expanded.append(int(part))
    return expanded


def append_spline_to_geo(geo_file, spline_id, expr):
    """
    Expand a shorthand expression and append a Spline definition to a .geo file.

    Parameters
    ----------
    geo_file : str or Path
        Path to the existing .geo file
    spline_id : int
        ID for the spline in Gmsh
    expr : str
        Shorthand point expression, e.g. "1, 2, 5:2:287"
    """
    geo_path = Path(geo_file)
    if not geo_path.exists():
        raise FileNotFoundError(f"{geo_file} not found")

    points = expand_gmsh_points(expr)
    spline_line = f"Spline({spline_id}) = {{{', '.join(map(str, points))}}};\n"

    with open(geo_path, "a") as f:
        f.write(spline_line)

    print(f"Spline({spline_id}) added to {geo_file}:")
    # print(spline_line.strip())

    # Example usage
    # append_spline_to_geo("Nicoya_interface.geo", 1, "1, 2, 5:2:287")


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