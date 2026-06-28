import pandas as pd
import numpy as np
import re
import dolfin as dl

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


import meshio
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


def validate_fault_slip_pattern(mtrue_s, mesh, boundaries, fault_id, 
                               boundary_names=None, verbose=True):
    """
    Comprehensive validation of fault slip pattern to ensure it's applied only to the fault interface
    
    Parameters:
    -----------
    mtrue_s : dolfin.Vector or dolfin.Function.vector()
        The slip vector to validate
    mesh : dolfin.Mesh
        The computational mesh
    boundaries : dolfin.MeshFunction
        Boundary markers
    fault_id : int
        ID of the fault boundary
    boundary_names : dict, optional
        Mapping from boundary IDs to names for reporting
    verbose : bool, optional
        Whether to print detailed output (default: True)
    
    Returns:
    --------
    dict : Validation results with statistics
    """
    import numpy as np
    import dolfin as dl
    
    # Default boundary names if not provided
    if boundary_names is None:
        boundary_names = {
            1: "top", 2: "bottom", 3: "west", 4: "east", 
            5: "north", 6: "south", 7: "fault", 8: "blockleft", 9: "blockright"
        }
    
    # Create function space for marking boundaries
    Vm = dl.VectorFunctionSpace(mesh, "CG", 1, dim=2)
    
    results = {}
    
    if verbose:
        print("="*60)
        print("COMPREHENSIVE FAULT SLIP VALIDATION")
        print("="*60)
    
    # Validation 1: Check fault interface
    if verbose:
        print("\n=== FAULT INTERFACE ===")
    
    bc_fault_marker = dl.DirichletBC(Vm, (99, 99), boundaries, fault_id)
    um_fault = dl.Function(Vm)
    bc_fault_marker.apply(um_fault.vector())
    mtrue_s_fault = mtrue_s.get_local()[um_fault.vector().get_local() == 99]
    
    if len(mtrue_s_fault) > 0:
        fault_strike = mtrue_s_fault[0::2]
        fault_dip = mtrue_s_fault[1::2]
        
        results['fault'] = {
            'num_nodes': len(mtrue_s_fault) // 2,
            'strike_slip': {
                'min': float(fault_strike.min()),
                'max': float(fault_strike.max()),
                'mean': float(fault_strike.mean()),
                'std': float(fault_strike.std()),
                'nonzero_count': int(np.count_nonzero(fault_strike))
            },
            'dip_slip': {
                'min': float(fault_dip.min()),
                'max': float(fault_dip.max()),
                'mean': float(fault_dip.mean()),
                'std': float(fault_dip.std()),
                'nonzero_count': int(np.count_nonzero(fault_dip))
            }
        }
        
        if verbose:
            print(f"Fault nodes: {results['fault']['num_nodes']}")
            print(f"Strike slip - Min: {results['fault']['strike_slip']['min']:.6f}, "
                  f"Max: {results['fault']['strike_slip']['max']:.6f}, "
                  f"Mean: {results['fault']['strike_slip']['mean']:.6f}")
            print(f"Dip slip - Min: {results['fault']['dip_slip']['min']:.6f}, "
                  f"Max: {results['fault']['dip_slip']['max']:.6f}, "
                  f"Mean: {results['fault']['dip_slip']['mean']:.6f}")
            print(f"Non-zero dip slip values: {results['fault']['dip_slip']['nonzero_count']}/{results['fault']['num_nodes']}")
    else:
        results['fault'] = {'error': 'No fault nodes found'}
        if verbose:
            print("ERROR: No fault nodes found!")
    
    # Validation 2: Check non-fault boundaries
    if verbose:
        print("\n=== NON-FAULT BOUNDARIES ===")
    
    # Get all unique boundary IDs except the fault
    all_boundary_ids = set(boundaries.array()) - {fault_id}
    
    results['other_boundaries'] = {}
    
    for boundary_id in all_boundary_ids:
        if boundary_id == 0:  # Skip interior (typically 0)
            continue
            
        bc_other = dl.DirichletBC(Vm, (88, 88), boundaries, boundary_id)
        um_other = dl.Function(Vm)
        bc_other.apply(um_other.vector())
        mtrue_s_other = mtrue_s.get_local()[um_other.vector().get_local() == 88]
        
        if len(mtrue_s_other) > 0:
            other_strike = mtrue_s_other[0::2]
            other_dip = mtrue_s_other[1::2]
            
            boundary_name = boundary_names.get(boundary_id, f"boundary_{boundary_id}")
            
            results['other_boundaries'][boundary_name] = {
                'id': int(boundary_id),
                'num_nodes': len(mtrue_s_other) // 2,
                'strike_slip': {
                    'min': float(other_strike.min()),
                    'max': float(other_strike.max()),
                    'max_abs': float(np.abs(other_strike).max()),
                    'nonzero_count': int(np.count_nonzero(other_strike))
                },
                'dip_slip': {
                    'min': float(other_dip.min()),
                    'max': float(other_dip.max()),
                    'max_abs': float(np.abs(other_dip).max()),
                    'nonzero_count': int(np.count_nonzero(other_dip))
                }
            }
            
            if verbose:
                print(f"{boundary_name} (ID {boundary_id}):")
                print(f"  Nodes: {results['other_boundaries'][boundary_name]['num_nodes']}")
                print(f"  Strike slip - Min: {results['other_boundaries'][boundary_name]['strike_slip']['min']:.6f}, "
                      f"Max: {results['other_boundaries'][boundary_name]['strike_slip']['max']:.6f}")
                print(f"  Dip slip - Min: {results['other_boundaries'][boundary_name]['dip_slip']['min']:.6f}, "
                      f"Max: {results['other_boundaries'][boundary_name]['dip_slip']['max']:.6f}")
                
                # Flag potential issues
                max_abs_strike = results['other_boundaries'][boundary_name]['strike_slip']['max_abs']
                max_abs_dip = results['other_boundaries'][boundary_name]['dip_slip']['max_abs']
                if max_abs_strike > 1e-12 or max_abs_dip > 1e-12:
                    print(f"  WARNING: Non-zero values detected (max abs: strike={max_abs_strike:.2e}, dip={max_abs_dip:.2e})")
    
    # Validation 3: Check interior/volume nodes
    if verbose:
        print("\n=== INTERIOR/VOLUME NODES ===")
    
    # Create mask for all boundary nodes
    all_boundary_mask = dl.Function(Vm)
    all_boundary_mask.vector()[:] = 0  # Initialize to 0 (interior)
    
    # Mark all boundary nodes with 1
    for boundary_id in all_boundary_ids | {fault_id}:
        if boundary_id == 0:
            continue
        try:
            bc_boundary = dl.DirichletBC(Vm, (1, 1), boundaries, boundary_id)
            bc_boundary.apply(all_boundary_mask.vector())
        except:
            pass  # Skip if boundary doesn't exist
    
    # Interior nodes are where mask == 0
    interior_mask = (all_boundary_mask.vector().get_local() == 0)
    mtrue_s_interior = mtrue_s.get_local()[interior_mask]
    
    if len(mtrue_s_interior) > 0:
        interior_strike = mtrue_s_interior[0::2]
        interior_dip = mtrue_s_interior[1::2]
        
        results['interior'] = {
            'num_nodes': len(mtrue_s_interior) // 2,
            'strike_slip': {
                'min': float(interior_strike.min()),
                'max': float(interior_strike.max()),
                'max_abs': float(np.abs(interior_strike).max()),
                'nonzero_count': int(np.count_nonzero(interior_strike))
            },
            'dip_slip': {
                'min': float(interior_dip.min()),
                'max': float(interior_dip.max()),
                'max_abs': float(np.abs(interior_dip).max()),
                'nonzero_count': int(np.count_nonzero(interior_dip))
            }
        }
        
        if verbose:
            print(f"Interior nodes: {results['interior']['num_nodes']}")
            print(f"Strike slip - Min: {results['interior']['strike_slip']['min']:.6f}, "
                  f"Max: {results['interior']['strike_slip']['max']:.6f}")
            print(f"Dip slip - Min: {results['interior']['dip_slip']['min']:.6f}, "
                  f"Max: {results['interior']['dip_slip']['max']:.6f}")
            print(f"Non-zero interior values: {results['interior']['strike_slip']['nonzero_count'] + results['interior']['dip_slip']['nonzero_count']}/{len(mtrue_s_interior)}")
            
            # Flag issues
            total_nonzero = results['interior']['strike_slip']['nonzero_count'] + results['interior']['dip_slip']['nonzero_count']
            if total_nonzero > 0:
                print(f"WARNING: {total_nonzero} non-zero interior values detected!")
    else:
        results['interior'] = {'error': 'No interior nodes found or all nodes are on boundaries'}
        if verbose:
            print("No interior nodes found (or all nodes are on boundaries)")
    
    # Validation 4: Global statistics
    if verbose:
        print("\n=== GLOBAL STATISTICS ===")
    
    all_values = mtrue_s.get_local()
    total_nonzero = np.count_nonzero(all_values)
    
    results['global'] = {
        'total_dofs': len(all_values),
        'total_nonzero': int(total_nonzero),
        'percentage_nonzero': float(100 * total_nonzero / len(all_values)),
        'global_min': float(all_values.min()),
        'global_max': float(all_values.max()),
        'global_mean': float(all_values.mean()),
        'global_std': float(all_values.std())
    }
    
    if verbose:
        print(f"Total DOFs: {results['global']['total_dofs']}")
        print(f"Total non-zero values: {results['global']['total_nonzero']}")
        print(f"Global min: {results['global']['global_min']:.6f}")
        print(f"Global max: {results['global']['global_max']:.6f}")
        print(f"Percentage of non-zero DOFs: {results['global']['percentage_nonzero']:.2f}%")
    
    # Summary assessment
    if verbose:
        print("\n=== VALIDATION SUMMARY ===")
        
        issues = []
        
        # Check if slip is confined to fault
        for boundary_name, boundary_data in results.get('other_boundaries', {}).items():
            max_abs = max(boundary_data['strike_slip']['max_abs'], boundary_data['dip_slip']['max_abs'])
            if max_abs > 1e-12:
                issues.append(f"Non-zero slip on {boundary_name} boundary")
        
        if 'interior' in results and not 'error' in results['interior']:
            interior_nonzero = results['interior']['strike_slip']['nonzero_count'] + results['interior']['dip_slip']['nonzero_count']
            if interior_nonzero > 0:
                issues.append(f"{interior_nonzero} non-zero interior values")
        
        if len(issues) == 0:
            print("✅ VALIDATION PASSED: Slip pattern is correctly confined to fault interface")
        else:
            print("❌ VALIDATION ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")
    
    if verbose:
        print("="*60)
    
    return results

    # # Usage examples:

    # # Basic usage with your setup
    # validation_results = validate_fault_slip_pattern(
    #     mtrue_s=mtrue_s,
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault
    # )

    # # With custom boundary names
    # custom_boundary_names = {
    #     1: "surface", 2: "bottom", 3: "west", 4: "east", 
    #     5: "north", 6: "south", 7: "fault", 8: "left_block", 9: "right_block"
    # }

    # validation_results = validate_fault_slip_pattern(
    #     mtrue_s=mtrue_s,
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault,
    #     boundary_names=custom_boundary_names
    # )

    # # Silent validation (just get results without printing)
    # validation_results = validate_fault_slip_pattern(
    #     mtrue_s=mtrue_s,
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault,
    #     verbose=False
    # )

    # # Access specific results
    # fault_stats = validation_results['fault']
    # global_stats = validation_results['global']
    # print(f"Fault has {fault_stats['num_nodes']} nodes with dip slip range: "
    #     f"{fault_stats['dip_slip']['min']:.3f} to {fault_stats['dip_slip']['max']:.3f}")    


class CircularGaussianS(dl.UserExpression):
### PROBLEMATIC! MIGHT CREATE A VOLUMATRIC PATTERN RATHER THAN A PLANAR PATTERN ###   
# generate a Fenics expression of a 2D Gaussian slip model where the contours are circular
    def __init__(self, x0=0.0, y0=0.0, sigma=0.3, maxamp=100, **kwargs):
        super().__init__(**kwargs)
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.maxamp = maxamp

    def eval(self, values, x):
        r2 = (x[0] - self.x0)**2 + (x[1] - self.y0)**2
        s = self.maxamp * np.exp(-r2 / (2 * self.sigma**2))
        values[0] = s

    def value_shape(self):
        return ()
    
    # Example usage 
    # # design a 2D Gaussian slip model where the contours are circular, m_strike = 0
    # x0 = 0
    # y0 = 0
    # sigma = 10e3  # the Gaussian would be nearly 0 at +- 3*sigma
    # maxamp = 0.2     # max. slip, e.g. 200 mm
    # mtrue_s_expr_dip = CircularGaussianS(x0=x0, y0=y0, sigma=sigma, maxamp=maxamp, degree=3)
    # # Final true model expression (m_strike = 0, m_dip = checkerboard inside the specified region)
    # mtrue_s_expr_gt = dl.Expression(('0.', 'mask1'), mask1=mtrue_s_expr_dip, degree=3)
    

class CircularStepS(dl.UserExpression):
### PROBLEMATIC! MIGHT CREATE A VOLUMATRIC PATTERN RATHER THAN A PLANAR PATTERN ###   
# generate a Fenics expression of a circular step function or characteristic function: the value is constant inside a circular region and zero outside.
    def __init__(self, x0=0.5, y0=0.5, R=0.3, A=100.0, **kwargs):
        super().__init__(**kwargs)
        self.x0 = x0
        self.y0 = y0
        self.R = R
        self.A = A

    def eval(self, values, x):
        r2 = (x[0] - self.x0)**2 + (x[1] - self.y0)**2
        if r2 <= self.R**2:
            values[0] = self.A
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()    
    
    # Example usage 
    # # design a step model where the slip are constant inside a circular region, otherwise 0; m_strike = 0
    # x0 = 10e3
    # y0 = 0
    # radius = 40e3    # radius of the circular region, in m
    # nevt = 5          # number of events stacked, e.g., 5 events
    # maxamp = -0.2*nevt     # max. slip, e.g. 200 mm, note that the negative sign here aligns with a thrust shear slip; assume a stack of 5 events
    # mtrue_s_expr_dip = CircularStepS(x0=x0, y0=y0, R=radius, A=maxamp, degree=1)
    # # Final true model expression (m_strike = 0, m_dip = checkerboard inside the specified region)
    # mtrue_s_expr_gt = dl.Expression(('0.', 'mask1'), mask1=mtrue_s_expr_dip, degree=1)


class FaultLocalCircularGaussian(dl.UserExpression):
    """
    FIXED VERSION: Creates circular Gaussian slip pattern confined to fault interface
    
    Key improvements:
    - Fault-confined: Only applies pattern to cells adjacent to fault boundary
    - Fault-local coordinates: Uses proper strike-dip coordinate system
    - 3D slip vector output: Returns (strike_slip, dip_slip) components
    - Supports rotation and constraints like other fault-local patterns
    """
    
    def __init__(self, mesh, boundaries, fault_id, amp, x0=0, y0=0, sigma=10e3, 
                 rotation_deg=0.0, zmin=None, zmax=None, 
                 strike_slip=False, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.boundaries = boundaries
        self.fault_id = fault_id
        self.amp = amp
        self.x0 = x0  # center position along strike (fault-local)
        self.y0 = y0  # center position up-dip (fault-local)
        self.sigma = sigma  # Gaussian width parameter
        
        # Pattern rotation in fault-local coordinates
        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.cos_rot = np.cos(self.rotation_rad)
        self.sin_rot = np.sin(self.rotation_rad)
        
        # Slip direction: True = strike-slip, False = dip-slip
        self.strike_slip = strike_slip
        
        if abs(rotation_deg) > 1e-10:
            print(f"Gaussian pattern rotation: {rotation_deg:.1f}° in fault-local coordinates")
        
        # Optional depth constraints
        self.use_depth_constraint = (zmin is not None) and (zmax is not None)
        if self.use_depth_constraint:
            self.zmin = zmin
            self.zmax = zmax
            print(f"Using depth constraints: {zmin/1000:.1f} to {zmax/1000:.1f} km")
        
        # Compute fault geometry from mesh
        self._compute_fault_geometry()
        
        print(f"Circular Gaussian pattern:")
        print(f"  Center: ({x0/1000:.1f}, {y0/1000:.1f}) km in fault-local coords")
        print(f"  Sigma: {sigma/1000:.1f} km")
        print(f"  Amplitude: {amp} m")
        print(f"  Slip direction: {'strike' if strike_slip else 'dip'}")
        
    def _compute_fault_geometry(self):
        """Extract fault strike and dip from mesh geometry - same as FaultLocalCheckerboard"""
        print("Computing fault geometry from mesh...")
        
        # Collect fault boundary facets and their properties
        fault_facets = []
        fault_normals = []
        fault_centers = []
        
        # Iterate through all facets
        for facet in dl.facets(self.mesh):
            if self.boundaries[facet] == self.fault_id:
                fault_facets.append(facet)
                
                # Get facet normal
                normal = facet.normal()
                fault_normals.append([normal.x(), normal.y(), normal.z()])
                
                # Get facet center
                center = facet.midpoint()
                fault_centers.append([center.x(), center.y(), center.z()])
        
        if not fault_facets:
            raise ValueError(f"No facets found with fault_id {self.fault_id}")
        
        # Convert to numpy arrays
        fault_normals = np.array(fault_normals)
        fault_centers = np.array(fault_centers)
        
        # Compute average normal vector (should be consistent across fault)
        avg_normal = np.mean(fault_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)  # normalize
        
        # Define fault-local coordinate system
        # Strike direction: horizontal vector perpendicular to normal
        vertical = np.array([0, 0, 1])  # z-direction
        strike_vector = np.cross(avg_normal, vertical)
        strike_vector = strike_vector / np.linalg.norm(strike_vector)
        
        # Dip direction: perpendicular to both strike and normal
        dip_vector = np.cross(strike_vector, avg_normal)
        dip_vector = dip_vector / np.linalg.norm(dip_vector)
        
        # Store coordinate system
        self.normal_vector = avg_normal
        self.strike_vector = strike_vector  
        self.dip_vector = dip_vector
        
        print(f"Fault geometry computed:")
        print(f"  Normal vector: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]")
        print(f"  Strike vector: [{strike_vector[0]:.3f}, {strike_vector[1]:.3f}, {strike_vector[2]:.3f}]")
        print(f"  Dip vector: [{dip_vector[0]:.3f}, {dip_vector[1]:.3f}, {dip_vector[2]:.3f}]")
    
    def _global_to_fault_coords(self, x):
        """Transform global coordinates to fault-local coordinates"""
        s = np.dot([x[0], x[1], x[2]], self.strike_vector)
        d = np.dot([x[0], x[1], x[2]], self.dip_vector)  
        n = np.dot([x[0], x[1], x[2]], self.normal_vector)
        
        return s, d, n
    
    def _rotate_fault_coords(self, s, d):
        """Rotate coordinates in the fault plane by specified angle"""
        # 2D rotation matrix in fault plane
        s_centered = s - self.x0
        d_centered = d - self.y0
        
        s_rot = self.cos_rot * s_centered - self.sin_rot * d_centered
        d_rot = self.sin_rot * s_centered + self.cos_rot * d_centered
        
        return s_rot, d_rot
    
    def _gaussian_pattern(self, s_rot, d_rot):
        """Compute Gaussian pattern value"""
        r_squared = s_rot**2 + d_rot**2
        gaussian_value = np.exp(-r_squared / (2 * self.sigma**2))
        return gaussian_value
    
    def eval_cell(self, values, x, cell):
        """Evaluate the expression at a given point - same logic as other fault-local classes"""
        # Check if this cell is adjacent to the fault boundary
        cell_obj = dl.Cell(self.mesh, cell.index)
        is_on_fault = False
        
        for facet in dl.facets(cell_obj):
            if self.boundaries[facet] == self.fault_id:
                is_on_fault = True
                break
        
        if is_on_fault:
            # Apply additional depth constraint if specified
            if self.use_depth_constraint:
                depth_mask = 1.0 if (self.zmin <= x[2] <= self.zmax) else 0.0
            else:
                depth_mask = 1.0  # No additional depth constraint
            
            if depth_mask > 0:  # Only compute pattern if within depth range
                # Transform to fault-local coordinates
                s, d, n = self._global_to_fault_coords(x)
                
                # Apply rotation in fault-local coordinates
                s_rot, d_rot = self._rotate_fault_coords(s, d)
                
                # Create Gaussian pattern using rotated coordinates
                gaussian_value = self._gaussian_pattern(s_rot, d_rot)
                
                # Set slip components based on slip direction
                if self.strike_slip:
                    values[0] = self.amp * gaussian_value  # strike slip
                    values[1] = 0.0  # no dip slip
                else:
                    values[0] = 0.0  # no strike slip
                    values[1] = self.amp * gaussian_value  # dip slip
            else:
                # Outside depth range
                values[0] = 0.0
                values[1] = 0.0
        else:
            # Not on fault interface
            values[0] = 0.0
            values[1] = 0.0
    
    def value_shape(self):
        return (2,)  # Return 2D slip vector (strike, dip)


class FaultLocalCircularStep(dl.UserExpression):
    """
    FIXED VERSION: Creates circular step function slip pattern confined to fault interface
    
    Key improvements:
    - Fault-confined: Only applies pattern to cells adjacent to fault boundary  
    - Fault-local coordinates: Uses proper strike-dip coordinate system
    - 3D slip vector output: Returns (strike_slip, dip_slip) components
    - Supports rotation and constraints like other fault-local patterns
    """
    
    def __init__(self, mesh, boundaries, fault_id, amp, x0=0, y0=0, radius=20e3,
                 rotation_deg=0.0, zmin=None, zmax=None,
                 strike_slip=False, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.boundaries = boundaries
        self.fault_id = fault_id
        self.amp = amp
        self.x0 = x0  # center position along strike (fault-local)
        self.y0 = y0  # center position up-dip (fault-local)
        self.radius = radius  # circular region radius
        
        # Pattern rotation in fault-local coordinates
        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.cos_rot = np.cos(self.rotation_rad)
        self.sin_rot = np.sin(self.rotation_rad)
        
        # Slip direction: True = strike-slip, False = dip-slip
        self.strike_slip = strike_slip
        
        if abs(rotation_deg) > 1e-10:
            print(f"Step pattern rotation: {rotation_deg:.1f}° in fault-local coordinates")
        
        # Optional depth constraints
        self.use_depth_constraint = (zmin is not None) and (zmax is not None)
        if self.use_depth_constraint:
            self.zmin = zmin
            self.zmax = zmax
            print(f"Using depth constraints: {zmin/1000:.1f} to {zmax/1000:.1f} km")
        
        # Compute fault geometry from mesh
        self._compute_fault_geometry()
        
        print(f"Circular step pattern:")
        print(f"  Center: ({x0/1000:.1f}, {y0/1000:.1f}) km in fault-local coords")
        print(f"  Radius: {radius/1000:.1f} km")
        print(f"  Amplitude: {amp} m")
        print(f"  Slip direction: {'strike' if strike_slip else 'dip'}")
    
    def _compute_fault_geometry(self):
        """Extract fault strike and dip from mesh geometry - identical to Gaussian version"""
        print("Computing fault geometry from mesh...")
        
        # Collect fault boundary facets and their properties
        fault_facets = []
        fault_normals = []
        fault_centers = []
        
        # Iterate through all facets
        for facet in dl.facets(self.mesh):
            if self.boundaries[facet] == self.fault_id:
                fault_facets.append(facet)
                
                # Get facet normal
                normal = facet.normal()
                fault_normals.append([normal.x(), normal.y(), normal.z()])
                
                # Get facet center
                center = facet.midpoint()
                fault_centers.append([center.x(), center.y(), center.z()])
        
        if not fault_facets:
            raise ValueError(f"No facets found with fault_id {self.fault_id}")
        
        # Convert to numpy arrays
        fault_normals = np.array(fault_normals)
        fault_centers = np.array(fault_centers)
        
        # Compute average normal vector
        avg_normal = np.mean(fault_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # Define fault-local coordinate system
        vertical = np.array([0, 0, 1])
        strike_vector = np.cross(avg_normal, vertical)
        strike_vector = strike_vector / np.linalg.norm(strike_vector)
        
        dip_vector = np.cross(strike_vector, avg_normal)
        dip_vector = dip_vector / np.linalg.norm(dip_vector)
        
        # Store coordinate system
        self.normal_vector = avg_normal
        self.strike_vector = strike_vector
        self.dip_vector = dip_vector
        
        print(f"Fault geometry computed:")
        print(f"  Normal vector: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]")
        print(f"  Strike vector: [{strike_vector[0]:.3f}, {strike_vector[1]:.3f}, {strike_vector[2]:.3f}]")
        print(f"  Dip vector: [{dip_vector[0]:.3f}, {dip_vector[1]:.3f}, {dip_vector[2]:.3f}]")
    
    def _global_to_fault_coords(self, x):
        """Transform global coordinates to fault-local coordinates"""
        s = np.dot([x[0], x[1], x[2]], self.strike_vector)
        d = np.dot([x[0], x[1], x[2]], self.dip_vector)
        n = np.dot([x[0], x[1], x[2]], self.normal_vector)
        
        return s, d, n
    
    def _rotate_fault_coords(self, s, d):
        """Rotate coordinates in the fault plane by specified angle"""
        # 2D rotation matrix in fault plane
        s_centered = s - self.x0
        d_centered = d - self.y0
        
        s_rot = self.cos_rot * s_centered - self.sin_rot * d_centered
        d_rot = self.sin_rot * s_centered + self.cos_rot * d_centered
        
        return s_rot, d_rot
    
    def _step_pattern(self, s_rot, d_rot):
        """Compute step function pattern value"""
        r_squared = s_rot**2 + d_rot**2
        if r_squared <= self.radius**2:
            return 1.0
        else:
            return 0.0
    
    def eval_cell(self, values, x, cell):
        """Evaluate the expression at a given point - same logic as Gaussian"""
        # Check if this cell is adjacent to the fault boundary
        cell_obj = dl.Cell(self.mesh, cell.index)
        is_on_fault = False
        
        for facet in dl.facets(cell_obj):
            if self.boundaries[facet] == self.fault_id:
                is_on_fault = True
                break
        
        if is_on_fault:
            # Apply additional depth constraint if specified
            if self.use_depth_constraint:
                depth_mask = 1.0 if (self.zmin <= x[2] <= self.zmax) else 0.0
            else:
                depth_mask = 1.0  # No additional depth constraint
            
            if depth_mask > 0:  # Only compute pattern if within depth range
                # Transform to fault-local coordinates
                s, d, n = self._global_to_fault_coords(x)
                
                # Apply rotation in fault-local coordinates
                s_rot, d_rot = self._rotate_fault_coords(s, d)
                
                # Create step pattern using rotated coordinates
                step_value = self._step_pattern(s_rot, d_rot)
                
                # Set slip components based on slip direction
                if self.strike_slip:
                    values[0] = self.amp * step_value  # strike slip
                    values[1] = 0.0  # no dip slip
                else:
                    values[0] = 0.0  # no strike slip
                    values[1] = self.amp * step_value  # dip slip
            else:
                # Outside depth range
                values[0] = 0.0
                values[1] = 0.0
        else:
            # Not on fault interface
            values[0] = 0.0
            values[1] = 0.0
    
    def value_shape(self):
        return (2,)  # Return 2D slip vector (strike, dip)


# Factory functions for convenience (similar to create_fault_local_checkerboard)
def create_fault_local_circular_gaussian(mesh, boundaries, fault_id, amp, x0=0, y0=0, 
                                        sigma=10e3, rotation_deg=0.0, zmin=None, zmax=None,
                                        strike_slip=False, **kwargs):
    """
    Factory function to create fault-local circular Gaussian expression
    
    Parameters:
    -----------
    mesh : dolfin.Mesh
        The computational mesh
    boundaries : dolfin.MeshFunction  
        Boundary markers
    fault_id : int
        ID of the fault boundary
    amp : float
        Amplitude of the Gaussian pattern
    x0, y0 : float, optional
        Center position in fault-local coordinates (default: 0)
    sigma : float, optional
        Gaussian width parameter (default: 10 km)
    rotation_deg : float, optional
        Rotation angle in degrees (default: 0)
    zmin, zmax : float, optional
        Additional depth constraints (default: None)
    strike_slip : bool, optional
        If True, creates strike-slip; if False, creates dip-slip (default: False)
        
    Returns:
    --------
    FaultLocalCircularGaussian expression
    """
    return FaultLocalCircularGaussian(
        mesh=mesh, boundaries=boundaries, fault_id=fault_id, amp=amp,
        x0=x0, y0=y0, sigma=sigma, rotation_deg=rotation_deg,
        zmin=zmin, zmax=zmax, strike_slip=strike_slip, **kwargs
    )


def create_fault_local_circular_step(mesh, boundaries, fault_id, amp, x0=0, y0=0,
                                   radius=20e3, rotation_deg=0.0, zmin=None, zmax=None,
                                   strike_slip=False, **kwargs):
    """
    Factory function to create fault-local circular step expression
    
    Parameters:
    -----------
    mesh : dolfin.Mesh
        The computational mesh
    boundaries : dolfin.MeshFunction
        Boundary markers  
    fault_id : int
        ID of the fault boundary
    amp : float
        Amplitude of the step pattern
    x0, y0 : float, optional
        Center position in fault-local coordinates (default: 0)
    radius : float, optional
        Radius of circular region (default: 20 km)
    rotation_deg : float, optional
        Rotation angle in degrees (default: 0)
    zmin, zmax : float, optional
        Additional depth constraints (default: None)
    strike_slip : bool, optional
        If True, creates strike-slip; if False, creates dip-slip (default: False)
        
    Returns:
    --------
    FaultLocalCircularStep expression
    """
    return FaultLocalCircularStep(
        mesh=mesh, boundaries=boundaries, fault_id=fault_id, amp=amp,
        x0=x0, y0=y0, radius=radius, rotation_deg=rotation_deg,
        zmin=zmin, zmax=zmax, strike_slip=strike_slip, **kwargs
    )


# Example usage patterns to replace the old classes:
"""
# OLD WAY (creates volumetric pattern - WRONG):
mtrue_s_expr_dip = CircularGaussianS(x0=x0, y0=y0, sigma=sigma, maxamp=maxamp, degree=3)
mtrue_s_expr_gt = dl.Expression(('0.', 'mask1'), mask1=mtrue_s_expr_dip, degree=3)

# NEW WAY (creates fault-confined pattern - CORRECT):
mtrue_s_expr_gt = create_fault_local_circular_gaussian(
    mesh=mesh,
    boundaries=boundaries,
    fault_id=fault,
    amp=maxamp,
    x0=x0,
    y0=y0,
    sigma=sigma,
    strike_slip=False,  # Creates dip-slip like the old version
    degree=3
)

# OLD WAY (creates volumetric pattern - WRONG):
mtrue_s_expr_dip = CircularStepS(x0=x0, y0=y0, R=radius, A=maxamp, degree=1)
mtrue_s_expr_gt = dl.Expression(('0.', 'mask1'), mask1=mtrue_s_expr_dip, degree=1)

# NEW WAY (creates fault-confined pattern - CORRECT):
mtrue_s_expr_gt = create_fault_local_circular_step(
    mesh=mesh,
    boundaries=boundaries,
    fault_id=fault,
    amp=maxamp,
    x0=x0,
    y0=y0,
    radius=radius,
    strike_slip=False,  # Creates dip-slip like the old version
    degree=1
)
"""

class FaultLocalCheckerboard(dl.UserExpression):
    def __init__(self, mesh, boundaries, fault_id, amp, dx, dy, x0=0, y0=0, 
                 rotation_deg=0.0, zmin=None, zmax=None, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.boundaries = boundaries
        self.fault_id = fault_id
        self.amp = amp
        self.omega_s = np.pi / dx  # frequency along strike
        self.omega_d = np.pi / dy  # frequency up-dip
        self.s0 = x0  # offset along strike
        self.d0 = y0  # offset up-dip
        
        # Pattern rotation in fault-local coordinates
        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.cos_rot = np.cos(self.rotation_rad)
        self.sin_rot = np.sin(self.rotation_rad)
        
        if abs(rotation_deg) > 1e-10:
            print(f"Pattern rotation: {rotation_deg:.1f}° (counterclockwise) in fault-local coordinates")
        
        # Optional depth constraints
        self.use_depth_constraint = (zmin is not None) and (zmax is not None)
        if self.use_depth_constraint:
            self.zmin = zmin
            self.zmax = zmax
            print(f"Using additional depth constraints: {zmin/1000:.1f} to {zmax/1000:.1f} km")
        else:
            print("Using natural fault interface depth range from mesh")
        
        # Compute fault geometry from mesh
        self._compute_fault_geometry()
        
    def _compute_fault_geometry(self):
        """Extract fault strike and dip from mesh geometry"""
        print("Computing fault geometry from mesh...")
        
        # Collect fault boundary facets and their properties
        fault_facets = []
        fault_normals = []
        fault_centers = []
        
        # Iterate through all facets
        for facet in dl.facets(self.mesh):
            if self.boundaries[facet] == self.fault_id:
                fault_facets.append(facet)
                
                # Get facet normal
                normal = facet.normal()
                fault_normals.append([normal.x(), normal.y(), normal.z()])
                
                # Get facet center
                center = facet.midpoint()
                fault_centers.append([center.x(), center.y(), center.z()])
        
        fault_normals = np.array(fault_normals)
        fault_centers = np.array(fault_centers)
        
        print(f"Found {len(fault_facets)} fault facets")
        
        # Print fault depth range from mesh
        fault_depths = fault_centers[:, 2]
        print(f"Fault depth range from mesh: {fault_depths.min()/1000:.1f} to {fault_depths.max()/1000:.1f} km")
        
        # Compute average fault normal
        avg_normal = np.mean(fault_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)  # normalize
        
        print(f"Average fault normal: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]")
        
        # Compute fault strike and dip from normal vector
        horizontal_normal = np.array([avg_normal[0], avg_normal[1], 0])
        horizontal_normal_mag = np.linalg.norm(horizontal_normal)
        
        if horizontal_normal_mag > 1e-10:  # fault is not vertical
            strike_vector = np.array([-avg_normal[1], avg_normal[0], 0])
            strike_vector = strike_vector / np.linalg.norm(strike_vector)
        else:  # nearly vertical fault
            strike_vector = np.array([1, 0, 0])  # arbitrary horizontal direction
        
        # Dip direction (down-dip vector)
        dip_vector = np.cross(strike_vector, avg_normal)
        dip_vector = dip_vector / np.linalg.norm(dip_vector)
        
        # Ensure dip vector points downward (negative z component)
        if dip_vector[2] > 0:
            dip_vector = -dip_vector
            
        # Store the coordinate transformation vectors
        self.strike_vector = strike_vector  # along-strike direction
        self.dip_vector = dip_vector        # up-dip direction  
        self.normal_vector = avg_normal     # fault-normal direction
        
        # Compute strike angle (from north)
        self.strike_angle = np.arctan2(strike_vector[0], strike_vector[1])  # radians from north
        
        # Compute dip angle (from horizontal)
        horizontal_dip_mag = np.sqrt(dip_vector[0]**2 + dip_vector[1]**2)
        self.dip_angle = np.arctan2(-dip_vector[2], horizontal_dip_mag)  # radians from horizontal
        
        print(f"Computed fault strike: {np.rad2deg(self.strike_angle):.1f}° from north")
        print(f"Computed fault dip: {np.rad2deg(self.dip_angle):.1f}° from horizontal")
        print(f"Strike vector: [{self.strike_vector[0]:.3f}, {self.strike_vector[1]:.3f}, {self.strike_vector[2]:.3f}]")
        print(f"Dip vector: [{self.dip_vector[0]:.3f}, {self.dip_vector[1]:.3f}, {self.dip_vector[2]:.3f}]")
        
    def _global_to_fault_coords(self, x):
        """Transform global coordinates to fault-local coordinates"""
        s = np.dot([x[0], x[1], x[2]], self.strike_vector)
        d = np.dot([x[0], x[1], x[2]], self.dip_vector)
        n = np.dot([x[0], x[1], x[2]], self.normal_vector)
        
        return s, d, n
    
    def _rotate_fault_coords(self, s, d):
        """
        Rotate coordinates in the fault plane by specified angle
        
        Parameters:
        -----------
        s, d : float
            Along-strike and up-dip coordinates
            
        Returns:
        --------
        s_rot, d_rot : float
            Rotated coordinates (counterclockwise rotation)
        """
        # 2D rotation matrix in fault plane:
        # [s_rot]   [cos(θ)  -sin(θ)] [s - s0]
        # [d_rot] = [sin(θ)   cos(θ)] [d - d0]
        s_centered = s - self.s0
        d_centered = d - self.d0
        
        s_rot = self.cos_rot * s_centered - self.sin_rot * d_centered
        d_rot = self.sin_rot * s_centered + self.cos_rot * d_centered
        
        return s_rot, d_rot
        
    def eval_cell(self, values, x, cell):
        """Evaluate the expression at a given point"""
        # Check if this cell is adjacent to the fault boundary
        cell_obj = dl.Cell(self.mesh, cell.index)
        is_on_fault = False
        
        for facet in dl.facets(cell_obj):
            if self.boundaries[facet] == self.fault_id:
                is_on_fault = True
                break
        
        if is_on_fault:
            # Apply additional depth constraint if specified
            if self.use_depth_constraint:
                depth_mask = 1.0 if (self.zmin <= x[2] <= self.zmax) else 0.0
            else:
                depth_mask = 1.0  # No additional depth constraint - use natural fault range
            
            if depth_mask > 0:  # Only compute pattern if within depth range
                # Transform to fault-local coordinates
                s, d, n = self._global_to_fault_coords(x)
                
                # Apply rotation in fault-local coordinates
                s_rot, d_rot = self._rotate_fault_coords(s, d)
                
                # Create checkerboard pattern using rotated coordinates
                pattern_s = 0.5 * (np.sin(self.omega_s * s_rot) + 1.0)
                pattern_d = 0.5 * (np.sin(self.omega_d * d_rot) + 1.0)
                checkerboard = pattern_s * pattern_d
                
                # Set slip components
                values[0] = 0.0  # no strike slip
                values[1] = self.amp * checkerboard  # dip slip
            else:
                # Outside depth range
                values[0] = 0.0
                values[1] = 0.0
        else:
            # Not on fault - zero slip
            values[0] = 0.0
            values[1] = 0.0
    
    def value_shape(self):
        return (2,)

# Updated factory function with rotation option
def create_fault_local_checkerboard(mesh, boundaries, fault_id, amp, dx, dy, 
                                   x0=0, y0=0, rotation_deg=0.0, 
                                   zmin=None, zmax=None, **kwargs):
    """
    Factory function to create fault-local checkerboard expression with rotation
    
    Parameters:
    -----------
    mesh : dolfin.Mesh
        The computational mesh
    boundaries : dolfin.MeshFunction
        Boundary markers
    fault_id : int
        ID of the fault boundary
    amp : float
        Amplitude of the checkerboard pattern
    dx, dy : float
        Grid spacing in fault-local coordinates (along-strike, up-dip)
    x0, y0 : float, optional
        Pattern offset in fault-local coordinates (default: 0)
    rotation_deg : float, optional
        Rotation angle in degrees (counterclockwise positive) in fault-local coordinates (default: 0)
    zmin, zmax : float, optional
        Additional depth constraints. If None, uses natural fault range from mesh
    
    Returns:
    --------
    FaultLocalCheckerboard expression
    """
    return FaultLocalCheckerboard(
        mesh=mesh,
        boundaries=boundaries, 
        fault_id=fault_id,
        amp=amp,
        dx=dx,
        dy=dy,
        x0=x0,
        y0=y0,
        rotation_deg=rotation_deg,
        zmin=zmin,
        zmax=zmax,
        **kwargs
    )

    # # Usage examples with rotation:

    # # Example 1: No rotation (default)
    # print("Creating fault-local checkerboard with no rotation...")
    # fault_checkerboard_0deg = create_fault_local_checkerboard(
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault,
    #     amp=amp,
    #     dx=40e3,
    #     dy=40e3,
    #     x0=-20e3,
    #     y0=-20e3,
    #     rotation_deg=0.0,  # No rotation
    #     degree=5
    # )

    # # Example 2: 45-degree counterclockwise rotation
    # print("Creating fault-local checkerboard with 45° rotation...")
    # fault_checkerboard_45deg = create_fault_local_checkerboard(
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault,
    #     amp=amp,
    #     dx=40e3,
    #     dy=40e3,
    #     x0=-20e3,
    #     y0=-20e3,
    #     rotation_deg=45.0,  # 45° counterclockwise
    #     degree=5
    # )


class FaultLocalStripes(dl.UserExpression):
    def __init__(self, mesh, boundaries, fault_id, amp, stripe_width, stripe_spacing, 
                 stripe_length, x0=0, y0=0, rotation_deg=0.0, zmin=None, zmax=None, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.boundaries = boundaries
        self.fault_id = fault_id
        self.amp = amp
        
        # Stripe parameters
        self.stripe_width = stripe_width      # Width of each stripe
        self.stripe_spacing = stripe_spacing  # Center-to-center spacing between stripes
        self.stripe_length = stripe_length    # Length of each stripe (perpendicular to stripe direction)
        self.s0 = x0  # offset along strike
        self.d0 = y0  # offset up-dip
        
        # Validate stripe parameters
        if stripe_width >= stripe_spacing:
            print(f"WARNING: stripe_width ({stripe_width/1000:.1f} km) >= stripe_spacing ({stripe_spacing/1000:.1f} km)")
            print("This will create overlapping or continuous stripes")
        
        gap_width = stripe_spacing - stripe_width
        print(f"Stripe pattern parameters:")
        print(f"  Stripe width: {stripe_width/1000:.1f} km")
        print(f"  Stripe spacing (center-to-center): {stripe_spacing/1000:.1f} km")
        print(f"  Gap width: {gap_width/1000:.1f} km")
        print(f"  Stripe length: {stripe_length/1000:.1f} km")
        
        # Pattern rotation in fault-local coordinates
        # Convention: 0° = stripes along strike, 90° = stripes along dip
        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.cos_rot = np.cos(self.rotation_rad)
        self.sin_rot = np.sin(self.rotation_rad)
        
        if abs(rotation_deg) > 1e-10:
            print(f"Pattern rotation: {rotation_deg:.1f}° (counterclockwise) from strike direction")
            if abs(rotation_deg % 90) < 1e-6:
                if abs(rotation_deg % 180) < 1e-6:
                    print("  -> Stripes along strike direction")
                else:
                    print("  -> Stripes along dip direction")
        else:
            print("Pattern orientation: Stripes along strike direction (0° rotation)")
        
        # Optional depth constraints
        self.use_depth_constraint = (zmin is not None) and (zmax is not None)
        if self.use_depth_constraint:
            self.zmin = zmin
            self.zmax = zmax
            print(f"Using additional depth constraints: {zmin/1000:.1f} to {zmax/1000:.1f} km")
        else:
            print("Using natural fault interface depth range from mesh")
        
        # Compute fault geometry from mesh
        self._compute_fault_geometry()
        
    def _compute_fault_geometry(self):
        """Extract fault strike and dip from mesh geometry"""
        print("Computing fault geometry from mesh...")
        
        # Collect fault boundary facets and their properties
        fault_facets = []
        fault_normals = []
        fault_centers = []
        
        # Iterate through all facets
        for facet in dl.facets(self.mesh):
            if self.boundaries[facet] == self.fault_id:
                fault_facets.append(facet)
                
                # Get facet normal
                normal = facet.normal()
                fault_normals.append([normal.x(), normal.y(), normal.z()])
                
                # Get facet center
                center = facet.midpoint()
                fault_centers.append([center.x(), center.y(), center.z()])
        
        fault_normals = np.array(fault_normals)
        fault_centers = np.array(fault_centers)
        
        print(f"Found {len(fault_facets)} fault facets")
        
        # Print fault depth range from mesh
        fault_depths = fault_centers[:, 2]
        print(f"Fault depth range from mesh: {fault_depths.min()/1000:.1f} to {fault_depths.max()/1000:.1f} km")
        
        # Compute average fault normal
        avg_normal = np.mean(fault_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)  # normalize
        
        print(f"Average fault normal: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]")
        
        # Compute fault strike and dip from normal vector
        horizontal_normal = np.array([avg_normal[0], avg_normal[1], 0])
        horizontal_normal_mag = np.linalg.norm(horizontal_normal)
        
        if horizontal_normal_mag > 1e-10:  # fault is not vertical
            strike_vector = np.array([-avg_normal[1], avg_normal[0], 0])
            strike_vector = strike_vector / np.linalg.norm(strike_vector)
        else:  # nearly vertical fault
            strike_vector = np.array([1, 0, 0])  # arbitrary horizontal direction
        
        # Dip direction (down-dip vector)
        dip_vector = np.cross(strike_vector, avg_normal)
        dip_vector = dip_vector / np.linalg.norm(dip_vector)
        
        # Ensure dip vector points downward (negative z component)
        if dip_vector[2] > 0:
            dip_vector = -dip_vector
            
        # Store the coordinate transformation vectors
        self.strike_vector = strike_vector  # along-strike direction
        self.dip_vector = dip_vector        # up-dip direction  
        self.normal_vector = avg_normal     # fault-normal direction
        
        # Compute strike angle (from north)
        self.strike_angle = np.arctan2(strike_vector[0], strike_vector[1])  # radians from north
        
        # Compute dip angle (from horizontal)
        horizontal_dip_mag = np.sqrt(dip_vector[0]**2 + dip_vector[1]**2)
        self.dip_angle = np.arctan2(-dip_vector[2], horizontal_dip_mag)  # radians from horizontal
        
        print(f"Computed fault strike: {np.rad2deg(self.strike_angle):.1f}° from north")
        print(f"Computed fault dip: {np.rad2deg(self.dip_angle):.1f}° from horizontal")
        print(f"Strike vector: [{self.strike_vector[0]:.3f}, {self.strike_vector[1]:.3f}, {self.strike_vector[2]:.3f}]")
        print(f"Dip vector: [{self.dip_vector[0]:.3f}, {self.dip_vector[1]:.3f}, {self.dip_vector[2]:.3f}]")
        
    def _global_to_fault_coords(self, x):
        """Transform global coordinates to fault-local coordinates"""
        s = np.dot([x[0], x[1], x[2]], self.strike_vector)
        d = np.dot([x[0], x[1], x[2]], self.dip_vector)
        n = np.dot([x[0], x[1], x[2]], self.normal_vector)
        
        return s, d, n
    
    def _rotate_fault_coords(self, s, d):
        """
        Rotate coordinates in the fault plane by specified angle
        Convention: 0° = along strike, 90° = along dip
        
        Parameters:
        -----------
        s, d : float
            Along-strike and up-dip coordinates
            
        Returns:
        --------
        s_rot, d_rot : float
            Rotated coordinates (counterclockwise rotation from strike direction)
        """
        # 2D rotation matrix in fault plane:
        # [s_rot]   [cos(θ)  -sin(θ)] [s - s0]
        # [d_rot] = [sin(θ)   cos(θ)] [d - d0]
        s_centered = s - self.s0
        d_centered = d - self.d0
        
        s_rot = self.cos_rot * s_centered - self.sin_rot * d_centered
        d_rot = self.sin_rot * s_centered + self.cos_rot * d_centered
        
        return s_rot, d_rot
    
    def _stripe_pattern(self, s_rot, d_rot):
        """
        Create stripe pattern in rotated fault-local coordinates
        For rotation = 0°: stripes run along strike (primary variation in dip direction)
        For rotation = 90°: stripes run along dip (primary variation in strike direction)
        
        Parameters:
        -----------
        s_rot, d_rot : float
            Rotated fault-local coordinates
            
        Returns:
        --------
        float : Pattern value (0 or 1)
        """
        # For 0° rotation (stripes along strike):
        # - Stripes extend along s_rot direction (strike)
        # - Periodic variation in d_rot direction (dip)
        # - stripe_length limits extent in s_rot direction
        
        # Use modulo to create repeating pattern in the dip direction (d_rot)
        d_mod = abs(d_rot) % self.stripe_spacing
        
        # Check if we're inside a stripe (centered around 0 in dip direction)
        stripe_half_width = self.stripe_width / 2.0
        in_stripe_d = (d_mod <= stripe_half_width) or (d_mod >= (self.stripe_spacing - stripe_half_width))
        
        # Check if we're within the stripe length in the strike direction (s_rot)
        stripe_half_length = self.stripe_length / 2.0
        in_stripe_s = abs(s_rot) <= stripe_half_length
        
        # Pattern is 1 if both conditions are met
        return 1.0 if (in_stripe_d and in_stripe_s) else 0.0
        
    def eval_cell(self, values, x, cell):
        """Evaluate the expression at a given point"""
        # Check if this cell is adjacent to the fault boundary
        cell_obj = dl.Cell(self.mesh, cell.index)
        is_on_fault = False
        
        for facet in dl.facets(cell_obj):
            if self.boundaries[facet] == self.fault_id:
                is_on_fault = True
                break
        
        if is_on_fault:
            # Apply additional depth constraint if specified
            if self.use_depth_constraint:
                depth_mask = 1.0 if (self.zmin <= x[2] <= self.zmax) else 0.0
            else:
                depth_mask = 1.0  # No additional depth constraint - use natural fault range
            
            if depth_mask > 0:  # Only compute pattern if within depth range
                # Transform to fault-local coordinates
                s, d, n = self._global_to_fault_coords(x)
                
                # Apply rotation in fault-local coordinates
                s_rot, d_rot = self._rotate_fault_coords(s, d)
                
                # Create stripe pattern using rotated coordinates
                stripe_value = self._stripe_pattern(s_rot, d_rot)
                
                # Set slip components
                values[0] = 0.0  # no strike slip
                values[1] = self.amp * stripe_value  # dip slip
            else:
                # Outside depth range
                values[0] = 0.0
                values[1] = 0.0
        else:
            # Not on fault - zero slip
            values[0] = 0.0
            values[1] = 0.0
    
    def value_shape(self):
        return (2,)

# Factory function for creating fault-local stripe patterns
def create_fault_local_stripes(mesh, boundaries, fault_id, amp, stripe_width, stripe_spacing, 
                              stripe_length, x0=0, y0=0, rotation_deg=0.0, 
                              zmin=None, zmax=None, **kwargs):
    """
    Factory function to create fault-local stripe expression with rotation
    
    Parameters:
    -----------
    mesh : dolfin.Mesh
        The computational mesh
    boundaries : dolfin.MeshFunction
        Boundary markers
    fault_id : int
        ID of the fault boundary
    amp : float
        Amplitude of the stripe pattern
    stripe_width : float
        Width of each stripe (perpendicular to stripe direction)
    stripe_spacing : float
        Center-to-center spacing between stripes
    stripe_length : float
        Length of each stripe (along stripe direction)
    x0, y0 : float, optional
        Pattern offset in fault-local coordinates (default: 0)
    rotation_deg : float, optional
        Rotation angle in degrees (counterclockwise from strike direction)
        0° = stripes along strike, 90° = stripes along dip (default: 0)
    zmin, zmax : float, optional
        Additional depth constraints. If None, uses natural fault range from mesh
    
    Returns:
    --------
    FaultLocalStripes expression
    
    Notes:
    ------
    Rotation convention:
    - 0°: Stripes run along fault strike direction
    - 90°: Stripes run along fault dip direction  
    - 45°: Diagonal stripes
    - Positive angles: counterclockwise rotation from strike direction
    """
    return FaultLocalStripes(
        mesh=mesh,
        boundaries=boundaries, 
        fault_id=fault_id,
        amp=amp,
        stripe_width=stripe_width,
        stripe_spacing=stripe_spacing,
        stripe_length=stripe_length,
        x0=x0,
        y0=y0,
        rotation_deg=rotation_deg,
        zmin=zmin,
        zmax=zmax,
        **kwargs
    )

    # # Usage examples with corrected rotation convention:

    # # Example 1: Stripes along strike (0° rotation) - DEFAULT
    # print("Creating stripes along fault strike direction (0° rotation)...")
    # fault_stripes_along_strike = create_fault_local_stripes(
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault,
    #     amp=amp,
    #     stripe_width=20e3,      # 20 km wide stripes
    #     stripe_spacing=50e3,    # 50 km center-to-center spacing
    #     stripe_length=200e3,    # 200 km long stripes (along strike)
    #     x0=0,
    #     y0=0,
    #     rotation_deg=0.0,       # Stripes along strike
    #     degree=5
    # )

    # # Example 2: Diagonal stripes (45° rotation)
    # print("Creating diagonal stripes (45° rotation)...")
    # fault_stripes_diagonal = create_fault_local_stripes(
    #     mesh=mesh,
    #     boundaries=boundaries,
    #     fault_id=fault,
    #     amp=amp,
    #     stripe_width=25e3,      # 25 km wide stripes
    #     stripe_spacing=60e3,    # 60 km spacing
    #     stripe_length=180e3,    # 180 km long stripes
    #     x0=-10e3,
    #     y0=-10e3,
    #     rotation_deg=45.0,      # Diagonal stripes
    #     degree=5
    # )


def _compute_az_per_facet_coeffs(mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg):
    """
    Precompute per-facet (c_strike, c_dip) coefficients for a prescribed geographic azimuth.

    For each fault facet, finds the unit vector v̂ that:
      - lies in the fault plane  (v̂ · n̂ = 0)
      - has its horizontal projection parallel to the prescribed azimuth direction

    Then decomposes v̂ into the facet's local (strike, dip) basis — matching exactly the
    convention used by dir_strike / dir_dip in the weak form — and returns the coefficients.

    Convention for dir_strike / dir_dip (must match weak form exactly):
        strike = (n × ẑ) / |n × ẑ|
        dip    = strike × n

    Parameters
    ----------
    mesh : dolfin.Mesh
    boundaries : dolfin.MeshFunction
    fault_id : int
    azimuth_deg : float
        Geographic azimuth of desired slip direction, degrees clockwise from North.
        E.g. 45 = N45E, 0 = North, 90 = East.
    mesh_rotation_deg : float
        CCW rotation (degrees) applied when building the mesh from geographic coordinates.
        For Nicoya: 45.  The mesh x-axis points in the direction (azimuth = mesh_rotation_deg).

    Returns
    -------
    dict  {facet_index -> (c_strike, c_dip)}
    """
    # Convert prescribed geographic azimuth to a horizontal unit vector in mesh coords.
    # Geographic unit vector for azimuth α (CW from N): (East=sin α, North=cos α).
    # Mesh is rotated CCW by mesh_rotation_deg relative to geographic, so:
    #   x_mesh = East·cos(rot) + North·sin(rot) = sin(α+rot)
    #   y_mesh = −East·sin(rot) + North·cos(rot) = cos(α+rot)
    alpha = np.deg2rad(azimuth_deg)
    rot   = np.deg2rad(mesh_rotation_deg)
    h = np.array([np.sin(alpha + rot), np.cos(alpha + rot), 0.0])

    z_hat = np.array([0., 0., 1.])

    # First pass: compute average fault normal to establish a reference orientation.
    # facet.normal() for interior facets may point in either direction; we normalise
    # each per-facet normal to be on the same side as the average so that the
    # strike direction (which flips with normal sign) is consistent with n('+') in
    # the weak form.  The dip direction is invariant to normal sign, but strike is not.
    normals_raw = []
    for facet in dl.facets(mesh):
        if boundaries[facet] == fault_id:
            fn = facet.normal()
            normals_raw.append([fn.x(), fn.y(), fn.z()])
    normals_raw = np.array(normals_raw)
    avg_normal = np.mean(normals_raw, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)

    coeffs = {}

    for facet in dl.facets(mesh):
        if boundaries[facet] != fault_id:
            continue

        fn = facet.normal()
        n = np.array([fn.x(), fn.y(), fn.z()])

        # Enforce consistent orientation: flip if opposite to average normal
        if np.dot(n, avg_normal) < 0:
            n = -n

        # --- local strike/dip basis (mirrors dir_strike / dir_dip in the weak form) ---
        n_cross_z = np.cross(n, z_hat)   # n × ẑ
        mag = np.linalg.norm(n_cross_z)
        if mag > 1e-10:
            strike_vec = n_cross_z / mag
        else:
            strike_vec = np.array([1., 0., 0.])  # nearly horizontal fault

        dip_vec = np.cross(strike_vec, n)
        dip_vec = dip_vec / np.linalg.norm(dip_vec)

        # --- in-plane vector whose horizontal projection is along h ---
        # v = (h_x, h_y, vz),  v·n = 0  →  vz = −(h_x·nx + h_y·ny) / nz
        nz = n[2]
        if abs(nz) > 1e-10:
            vz = -(h[0]*n[0] + h[1]*n[1]) / nz
            v = np.array([h[0], h[1], vz])
        else:
            # Nearly vertical fault: project h onto fault plane directly
            v = h - np.dot(h, n) * n

        v_mag = np.linalg.norm(v)
        v_hat = v / v_mag if v_mag > 1e-10 else dip_vec  # fallback: pure dip

        # Sign convention: positive m_dip in the weak form corresponds to up-dip (thrust)
        # slip of the hanging wall, which is the anti-convergence direction (SW+up for Nicoya).
        # The prescribed azimuth is the plate convergence direction (e.g. N45E = NE footwall motion).
        # So the hanging wall slip direction = -v_hat, and we negate the coefficients accordingly.
        coeffs[facet.index()] = (-float(np.dot(v_hat, strike_vec)),
                                  -float(np.dot(v_hat, dip_vec)))

    return coeffs


class FaultLocalCheckerboardAz(dl.UserExpression):
    """
    Checkerboard slip pattern with slip direction prescribed by a geographic azimuth.

    Unlike FaultLocalCheckerboard (which always returns pure dip-slip), this class
    projects the prescribed horizontal azimuth onto each facet's local fault plane
    and decomposes the result into per-facet (strike, dip) coefficients.  The pattern
    shape (which facets get slip) is still laid out using the averaged fault-local frame.

    Parameters
    ----------
    mesh, boundaries, fault_id : standard FEniCS objects
    amp : float
        Slip amplitude (metres).
    dx, dy : float
        Checkerboard cell half-wavelength in the averaged fault-local (strike, dip) frame (metres).
    azimuth_deg : float
        Geographic azimuth of the desired slip direction, degrees CW from North.
    mesh_rotation_deg : float
        CCW rotation used when building the mesh from geographic coords (default 45 for Nicoya).
    x0, y0 : float
        Pattern offset in averaged fault-local coordinates.
    rotation_deg : float
        In-plane rotation of the checkerboard pattern (deg, CCW from strike).
    zmin, zmax : float or None
        Optional depth constraints (metres, negative down).
    """
    def __init__(self, mesh, boundaries, fault_id, amp, dx, dy,
                 azimuth_deg, mesh_rotation_deg=45.0,
                 x0=0, y0=0, rotation_deg=0.0,
                 zmin=None, zmax=None, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.boundaries = boundaries
        self.fault_id = fault_id
        self.amp = amp
        self.omega_s = np.pi / dx
        self.omega_d = np.pi / dy
        self.s0 = x0
        self.d0 = y0

        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.cos_rot = np.cos(self.rotation_rad)
        self.sin_rot = np.sin(self.rotation_rad)

        self.use_depth_constraint = (zmin is not None) and (zmax is not None)
        if self.use_depth_constraint:
            self.zmin = zmin
            self.zmax = zmax

        print(f"Prescribed slip azimuth: {azimuth_deg:.1f}° CW from North "
              f"(mesh rotation {mesh_rotation_deg:.1f}° CCW)")

        # Averaged fault frame — used only for pattern geometry
        self._compute_fault_geometry()

        # Per-facet slip direction coefficients
        self._facet_coeffs = _compute_az_per_facet_coeffs(
            mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg)
        print(f"Per-facet slip coefficients computed for {len(self._facet_coeffs)} facets")

    def _compute_fault_geometry(self):
        """Averaged fault frame for pattern layout (identical to FaultLocalCheckerboard)."""
        fault_normals, fault_centers = [], []
        for facet in dl.facets(self.mesh):
            if self.boundaries[facet] == self.fault_id:
                n = facet.normal()
                fault_normals.append([n.x(), n.y(), n.z()])
                c = facet.midpoint()
                fault_centers.append([c.x(), c.y(), c.z()])

        fault_normals  = np.array(fault_normals)
        fault_centers  = np.array(fault_centers)
        print(f"Found {len(fault_normals)} fault facets")
        print(f"Fault depth range: {fault_centers[:,2].min()/1e3:.1f} to "
              f"{fault_centers[:,2].max()/1e3:.1f} km")

        avg_normal = np.mean(fault_normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)

        horiz_mag = np.linalg.norm(avg_normal[:2])
        if horiz_mag > 1e-10:
            strike_vec = np.array([-avg_normal[1], avg_normal[0], 0.])
            strike_vec /= np.linalg.norm(strike_vec)
        else:
            strike_vec = np.array([1., 0., 0.])

        dip_vec = np.cross(strike_vec, avg_normal)
        dip_vec /= np.linalg.norm(dip_vec)
        if dip_vec[2] > 0:
            dip_vec = -dip_vec

        self.strike_vector = strike_vec
        self.dip_vector    = dip_vec
        self.normal_vector = avg_normal

    def _global_to_fault_coords(self, x):
        s = np.dot(x[:3], self.strike_vector)
        d = np.dot(x[:3], self.dip_vector)
        return s, d

    def _rotate_fault_coords(self, s, d):
        sc = s - self.s0
        dc = d - self.d0
        return (self.cos_rot*sc - self.sin_rot*dc,
                self.sin_rot*sc + self.cos_rot*dc)

    def eval_cell(self, values, x, cell):
        cell_obj = dl.Cell(self.mesh, cell.index)
        facet_key = None
        for facet in dl.facets(cell_obj):
            if self.boundaries[facet] == self.fault_id:
                facet_key = facet.index()
                break

        if facet_key is None:
            values[0] = values[1] = 0.0
            return

        if self.use_depth_constraint and not (self.zmin <= x[2] <= self.zmax):
            values[0] = values[1] = 0.0
            return

        s, d = self._global_to_fault_coords(x)
        s_rot, d_rot = self._rotate_fault_coords(s, d)
        pattern = (0.5*(np.sin(self.omega_s*s_rot) + 1.0) *
                   0.5*(np.sin(self.omega_d*d_rot) + 1.0))

        c_strike, c_dip = self._facet_coeffs.get(facet_key, (0.0, 1.0))
        values[0] = self.amp * pattern * c_strike
        values[1] = self.amp * pattern * c_dip

    def value_shape(self):
        return (2,)


class FaultLocalStripesAz(dl.UserExpression):
    """
    Stripe slip pattern with slip direction prescribed by a geographic azimuth.

    Unlike FaultLocalStripes (which always returns pure dip-slip), this class
    projects the prescribed horizontal azimuth onto each facet's local fault plane
    and decomposes the result into per-facet (strike, dip) coefficients.  The pattern
    shape (which facets get slip) is still laid out using the averaged fault-local frame.

    Parameters
    ----------
    mesh, boundaries, fault_id : standard FEniCS objects
    amp : float
        Slip amplitude (metres).
    stripe_width, stripe_spacing, stripe_length : float
        Stripe geometry in averaged fault-local coordinates (metres).
        stripe_spacing is centre-to-centre distance; stripe_width < stripe_spacing.
    azimuth_deg : float
        Geographic azimuth of the desired slip direction, degrees CW from North.
    mesh_rotation_deg : float
        CCW rotation used when building the mesh from geographic coords (default 45 for Nicoya).
    x0, y0 : float
        Pattern offset in averaged fault-local coordinates.
    rotation_deg : float
        In-plane rotation of the stripe pattern (deg, CCW from strike).
        0° → stripes along strike; 90° → stripes along dip.
    zmin, zmax : float or None
        Optional depth constraints (metres, negative down).
    """
    def __init__(self, mesh, boundaries, fault_id, amp,
                 stripe_width, stripe_spacing, stripe_length,
                 azimuth_deg, mesh_rotation_deg=45.0,
                 x0=0, y0=0, rotation_deg=0.0,
                 zmin=None, zmax=None, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.boundaries = boundaries
        self.fault_id = fault_id
        self.amp = amp
        self.stripe_width   = stripe_width
        self.stripe_spacing = stripe_spacing
        self.stripe_length  = stripe_length
        self.s0 = x0
        self.d0 = y0

        if stripe_width >= stripe_spacing:
            print(f"WARNING: stripe_width ({stripe_width/1e3:.1f} km) >= "
                  f"stripe_spacing ({stripe_spacing/1e3:.1f} km) — overlapping stripes")
        print(f"Stripe parameters: width={stripe_width/1e3:.1f} km, "
              f"spacing={stripe_spacing/1e3:.1f} km, length={stripe_length/1e3:.1f} km")

        self.rotation_deg = rotation_deg
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.cos_rot = np.cos(self.rotation_rad)
        self.sin_rot = np.sin(self.rotation_rad)

        self.use_depth_constraint = (zmin is not None) and (zmax is not None)
        if self.use_depth_constraint:
            self.zmin = zmin
            self.zmax = zmax

        print(f"Prescribed slip azimuth: {azimuth_deg:.1f}° CW from North "
              f"(mesh rotation {mesh_rotation_deg:.1f}° CCW)")

        # Averaged fault frame — used only for pattern geometry
        self._compute_fault_geometry()

        # Per-facet slip direction coefficients
        self._facet_coeffs = _compute_az_per_facet_coeffs(
            mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg)
        print(f"Per-facet slip coefficients computed for {len(self._facet_coeffs)} facets")

    def _compute_fault_geometry(self):
        """Averaged fault frame for pattern layout (identical to FaultLocalStripes)."""
        fault_normals, fault_centers = [], []
        for facet in dl.facets(self.mesh):
            if self.boundaries[facet] == self.fault_id:
                n = facet.normal()
                fault_normals.append([n.x(), n.y(), n.z()])
                c = facet.midpoint()
                fault_centers.append([c.x(), c.y(), c.z()])

        fault_normals = np.array(fault_normals)
        fault_centers = np.array(fault_centers)
        print(f"Found {len(fault_normals)} fault facets")
        print(f"Fault depth range: {fault_centers[:,2].min()/1e3:.1f} to "
              f"{fault_centers[:,2].max()/1e3:.1f} km")

        avg_normal = np.mean(fault_normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)

        horiz_mag = np.linalg.norm(avg_normal[:2])
        if horiz_mag > 1e-10:
            strike_vec = np.array([-avg_normal[1], avg_normal[0], 0.])
            strike_vec /= np.linalg.norm(strike_vec)
        else:
            strike_vec = np.array([1., 0., 0.])

        dip_vec = np.cross(strike_vec, avg_normal)
        dip_vec /= np.linalg.norm(dip_vec)
        if dip_vec[2] > 0:
            dip_vec = -dip_vec

        self.strike_vector = strike_vec
        self.dip_vector    = dip_vec
        self.normal_vector = avg_normal

    def _global_to_fault_coords(self, x):
        s = np.dot(x[:3], self.strike_vector)
        d = np.dot(x[:3], self.dip_vector)
        return s, d

    def _rotate_fault_coords(self, s, d):
        sc = s - self.s0
        dc = d - self.d0
        return (self.cos_rot*sc - self.sin_rot*dc,
                self.sin_rot*sc + self.cos_rot*dc)

    def _stripe_pattern(self, s_rot, d_rot):
        d_mod = abs(d_rot) % self.stripe_spacing
        hw = self.stripe_width / 2.0
        in_d = (d_mod <= hw) or (d_mod >= (self.stripe_spacing - hw))
        in_s = abs(s_rot) <= self.stripe_length / 2.0
        return 1.0 if (in_d and in_s) else 0.0

    def eval_cell(self, values, x, cell):
        cell_obj = dl.Cell(self.mesh, cell.index)
        facet_key = None
        for facet in dl.facets(cell_obj):
            if self.boundaries[facet] == self.fault_id:
                facet_key = facet.index()
                break

        if facet_key is None:
            values[0] = values[1] = 0.0
            return

        if self.use_depth_constraint and not (self.zmin <= x[2] <= self.zmax):
            values[0] = values[1] = 0.0
            return

        s, d = self._global_to_fault_coords(x)
        s_rot, d_rot = self._rotate_fault_coords(s, d)
        stripe_value = self._stripe_pattern(s_rot, d_rot)

        c_strike, c_dip = self._facet_coeffs.get(facet_key, (0.0, 1.0))
        values[0] = self.amp * stripe_value * c_strike
        values[1] = self.amp * stripe_value * c_dip

    def value_shape(self):
        return (2,)


def create_fault_local_checkerboard_az(mesh, boundaries, fault_id, amp, dx, dy,
                                       azimuth_deg, mesh_rotation_deg=45.0,
                                       x0=0, y0=0, rotation_deg=0.0,
                                       zmin=None, zmax=None, **kwargs):
    """
    Factory for FaultLocalCheckerboardAz.

    Prescribes checkerboard slip whose physical direction on each facet is the
    in-plane vector whose horizontal projection points along `azimuth_deg` (CW from North).

    Parameters
    ----------
    azimuth_deg : float
        Geographic azimuth of slip direction, degrees CW from North (e.g. 45 = N45E).
    mesh_rotation_deg : float
        CCW rotation used when building the mesh (default 45 for Nicoya).
    dx, dy : float
        Checkerboard half-wavelength in averaged fault-local (strike, dip) coordinates (m).
    See FaultLocalCheckerboardAz for remaining parameters.
    """
    return FaultLocalCheckerboardAz(
        mesh=mesh, boundaries=boundaries, fault_id=fault_id,
        amp=amp, dx=dx, dy=dy,
        azimuth_deg=azimuth_deg, mesh_rotation_deg=mesh_rotation_deg,
        x0=x0, y0=y0, rotation_deg=rotation_deg,
        zmin=zmin, zmax=zmax, **kwargs)


def create_fault_local_stripes_az(mesh, boundaries, fault_id, amp,
                                   stripe_width, stripe_spacing, stripe_length,
                                   azimuth_deg, mesh_rotation_deg=45.0,
                                   x0=0, y0=0, rotation_deg=0.0,
                                   zmin=None, zmax=None, **kwargs):
    """
    Factory for FaultLocalStripesAz.

    Prescribes stripe slip whose physical direction on each facet is the
    in-plane vector whose horizontal projection points along `azimuth_deg` (CW from North).

    Parameters
    ----------
    azimuth_deg : float
        Geographic azimuth of slip direction, degrees CW from North (e.g. 45 = N45E).
    mesh_rotation_deg : float
        CCW rotation used when building the mesh (default 45 for Nicoya).
    stripe_width, stripe_spacing, stripe_length : float
        Stripe geometry in averaged fault-local coordinates (m).
    See FaultLocalStripesAz for remaining parameters.
    """
    return FaultLocalStripesAz(
        mesh=mesh, boundaries=boundaries, fault_id=fault_id,
        amp=amp,
        stripe_width=stripe_width, stripe_spacing=stripe_spacing,
        stripe_length=stripe_length,
        azimuth_deg=azimuth_deg, mesh_rotation_deg=mesh_rotation_deg,
        x0=x0, y0=y0, rotation_deg=rotation_deg,
        zmin=zmin, zmax=zmax, **kwargs)


import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import dolfin as dl

def get_layered_1d_value(mesh_depths, model_depths, model_values, side='right'):
    """
    Get values from layered 1D model (vectorized).

    Layer j covers the depth range (model_depths[j+1], model_depths[j]].

    At an exact layer boundary, behaviour depends on `side`:
    - 'right' (default): deeper layer value returned (larger value)
    - 'left':            shallower layer value returned (smaller value)

    Example with model_depths = [0, -5000, -9000], side='right':
    - z in (-5000, 0]       → model_values[0]
    - z in (-9000, -5000]   → model_values[1]   (z == -5000 → values[1])
    - z <= -9000            → model_values[2]

    Parameters
    ----------
    mesh_depths : array-like
        Depth values at mesh nodes (metres, negative downward).
    model_depths : array-like
        Layer node depths, sorted shallow to deep (e.g. [0, -5000, -9000, ...]).
    model_values : array-like
        Property value at each layer node.
    side : {'right', 'left'}, optional
        Boundary convention passed to np.searchsorted. Default 'right' (deeper layer).

    Returns
    -------
    result : ndarray
        Property values at mesh depths using step-function lookup.
    """
    neg_z        = -np.asarray(mesh_depths,  dtype=float)   # positive depths
    neg_model_z  = -np.asarray(model_depths, dtype=float)   # ascending
    model_values =  np.asarray(model_values, dtype=float)
    idx = np.searchsorted(neg_model_z, neg_z, side=side) - 1
    idx = np.clip(idx, 0, len(model_values) - 1)
    return model_values[idx]


def process_velocity_models(vel3d, vel1d, den1d, mesh, CG_mu_degree=1, verbose=True, side='right'):
    """
    Process 3D and 1D velocity models with density and project to FEniCS mesh.
    
    1D models are treated as layered (constant values within each layer).
    3D model uses interpolation between grid points.
    
    Parameters:
    -----------
    vel3d : pd.DataFrame
        3D velocity model with columns: x, y, z (m), vs (km/s)
    vel1d : pd.DataFrame  
        1D velocity model with columns: z (m), vs (km/s)
    den1d : pd.DataFrame
        1D density model with columns: z (m), den (kg/m³)
    mesh : dolfin.Mesh
        FEniCS mesh object
    verbose : bool, default=True
        If True, print detailed progress information
    
    Returns:
    --------
    vs_function : dolfin.Function
        Shear velocity field on mesh (km/s)
    density_function : dolfin.Function  
        Density field on mesh (kg/m³)
    shear_modulus_function : dolfin.Function
        Shear modulus field on mesh (GPa)
    CG_mu : dolfin.FunctionSpace
        Function space used
    """
    
    if verbose:
        print("Starting velocity model processing...")
    
    # Step 1: Clean the data - remove any NaN values
    vel3d_clean = vel3d.dropna(subset=['x', 'y', 'z', 'vs']).copy()
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).copy()
    den1d_clean = den1d.dropna(subset=['z', 'den']).copy()
    
    if verbose:
        print(f"Data after cleaning:")
        print(f"  3D model: {len(vel3d_clean)} points")
        print(f"  1D velocity: {len(vel1d_clean)} layers")
        print(f"  1D density: {len(den1d_clean)} layers")
    
    # Check if we have enough data
    if len(vel3d_clean) == 0:
        raise ValueError("No valid data points in 3D model after removing NaN values")
    if len(vel1d_clean) == 0:
        raise ValueError("No valid data points in 1D velocity model after removing NaN values")
    if len(den1d_clean) == 0:
        raise ValueError("No valid data points in 1D density model after removing NaN values")
    
    # Step 2: Sort 1D models by depth (shallow to deep: 0, -5000, -9000, ...)
    # This is CRITICAL for the layered model to work correctly
    vel1d_clean = vel1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"1D models sorted by depth (shallow to deep):")
        print(f"  Velocity depths: {vel1d_clean['z'].values[:5]} ... {vel1d_clean['z'].values[-2:]}")
        print(f"  Density depths: {den1d_clean['z'].values[:5]} ... {den1d_clean['z'].values[-2:]}")
    
    # Step 3: Extract 3D bounding box
    x_min, x_max = vel3d_clean['x'].min(), vel3d_clean['x'].max()
    y_min, y_max = vel3d_clean['y'].min(), vel3d_clean['y'].max()  
    z_min, z_max = vel3d_clean['z'].min(), vel3d_clean['z'].max()
    
    if verbose:
        print(f"3D model bounds:")
        print(f"  X: {x_min:.1f} to {x_max:.1f} m")
        print(f"  Y: {y_min:.1f} to {y_max:.1f} m") 
        print(f"  Z: {z_min:.1f} to {z_max:.1f} m")
    
    # Step 4: Get mesh node coordinates
    mesh_coords = mesh.coordinates()
    n_nodes = len(mesh_coords)
    
    if verbose:
        print(f"Mesh has {n_nodes} nodes")
        print(f"Mesh depth range: {mesh_coords[:, 2].min():.1f} to {mesh_coords[:, 2].max():.1f} m")
        print(f"1D model depth range: {vel1d_clean['z'].min():.1f} to {vel1d_clean['z'].max():.1f} m")
        
        # Check if mesh extends beyond 1D model
        if mesh_coords[:, 2].min() < vel1d_clean['z'].min():
            deepest_vs = vel1d_clean['vs'].iloc[-1]
            deepest_den = den1d_clean['den'].iloc[-1]
            print(f"Note: Mesh extends {vel1d_clean['z'].min() - mesh_coords[:, 2].min():.1f} m beyond deepest 1D layer")
            print(f"      These nodes will use deepest layer values (vs={deepest_vs:.3f} km/s, density={deepest_den:.1f} kg/m³)")
    
    # Step 5: Determine which nodes are inside 3D model domain
    inside_3d = ((mesh_coords[:, 0] >= x_min) & (mesh_coords[:, 0] <= x_max) &
                 (mesh_coords[:, 1] >= y_min) & (mesh_coords[:, 1] <= y_max) &
                 (mesh_coords[:, 2] >= z_min) & (mesh_coords[:, 2] <= z_max))
    
    n_inside = np.sum(inside_3d)
    n_outside = np.sum(~inside_3d)
    
    if verbose:
        print(f"Nodes inside 3D domain: {n_inside}")
        print(f"Nodes outside 3D domain: {n_outside}")
    
    # Step 6: Create FEniCS functions first (needed for DOF coordinates)
    if verbose:
        print("Creating FEniCS functions...")
    
    if CG_mu_degree != 1:
        print(f"Using CG function space of degree {CG_mu_degree} instead of default 1")
        
    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_degree)
    
    vs_function = dl.Function(CG_mu)
    density_function = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)
    
    # Get DOF coordinates (these are the actual function evaluation points)
    V_coords = CG_mu.tabulate_dof_coordinates()
    n_dofs = len(V_coords)
    
    if verbose:
        print(f"Function space CG{CG_mu_degree} has {n_dofs} DOFs")
        print(f"DOF depth range: {V_coords[:, 2].min():.1f} to {V_coords[:, 2].max():.1f} m")
    
    # Step 7: Apply 1D layered models to ALL DOFs (background)
    if verbose:
        print("Applying 1D layered models to all DOFs...")
    
    vs_at_dofs = get_layered_1d_value(
        V_coords[:, 2],
        vel1d_clean['z'].values,
        vel1d_clean['vs'].values,
        side=side
    )

    density_at_dofs = get_layered_1d_value(
        V_coords[:, 2],
        den1d_clean['z'].values,
        den1d_clean['den'].values,
        side=side
    )

    if verbose:
        print(f"  1D velocity range: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  1D density range: {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")

    # Step 8: Override with 3D model where available
    # Find which DOFs are inside 3D domain
    inside_3d_dofs = ((V_coords[:, 0] >= x_min) & (V_coords[:, 0] <= x_max) &
                      (V_coords[:, 1] >= y_min) & (V_coords[:, 1] <= y_max) &
                      (V_coords[:, 2] >= z_min) & (V_coords[:, 2] <= z_max))
    
    n_inside_dofs = np.sum(inside_3d_dofs)
    n_outside_dofs = np.sum(~inside_3d_dofs)
    
    if verbose:
        print(f"DOFs inside 3D domain: {n_inside_dofs}")
        print(f"DOFs outside 3D domain: {n_outside_dofs}")
    
    if n_inside_dofs > 0:
        if verbose:
            print("Overriding with 3D interpolated model...")
        
        # Interpolate 3D velocity using griddata
        vs_3d = griddata(
            points=vel3d_clean[['x', 'y', 'z']].values,
            values=vel3d_clean['vs'].values,
            xi=V_coords[inside_3d_dofs],
            method='linear',
            fill_value=np.nan
        )
        
        # Handle any NaN values with nearest neighbor
        nan_mask = np.isnan(vs_3d)
        if nan_mask.any():
            if verbose:
                print(f"    Filling {np.sum(nan_mask)} NaN values with nearest neighbor")
            else:
                print(f"Warning: Filled {np.sum(nan_mask)} NaN values with nearest neighbor")
            
            vs_3d[nan_mask] = griddata(
                points=vel3d_clean[['x', 'y', 'z']].values,
                values=vel3d_clean['vs'].values,
                xi=V_coords[inside_3d_dofs][nan_mask],
                method='nearest'
            )
        
        # Override 1D values with 3D interpolated values
        vs_at_dofs[inside_3d_dofs] = vs_3d
        # Density remains 1D layered everywhere (depth-dependent only)
        
        if verbose:
            print(f"  3D velocity range: {vs_at_dofs[inside_3d_dofs].min():.3f} to {vs_at_dofs[inside_3d_dofs].max():.3f} km/s")
    
    # Step 9: Assign values to FEniCS functions
    vs_function.vector()[:] = vs_at_dofs
    density_function.vector()[:] = density_at_dofs
    
    # Step 10: Debug verification (if verbose)
    if verbose:
        print("\nDebug: Verifying function values at test points...")
        test_coords = [
            [-1000, 0, -5000],
            [-20000, 0, -15000], 
            [-20000, 10000, -28000],
            [20000, -10000, -100000]
        ]
        
        for coord in test_coords:
            try:
                density_val = density_function(*coord)
                vs_val = vs_function(*coord)
                print(f"  Point {coord}: density = {density_val:.1f} kg/m³, vs = {vs_val:.3f} km/s")
            except:
                print(f"  Point {coord}: outside mesh domain")
        
        # Check layered consistency by depth ranges
        print("\nDebug: Checking density consistency by depth ranges...")
        depth_ranges = [(-5000, 0), (-15000, -10000), (-30000, -25000), (-100000, -50000)]
        
        for z_min, z_max in depth_ranges:
            mask = (V_coords[:, 2] >= z_min) & (V_coords[:, 2] < z_max)
            if np.any(mask):
                densities_in_range = density_at_dofs[mask]
                velocities_in_range = vs_at_dofs[mask]
                print(f"  Depth {z_min} to {z_max} m:")
                print(f"    Density: {densities_in_range.min():.1f} to {densities_in_range.max():.1f} kg/m³")
                print(f"    Velocity: {velocities_in_range.min():.3f} to {velocities_in_range.max():.3f} km/s")
    
    # Step 11: Compute shear modulus (μ = ρ × vs²)
    if verbose:
        print("Computing shear modulus field...")
    
    vs_ms = vs_at_dofs * 1000.0  # km/s to m/s
    shear_modulus_pa = density_at_dofs * vs_ms**2  # kg/m³ × (m/s)² = Pa
    shear_modulus_gpa = shear_modulus_pa / 1e9  # Pa to GPa
    
    shear_modulus_function.vector()[:] = shear_modulus_gpa
    
    # Step 12: Summary
    if verbose:
        print("\nProcessing complete! Summary:")
        print(f"Final velocity: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"Final density: {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")
        print(f"Final shear modulus: {shear_modulus_gpa.min():.2f} to {shear_modulus_gpa.max():.2f} GPa")
    
    return vs_function, density_function, shear_modulus_function, CG_mu


def get_layered_1d_value_interp(mesh_depths, model_depths, model_values):
    """
    Get values from 1D model using LINEAR INTERPOLATION.

    This ensures smooth variation with depth instead of sharp jumps at layer boundaries.
    Uses numpy.interp for fast vectorized linear interpolation.

    Interpolation behavior:
    - Points above shallowest depth: extrapolate using shallowest value (constant)
    - Points within model range: linear interpolation between layers
    - Points below deepest depth: extrapolate using deepest value (constant)

    Parameters:
    -----------
    mesh_depths : array
        Depth values at mesh nodes (can be any value, positive or negative)
    model_depths : array
        Depth values from 1D model (sorted shallow to deep: 0, -5000, -9000, ...)
        Must be sorted in descending order (shallowest to deepest)
    model_values : array
        Property values from 1D model

    Returns:
    --------
    result : array
        Property values at mesh depths using linear interpolation

    Example:
    --------
    model_depths = [0, -5000, -9000, -13000]
    model_values = [3.0, 3.5, 4.0, 4.5]
    mesh_depths = [100, -3000, -7000, -15000]

    Result: [3.0, 3.3, 3.75, 4.5]
    - 100 m (above ground) -> extrapolate: 3.0
    - -3000 m (between 0 and -5000) -> interpolate: 3.0 + 0.6 * (3.5-3.0) = 3.3
    - -7000 m (between -5000 and -9000) -> interpolate: 3.5 + 0.5 * (4.0-3.5) = 3.75
    - -15000 m (below deepest layer) -> extrapolate: 4.5
    """
    mesh_depths = np.asarray(mesh_depths)
    model_depths = np.asarray(model_depths)
    model_values = np.asarray(model_values)

    # Verify model_depths is sorted descending (shallow to deep)
    if not np.all(model_depths[:-1] >= model_depths[1:]):
        raise ValueError("model_depths must be sorted in descending order (shallow to deep)")

    if len(model_depths) != len(model_values):
        raise ValueError(f"model_depths and model_values must have same length, got {len(model_depths)} and {len(model_values)}")

    # np.interp requires x-coordinates (depths) in ASCENDING order
    # Our model_depths is descending (0, -5000, -9000, ...), so reverse both arrays
    result = np.interp(
        mesh_depths,                    # Query points
        model_depths[::-1],             # Reversed: ascending order (-9000, -5000, 0, ...)
        model_values[::-1],             # Reversed to match
        left=model_values[-1],          # Extrapolate below deepest layer
        right=model_values[0]           # Extrapolate above shallowest layer
    )

    return result


def signed_distance_to_box(points, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Compute signed distance from points to a rectangular box boundary.

    The signed distance is:
    - Negative when the point is INSIDE the box
    - Positive when the point is OUTSIDE the box
    - The magnitude represents the minimum distance to any face

    Parameters:
    -----------
    points : array, shape (n, 3)
        Array of 3D points [x, y, z]
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Box boundaries

    Returns:
    --------
    distances : array, shape (n,)
        Signed distances to box boundary
        Negative = inside box, Positive = outside box

    Example:
    --------
    Box: x=[0, 100], y=[0, 100], z=[-50, 0]
    Point [50, 50, -25] (center): distance = -25 (inside, 25m from top)
    Point [150, 50, -25] (outside): distance = 50 (outside, 50m from right face)
    """
    points = np.asarray(points)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Distance from each point to each face (positive = inside)
    dx_min = points[:, 0] - x_min  # Distance to left face
    dx_max = x_max - points[:, 0]  # Distance to right face
    dy_min = points[:, 1] - y_min  # Distance to front face
    dy_max = y_max - points[:, 1]  # Distance to back face
    dz_min = points[:, 2] - z_min  # Distance to bottom face
    dz_max = z_max - points[:, 2]  # Distance to top face

    # Minimum distance to any face
    # If all are positive, point is inside (return negative of smallest)
    # If any is negative, point is outside (return negative of that value)
    min_dist = np.minimum.reduce([dx_min, dx_max, dy_min, dy_max, dz_min, dz_max])

    # Convert to signed distance (negative inside, positive outside)
    signed_dist = -min_dist

    return signed_dist


def smooth_transition_weight(distance, width):
    """
    Compute smooth transition weights for blending 3D and 1D models.

    Uses a cosine taper to create smooth transition from 3D to 1D model.

    Parameters:
    -----------
    distance : array or float
        Signed distance from box boundary
        Negative = inside box, Positive = outside box
    width : float
        Total width of transition zone

    Returns:
    --------
    weight : array or float
        Blending weight in [0, 1]
        - 1.0 = pure 3D (deep inside box, distance < -width/2)
        - 0.0 = pure 1D (far outside box, distance > width/2)
        - Smooth cosine taper in between

    Example:
    --------
    width = 10000  # 10 km transition
    distance = [-8000, -3000, 0, 3000, 8000]
    weight = [1.0, 0.85, 0.5, 0.15, 0.0]  # approximate
    """
    distance = np.asarray(distance)
    half_width = width / 2.0

    # Allocate output
    weight = np.zeros_like(distance, dtype=float)

    # Deep inside: pure 3D
    inside_mask = distance < -half_width
    weight[inside_mask] = 1.0

    # Far outside: pure 1D
    outside_mask = distance > half_width
    weight[outside_mask] = 0.0

    # Transition zone: cosine taper
    transition_mask = ~inside_mask & ~outside_mask
    if np.any(transition_mask):
        normalized = (distance[transition_mask] + half_width) / width  # Maps to [0, 1]
        weight[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * normalized))

    return weight


def process_velocity_models_v2(vel3d, vel1d, den1d, mesh,
                                transition_width=10e3,
                                CG_mu_degree=1, 
                                use_smoothing=True,
                                verbose=True):
    """
    Enhanced version: Process 3D and 1D velocity models with smooth transitions.

    Improvements over original version:
    1. Adds smooth transition zone at 3D/1D model boundary to prevent discontinuities
    2. Optimized layer lookup using binary search (O(log n) vs O(n))
    3. Robust handling of edge cases (above-ground, below-deepest-layer points)
    4. Better diagnostics and validation
    5. Configurable transition zone width

    1D models are treated as layered (constant values within each layer).
    3D model uses interpolation between grid points.
    At the boundary, models are smoothly blended to avoid sharp discontinuities.

    Parameters:
    -----------
    vel3d : pd.DataFrame
        3D velocity model with columns: x, y, z (m), vs (km/s)
    vel1d : pd.DataFrame
        1D velocity model with columns: z (m), vs (km/s)
    den1d : pd.DataFrame
        1D density model with columns: z (m), den (kg/m³)
    mesh : dolfin.Mesh
        FEniCS mesh object
    transition_width : float, default=10e3
        Width of smooth transition zone at 3D/1D boundary (meters)
        Larger values = smoother but wider transition
        Set to 0 to disable smoothing (not recommended)
    use_smoothing : bool, default=True
        If True, apply smooth transition at 3D/1D boundary
        If False, use sharp cutoff (may cause numerical issues)
    verbose : bool, default=True
        If True, print detailed progress information

    Returns:
    --------
    vs_function : dolfin.Function
        Shear velocity field on mesh (km/s)
    density_function : dolfin.Function
        Density field on mesh (kg/m³)
    shear_modulus_function : dolfin.Function
        Shear modulus field on mesh (GPa)
    CG_mu : dolfin.FunctionSpace
        Function space used

    Notes:
    ------
    The transition zone blending formula is:
        vs_final = w * vs_3d + (1-w) * vs_1d
    where w is the smooth transition weight:
        - w = 1.0 deep inside 3D box (pure 3D model)
        - w = 0.0 far outside 3D box (pure 1D model)
        - w smoothly varies (cosine taper) in transition zone

    This prevents sharp discontinuities in shear modulus that could cause
    numerical instabilities in FEM solvers.
    """

    if verbose:
        print("=" * 70)
        print("Enhanced Velocity Model Processing (v2 with smoothing)")
        print("=" * 70)

    # Step 1: Clean the data - remove any NaN values
    vel3d_clean = vel3d.dropna(subset=['x', 'y', 'z', 'vs']).copy()
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).copy()
    den1d_clean = den1d.dropna(subset=['z', 'den']).copy()

    if verbose:
        print(f"\nStep 1: Data cleaning")
        print(f"  3D model: {len(vel3d_clean)} points (removed {len(vel3d) - len(vel3d_clean)} NaN)")
        print(f"  1D velocity: {len(vel1d_clean)} layers (removed {len(vel1d) - len(vel1d_clean)} NaN)")
        print(f"  1D density: {len(den1d_clean)} layers (removed {len(den1d) - len(den1d_clean)} NaN)")

    # Check if we have enough data
    if len(vel3d_clean) == 0:
        raise ValueError("No valid data points in 3D model after removing NaN values")
    if len(vel1d_clean) == 0:
        raise ValueError("No valid data points in 1D velocity model after removing NaN values")
    if len(den1d_clean) == 0:
        raise ValueError("No valid data points in 1D density model after removing NaN values")

    # Step 2: Sort 1D models by depth (shallow to deep: 0, -5000, -9000, ...)
    vel1d_clean = vel1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d_clean.sort_values('z', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\nStep 2: Sort 1D models by depth (shallow to deep)")
        print(f"  Velocity: {vel1d_clean['z'].iloc[0]:.0f} to {vel1d_clean['z'].iloc[-1]:.0f} m ({len(vel1d_clean)} layers)")
        print(f"  Density:  {den1d_clean['z'].iloc[0]:.0f} to {den1d_clean['z'].iloc[-1]:.0f} m ({len(den1d_clean)} layers)")

    # Step 3: Extract 3D bounding box
    x_min, x_max = vel3d_clean['x'].min(), vel3d_clean['x'].max()
    y_min, y_max = vel3d_clean['y'].min(), vel3d_clean['y'].max()
    z_min, z_max = vel3d_clean['z'].min(), vel3d_clean['z'].max()

    if verbose:
        print(f"\nStep 3: Extract 3D model bounding box")
        print(f"  X: {x_min/1e3:.1f} to {x_max/1e3:.1f} km (span: {(x_max-x_min)/1e3:.1f} km)")
        print(f"  Y: {y_min/1e3:.1f} to {y_max/1e3:.1f} km (span: {(y_max-y_min)/1e3:.1f} km)")
        print(f"  Z: {z_min/1e3:.1f} to {z_max/1e3:.1f} km (span: {(z_max-z_min)/1e3:.1f} km)")

    # Step 4: Get mesh coordinates
    mesh_coords = mesh.coordinates()
    n_nodes = len(mesh_coords)

    if verbose:
        print(f"\nStep 4: Analyze mesh")
        print(f"  Mesh nodes: {n_nodes}")
        print(f"  Mesh X range: {mesh_coords[:, 0].min()/1e3:.1f} to {mesh_coords[:, 0].max()/1e3:.1f} km")
        print(f"  Mesh Y range: {mesh_coords[:, 1].min()/1e3:.1f} to {mesh_coords[:, 1].max()/1e3:.1f} km")
        print(f"  Mesh Z range: {mesh_coords[:, 2].min()/1e3:.1f} to {mesh_coords[:, 2].max()/1e3:.1f} km")

        # Check coverage
        mesh_z_min = mesh_coords[:, 2].min()
        if mesh_z_min < vel1d_clean['z'].min():
            print(f"  Note: Mesh extends {(vel1d_clean['z'].min() - mesh_z_min)/1e3:.1f} km beyond deepest 1D layer")
            print(f"        Extrapolating with deepest layer values")

    # Step 5: Create FEniCS function space and functions
    if verbose:
        print(f"\nStep 5: Create FEniCS functions")
    
    if CG_mu_degree != 1:
        print(f"Using CG function space of degree {CG_mu_degree} instead of default 1")

    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_degree)
    vs_function = dl.Function(CG_mu)
    density_function = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)

    # Get DOF coordinates
    V_coords = CG_mu.tabulate_dof_coordinates()
    n_dofs = len(V_coords)

    if verbose:
        print(f"  Function space: CG{CG_mu_degree} with {n_dofs} DOFs")

    # Step 6: Apply 1D layered models to ALL DOFs (background)
    if verbose:
        print(f"\nStep 6: Apply 1D layered background models")

    vs_1d = get_layered_1d_value_interp(
        V_coords[:, 2],
        vel1d_clean['z'].values,
        vel1d_clean['vs'].values
    )

    density_at_dofs = get_layered_1d_value_interp(
        V_coords[:, 2],
        den1d_clean['z'].values,
        den1d_clean['den'].values
    )

    if verbose:
        print(f"  1D velocity: {vs_1d.min():.3f} to {vs_1d.max():.3f} km/s")
        print(f"  1D density:  {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")

    # Step 7: Determine which DOFs need 3D model
    inside_3d_dofs = ((V_coords[:, 0] >= x_min) & (V_coords[:, 0] <= x_max) &
                      (V_coords[:, 1] >= y_min) & (V_coords[:, 1] <= y_max) &
                      (V_coords[:, 2] >= z_min) & (V_coords[:, 2] <= z_max))

    n_inside_dofs = np.sum(inside_3d_dofs)
    n_outside_dofs = np.sum(~inside_3d_dofs)

    if verbose:
        print(f"\nStep 7: Classify DOFs relative to 3D box")
        print(f"  Inside 3D box:  {n_inside_dofs} DOFs ({100*n_inside_dofs/n_dofs:.1f}%)")
        print(f"  Outside 3D box: {n_outside_dofs} DOFs ({100*n_outside_dofs/n_dofs:.1f}%)")

    # Initialize with 1D model everywhere
    vs_at_dofs = vs_1d.copy()

    # Step 8: Apply 3D model with optional smoothing
    if n_inside_dofs > 0:
        if verbose:
            print(f"\nStep 8: Apply 3D model")
            if use_smoothing:
                print(f"  Smoothing: ENABLED (transition width = {transition_width/1e3:.1f} km)")
            else:
                print(f"  Smoothing: DISABLED (sharp boundary)")

        # Interpolate 3D velocity for points inside box
        vs_3d = griddata(
            points=vel3d_clean[['x', 'y', 'z']].values,
            values=vel3d_clean['vs'].values,
            xi=V_coords[inside_3d_dofs],
            method='linear',
            fill_value=np.nan
        )

        # Handle NaN values (points where linear interpolation failed)
        # Use RBF (Radial Basis Function) interpolation for smooth extrapolation
        # This preserves 3D heterogeneity while ensuring smoothness
        nan_mask = np.isnan(vs_3d)
        n_nan = np.sum(nan_mask)

        if n_nan > 0:
            if verbose:
                print(f"  Filling {n_nan} NaN values with RBF extrapolation ({100*n_nan/n_inside_dofs:.1f}% of inside points)")
                print(f"  (Using smooth extrapolation from 3D data to preserve heterogeneity)")

            # Use RBF interpolation for smooth extrapolation from 3D data
            from scipy.interpolate import Rbf

            # Create RBF interpolator from 3D data
            # Using 'thin_plate' for better preservation of extremes
            rbf_interp = Rbf(
                vel3d_clean['x'].values,
                vel3d_clean['y'].values,
                vel3d_clean['z'].values,
                vel3d_clean['vs'].values,
                function='thin_plate',  # Better preserves extremes than multiquadric
                smooth=0.0  # No smoothing - preserve heterogeneity
            )

            # Interpolate at NaN locations
            nan_coords = V_coords[inside_3d_dofs][nan_mask]
            vs_3d[nan_mask] = rbf_interp(nan_coords[:, 0], nan_coords[:, 1], nan_coords[:, 2])

            # Sanity check: clip to reasonable range based on 3D data extent
            # Use wider bounds to preserve heterogeneity
            vs_min_valid = vel3d_clean['vs'].min() * 0.7  # Allow 30% below minimum
            vs_max_valid = vel3d_clean['vs'].max() * 1.3  # Allow 30% above maximum
            vs_3d[nan_mask] = np.clip(vs_3d[nan_mask], vs_min_valid, vs_max_valid)

        if verbose:
            print(f"  3D velocity range: {vs_3d.min():.3f} to {vs_3d.max():.3f} km/s")

        # Apply smoothing if requested
        if use_smoothing and transition_width > 0:
            if verbose:
                print(f"\nStep 9: Apply smooth transition")

            # Compute signed distance to box for ALL DOFs
            distances = signed_distance_to_box(
                V_coords,
                x_min, x_max, y_min, y_max, z_min, z_max
            )

            # Compute smooth weights
            weights = smooth_transition_weight(distances, transition_width)

            # Count DOFs in each zone
            n_pure_3d = np.sum(weights >= 0.99)
            n_pure_1d = np.sum(weights <= 0.01)
            n_transition = n_dofs - n_pure_3d - n_pure_1d

            if verbose:
                print(f"  Pure 3D zone (w > 0.99):   {n_pure_3d} DOFs ({100*n_pure_3d/n_dofs:.1f}%)")
                print(f"  Transition zone:            {n_transition} DOFs ({100*n_transition/n_dofs:.1f}%)")
                print(f"  Pure 1D zone (w < 0.01):    {n_pure_1d} DOFs ({100*n_pure_1d/n_dofs:.1f}%)")

            # Apply blending: vs_final = w * vs_3d + (1-w) * vs_1d
            # For DOFs inside the box, blend 3D and 1D values
            vs_at_dofs[inside_3d_dofs] = (weights[inside_3d_dofs] * vs_3d +
                                          (1 - weights[inside_3d_dofs]) * vs_1d[inside_3d_dofs])

            # For DOFs outside the box but in transition zone, might need to blend too
            # But vs_3d is not defined there, so just keep 1D (weight will be ~0 anyway)

        else:
            # No smoothing: hard cutoff
            vs_at_dofs[inside_3d_dofs] = vs_3d

            if verbose and not use_smoothing:
                print(f"\n  WARNING: Sharp boundary between 3D and 1D models!")
                print(f"           This may cause numerical instabilities.")
                print(f"           Consider enabling smoothing with use_smoothing=True")

    # Step 10: Assign to FEniCS functions
    if verbose:
        print(f"\nStep 10: Assign values to FEniCS functions")

    vs_function.vector()[:] = vs_at_dofs
    density_function.vector()[:] = density_at_dofs

    # Step 11: Compute shear modulus
    if verbose:
        print(f"\nStep 11: Compute shear modulus (μ = ρ × vs²)")

    vs_ms = vs_at_dofs * 1000.0  # km/s to m/s
    shear_modulus_pa = density_at_dofs * vs_ms**2  # Pa
    shear_modulus_gpa = shear_modulus_pa / 1e9  # GPa

    shear_modulus_function.vector()[:] = shear_modulus_gpa

    if verbose:
        print(f"  Velocity range:      {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  Density range:       {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")
        print(f"  Shear modulus range: {shear_modulus_gpa.min():.2f} to {shear_modulus_gpa.max():.2f} GPa")

    # Step 12: Validation checks
    if verbose:
        print(f"\nStep 12: Validation")

        # Check for any remaining NaN or inf
        if np.any(np.isnan(vs_at_dofs)) or np.any(np.isinf(vs_at_dofs)):
            print(f"  ERROR: Found NaN or Inf in velocity field!")
        else:
            print(f"  ✓ No NaN or Inf values in velocity")

        if np.any(np.isnan(density_at_dofs)) or np.any(np.isinf(density_at_dofs)):
            print(f"  ERROR: Found NaN or Inf in density field!")
        else:
            print(f"  ✓ No NaN or Inf values in density")

        if np.any(np.isnan(shear_modulus_gpa)) or np.any(np.isinf(shear_modulus_gpa)):
            print(f"  ERROR: Found NaN or Inf in shear modulus!")
        else:
            print(f"  ✓ No NaN or Inf values in shear modulus")

    if verbose:
        print("=" * 70)
        print("Processing complete!")
        print("=" * 70)

    return vs_function, density_function, shear_modulus_function, CG_mu


def process_velocity_models_v3(vel3d, vel1d, den1d, mesh,
                               transition_width=100e3,
                               CG_mu_degree=1,
                               verbose=True):
    """
    Process 3D velocity and 1D density models onto FEniCS mesh (v3 - Depth-averaged background)

    Key differences from v2:
    - Inside 3D box: Full heterogeneity from 3D model (griddata + RBF for sparse points)
    - Outside 3D box: Depth-averaged velocity from 3D model at each depth
    - Smooth transition between heterogeneous 3D and uniform depth-averaged values
    - NO RBF extrapolation (prevents artifacts)

    This approach creates laterally uniform values outside the 3D domain while
    preserving depth variation from the 3D model, avoiding extrapolation artifacts.

    Parameters:
    -----------
    vel3d : pd.DataFrame
        3D velocity model with columns: ['x', 'y', 'z', 'vs']
        - x, y, z in meters (local coordinates)
        - vs in km/s
    vel1d : pd.DataFrame
        1D velocity model with columns: ['z', 'vs']
        - z in meters (negative downward: 0, -5000, -9000, ...)
        - vs in km/s (used only as fallback if depth-averaging fails)
    den1d : pd.DataFrame
        1D density model with columns: ['z', 'den']
        - z in meters (negative downward: 0, -5000, -9000, ...)
        - den in kg/m³
    mesh : dolfin.Mesh
        FEniCS mesh
    transition_width : float, default=100e3
        Width of smooth transition zone between 3D and depth-averaged regions (meters)
    verbose : bool, default=True
        Print detailed progress information

    Returns:
    --------
    vs_function : dolfin.Function
        Shear velocity field (km/s)
    density_function : dolfin.Function
        Density field (kg/m³)
    shear_modulus_function : dolfin.Function
        Shear modulus field (GPa)
    CG_mu : dolfin.FunctionSpace
        Function space used
    """

    if verbose:
        print("=" * 70)
        print("Velocity Model Processing v3 (Depth-Averaged Background)")
        print("=" * 70)

    # Step 1: Clean the data
    vel3d_clean = vel3d.dropna(subset=['x', 'y', 'z', 'vs']).copy()
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).copy()
    den1d_clean = den1d.dropna(subset=['z', 'den']).copy()

    if verbose:
        print(f"\nStep 1: Data cleaning")
        print(f"  3D velocity model: {len(vel3d_clean)} points (removed {len(vel3d) - len(vel3d_clean)} NaN)")
        print(f"  1D velocity model: {len(vel1d_clean)} layers (removed {len(vel1d) - len(vel1d_clean)} NaN)")
        print(f"  1D density model: {len(den1d_clean)} layers (removed {len(den1d) - len(den1d_clean)} NaN)")

    # Check if we have enough data
    if len(vel3d_clean) == 0:
        raise ValueError("No valid data points in 3D velocity model")
    if len(vel1d_clean) == 0:
        raise ValueError("No valid data points in 1D velocity model")
    if len(den1d_clean) == 0:
        raise ValueError("No valid data points in 1D density model")

    # Step 2: Sort 1D models by depth
    vel1d_clean = vel1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d_clean.sort_values('z', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\nStep 2: Sort 1D models by depth (shallow to deep)")
        print(f"  Velocity: {vel1d_clean['z'].iloc[0]:.0f} to {vel1d_clean['z'].iloc[-1]:.0f} m ({len(vel1d_clean)} layers)")
        print(f"  Density:  {den1d_clean['z'].iloc[0]:.0f} to {den1d_clean['z'].iloc[-1]:.0f} m ({len(den1d_clean)} layers)")

    # Step 3: Analyze 3D velocity model extent
    x_min, x_max = vel3d_clean['x'].min(), vel3d_clean['x'].max()
    y_min, y_max = vel3d_clean['y'].min(), vel3d_clean['y'].max()
    z_min, z_max = vel3d_clean['z'].min(), vel3d_clean['z'].max()

    if verbose:
        print(f"\nStep 3: 3D velocity model extent")
        print(f"  X: {x_min/1e3:.1f} to {x_max/1e3:.1f} km (span: {(x_max-x_min)/1e3:.1f} km)")
        print(f"  Y: {y_min/1e3:.1f} to {y_max/1e3:.1f} km (span: {(y_max-y_min)/1e3:.1f} km)")
        print(f"  Z: {z_min/1e3:.1f} to {z_max/1e3:.1f} km (span: {(z_max-z_min)/1e3:.1f} km)")
        print(f"  Velocity range: {vel3d_clean['vs'].min():.3f} to {vel3d_clean['vs'].max():.3f} km/s")

    # Step 4: Analyze mesh
    mesh_coords = mesh.coordinates()
    n_nodes = len(mesh_coords)

    if verbose:
        print(f"\nStep 4: Mesh analysis")
        print(f"  Mesh nodes: {n_nodes}")
        print(f"  Mesh X range: {mesh_coords[:, 0].min()/1e3:.1f} to {mesh_coords[:, 0].max()/1e3:.1f} km")
        print(f"  Mesh Y range: {mesh_coords[:, 1].min()/1e3:.1f} to {mesh_coords[:, 1].max()/1e3:.1f} km")
        print(f"  Mesh Z range: {mesh_coords[:, 2].min()/1e3:.1f} to {mesh_coords[:, 2].max()/1e3:.1f} km")

    # Step 5: Create FEniCS function space
    if verbose:
        print(f"\nStep 5: Create FEniCS functions")
    
    if CG_mu_degree != 1:
        print(f"Using CG function space of degree {CG_mu_degree} instead of default 1")

    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_degree)
    vs_function = dl.Function(CG_mu)
    density_function = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)

    # Get DOF coordinates
    V_coords = CG_mu.tabulate_dof_coordinates()
    n_dofs = len(V_coords)

    if verbose:
        print(f"  Function space: CG{CG_mu_degree} with {n_dofs} DOFs")

    # Step 6: Apply 1D density model
    if verbose:
        print(f"\nStep 6: Apply 1D density model")

    density_at_dofs = get_layered_1d_value_interp(
        V_coords[:, 2],
        den1d_clean['z'].values,
        den1d_clean['den'].values
    )

    if verbose:
        print(f"  1D density:  {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")

    # Step 7: Classify DOFs inside/outside 3D box
    inside_3d_dofs = ((V_coords[:, 0] >= x_min) & (V_coords[:, 0] <= x_max) &
                      (V_coords[:, 1] >= y_min) & (V_coords[:, 1] <= y_max) &
                      (V_coords[:, 2] >= z_min) & (V_coords[:, 2] <= z_max))

    n_inside_dofs = np.sum(inside_3d_dofs)
    n_outside_dofs = np.sum(~inside_3d_dofs)

    if verbose:
        print(f"\nStep 7: Classify DOFs relative to 3D box")
        print(f"  Inside 3D box:  {n_inside_dofs} DOFs ({100*n_inside_dofs/n_dofs:.1f}%)")
        print(f"  Outside 3D box: {n_outside_dofs} DOFs ({100*n_outside_dofs/n_dofs:.1f}%)")

    # Step 8: Project 3D model onto mesh DOFs inside the box
    vs_at_dofs = np.zeros(n_dofs)

    if n_inside_dofs > 0:
        if verbose:
            print(f"\nStep 8: Project 3D model onto inside DOFs")

        # Use griddata linear interpolation for points inside box
        vs_3d = griddata(
            points=vel3d_clean[['x', 'y', 'z']].values,
            values=vel3d_clean['vs'].values,
            xi=V_coords[inside_3d_dofs],
            method='linear',
            fill_value=np.nan
        )

        # Handle NaN values with RBF (sparse points inside box)
        nan_mask = np.isnan(vs_3d)
        n_nan = np.sum(nan_mask)

        if n_nan > 0:
            if verbose:
                print(f"  Filling {n_nan} sparse points with RBF ({100*n_nan/n_inside_dofs:.1f}% of inside points)")

            from scipy.interpolate import Rbf

            rbf_interp = Rbf(
                vel3d_clean['x'].values,
                vel3d_clean['y'].values,
                vel3d_clean['z'].values,
                vel3d_clean['vs'].values,
                function='thin_plate',
                smooth=0.0
            )

            nan_coords = V_coords[inside_3d_dofs][nan_mask]
            vs_3d[nan_mask] = rbf_interp(nan_coords[:, 0], nan_coords[:, 1], nan_coords[:, 2])

            # Clip to reasonable range
            vs_min_valid = vel3d_clean['vs'].min() * 0.7
            vs_max_valid = vel3d_clean['vs'].max() * 1.3
            vs_3d[nan_mask] = np.clip(vs_3d[nan_mask], vs_min_valid, vs_max_valid)

        # Assign to inside DOFs
        vs_at_dofs[inside_3d_dofs] = vs_3d

        if verbose:
            print(f"  Inside 3D velocity range: {vs_3d.min():.3f} to {vs_3d.max():.3f} km/s")

    # Step 9: Compute depth-averaged values for outside DOFs
    if n_outside_dofs > 0:
        if verbose:
            print(f"\nStep 9: Compute depth-averaged values for outside DOFs")
            print(f"  Using progressive windowing (adaptive to mesh size)")

        # Get indices and depths of inside DOFs
        inside_indices = np.where(inside_3d_dofs)[0]
        inside_depths = V_coords[inside_3d_dofs, 2]
        inside_vs = vs_at_dofs[inside_3d_dofs]

        # Get indices and depths of outside DOFs
        outside_indices = np.where(~inside_3d_dofs)[0]
        outside_depths = V_coords[~inside_3d_dofs, 2]

        # Progressive depth windows (in meters)
        depth_windows = [500, 1000, 2000, 5000, 10000, 20000]

        n_no_match = 0
        for i, out_idx in enumerate(outside_indices):
            out_depth = outside_depths[i]

            # Try progressively larger windows until we find enough points
            found = False
            for window in depth_windows:
                # Find inside DOFs within depth window
                depth_mask = np.abs(inside_depths - out_depth) <= window
                n_matches = np.sum(depth_mask)

                if n_matches >= 5:  # Require at least 5 points for averaging
                    # Average velocity from matched inside points
                    vs_at_dofs[out_idx] = np.mean(inside_vs[depth_mask])
                    found = True
                    break

            if not found:
                # Fallback: use 1D model if no matches found
                n_no_match += 1
                vs_at_dofs[out_idx] = get_layered_1d_value_interp(
                    np.array([out_depth]),
                    vel1d_clean['z'].values,
                    vel1d_clean['vs'].values
                )[0]

        if verbose:
            print(f"  Outside DOFs with depth-averaged values: {n_outside_dofs - n_no_match}")
            if n_no_match > 0:
                print(f"  Outside DOFs using 1D fallback: {n_no_match} ({100*n_no_match/n_outside_dofs:.1f}%)")
            print(f"  Outside velocity range: {vs_at_dofs[~inside_3d_dofs].min():.3f} to {vs_at_dofs[~inside_3d_dofs].max():.3f} km/s")

    # Step 10: Apply smooth transition between inside and outside
    if verbose:
        print(f"\nStep 10: Apply smooth transition")
        print(f"  Transition width: {transition_width/1e3:.1f} km")

    # Compute signed distance to box
    distances = signed_distance_to_box(
        V_coords,
        x_min, x_max, y_min, y_max, z_min, z_max
    )

    # Compute smooth weights
    weights = smooth_transition_weight(distances, transition_width)

    # Store original values before blending
    vs_inside_original = vs_at_dofs.copy()

    # For points in transition zone, blend inside and outside values
    # Weight = 1.0 (pure inside) deep in box, 0.0 (pure outside) far from box
    # We need to get "outside values" for inside DOFs in transition zone

    # For inside DOFs in transition zone, compute what their "outside" value would be
    inside_transition_mask = inside_3d_dofs & (weights > 0.01) & (weights < 0.99)

    if np.any(inside_transition_mask):
        # For these transition points, compute depth-averaged value from nearby inside points
        trans_indices = np.where(inside_transition_mask)[0]

        for trans_idx in trans_indices:
            trans_depth = V_coords[trans_idx, 2]

            # Find other inside DOFs at similar depth (excluding this point)
            other_inside_mask = inside_3d_dofs.copy()
            other_inside_mask[trans_idx] = False
            other_inside_depths = V_coords[other_inside_mask, 2]

            # Use progressive windowing
            for window in depth_windows:
                depth_mask = np.abs(other_inside_depths - trans_depth) <= window
                if np.sum(depth_mask) >= 5:
                    outside_value = np.mean(vs_inside_original[other_inside_mask][depth_mask])
                    # Blend
                    w = weights[trans_idx]
                    vs_at_dofs[trans_idx] = w * vs_inside_original[trans_idx] + (1 - w) * outside_value
                    break

    # Count DOFs in each zone
    n_pure_inside = np.sum(weights >= 0.99)
    n_pure_outside = np.sum(weights <= 0.01)
    n_transition = n_dofs - n_pure_inside - n_pure_outside

    if verbose:
        print(f"  Pure inside zone (w > 0.99):    {n_pure_inside} DOFs ({100*n_pure_inside/n_dofs:.1f}%)")
        print(f"  Transition zone:                 {n_transition} DOFs ({100*n_transition/n_dofs:.1f}%)")
        print(f"  Pure outside zone (w < 0.01):    {n_pure_outside} DOFs ({100*n_pure_outside/n_dofs:.1f}%)")
        print(f"  Final velocity range: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")

    # Step 11: Assign to FEniCS functions
    if verbose:
        print(f"\nStep 11: Assign values to FEniCS functions")

    vs_function.vector()[:] = vs_at_dofs
    density_function.vector()[:] = density_at_dofs

    # Step 12: Compute shear modulus
    if verbose:
        print(f"\nStep 12: Compute shear modulus (μ = ρ × vs²)")

    vs_ms = vs_at_dofs * 1000.0  # km/s to m/s
    shear_modulus_pa = density_at_dofs * vs_ms**2  # Pa
    shear_modulus_gpa = shear_modulus_pa / 1e9  # GPa

    shear_modulus_function.vector()[:] = shear_modulus_gpa

    if verbose:
        print(f"  Velocity range:      {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  Density range:       {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")
        print(f"  Shear modulus range: {shear_modulus_gpa.min():.2f} to {shear_modulus_gpa.max():.2f} GPa")

    # Step 13: Validation
    if verbose:
        print(f"\nStep 13: Validation")

        # Check for NaN or inf
        if np.any(np.isnan(vs_at_dofs)) or np.any(np.isinf(vs_at_dofs)):
            print(f"  ✗ ERROR: Found NaN or Inf in velocity field!")
        else:
            print(f"  ✓ No NaN or Inf values in velocity")

        if np.any(np.isnan(density_at_dofs)) or np.any(np.isinf(density_at_dofs)):
            print(f"  ✗ ERROR: Found NaN or Inf in density field!")
        else:
            print(f"  ✓ No NaN or Inf values in density")

        if np.any(np.isnan(shear_modulus_gpa)) or np.any(np.isinf(shear_modulus_gpa)):
            print(f"  ✗ ERROR: Found NaN or Inf in shear modulus!")
        else:
            print(f"  ✓ No NaN or Inf values in shear modulus")

    if verbose:
        print("=" * 70)
        print("Processing complete!")
        print("=" * 70)

    return vs_function, density_function, shear_modulus_function, CG_mu


def process_velocity_models_v4(vel3d, vel1d, den1d, mesh,
                               use_rbf_sparse=False,
                               transition_width=0.0,
                               CG_mu_degree=1,
                               verbose=True):
    """
    Process 3D velocity and 1D density models onto FEniCS mesh (v4 - Enhanced v1)

    This version combines the proven strategy from v1 with modern improvements:
    - Sharp boundary at 3D box edge (works best, no artifacts)
    - 1D model as laterally uniform background (smooth at corners)
    - Linear interpolation for 1D models (smooth depth variation)
    - Choice between nearest-neighbor (safe) or RBF (better) for sparse points
    - Optional narrow transition zone (default off)

    Key differences from v1:
    - Uses linear interpolation instead of step function for 1D models
    - Optional RBF for sparse points (instead of always nearest-neighbor)
    - Optional transition zone (default: sharp boundary like v1)

    Parameters:
    -----------
    vel3d : pd.DataFrame
        3D velocity model with columns: ['x', 'y', 'z', 'vs']
        - x, y, z in meters (local coordinates)
        - vs in km/s
    vel1d : pd.DataFrame
        1D velocity model with columns: ['z', 'vs']
        - z in meters (negative downward: 0, -5000, -9000, ...)
        - vs in km/s
    den1d : pd.DataFrame
        1D density model with columns: ['z', 'den']
        - z in meters (negative downward: 0, -5000, -9000, ...)
        - den in kg/m³
    mesh : dolfin.Mesh
        FEniCS mesh
    use_rbf_sparse : bool, default=False
        If True, use RBF for sparse points inside 3D box
        If False, use nearest-neighbor (safer, like v1)
    transition_width : float, default=0.0
        Width of transition zone at 3D box boundary (meters)
        0.0 = sharp boundary (like v1, works best)
        Recommended: 0-10000 (0-10 km) if needed
    verbose : bool, default=True
        Print detailed progress information

    Returns:
    --------
    vs_function : dolfin.Function
        Shear velocity field (km/s)
    density_function : dolfin.Function
        Density field (kg/m³)
    shear_modulus_function : dolfin.Function
        Shear modulus field (GPa)
    CG_mu : dolfin.FunctionSpace
        Function space used
    """

    if verbose:
        print("=" * 70)
        print("Velocity Model Processing v4 (Enhanced v1)")
        print("=" * 70)

    # Step 1: Clean the data
    vel3d_clean = vel3d.dropna(subset=['x', 'y', 'z', 'vs']).copy()
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).copy()
    den1d_clean = den1d.dropna(subset=['z', 'den']).copy()

    if verbose:
        print(f"\nStep 1: Data cleaning")
        print(f"  3D velocity model: {len(vel3d_clean)} points (removed {len(vel3d) - len(vel3d_clean)} NaN)")
        print(f"  1D velocity model: {len(vel1d_clean)} layers (removed {len(vel1d) - len(vel1d_clean)} NaN)")
        print(f"  1D density model: {len(den1d_clean)} layers (removed {len(den1d) - len(den1d_clean)} NaN)")

    # Check if we have enough data
    if len(vel3d_clean) == 0:
        raise ValueError("No valid data points in 3D velocity model")
    if len(vel1d_clean) == 0:
        raise ValueError("No valid data points in 1D velocity model")
    if len(den1d_clean) == 0:
        raise ValueError("No valid data points in 1D density model")

    # Step 2: Sort 1D models by depth
    vel1d_clean = vel1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d_clean.sort_values('z', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\nStep 2: Sort 1D models by depth (shallow to deep)")
        print(f"  Velocity: {vel1d_clean['z'].iloc[0]:.0f} to {vel1d_clean['z'].iloc[-1]:.0f} m ({len(vel1d_clean)} layers)")
        print(f"  Density:  {den1d_clean['z'].iloc[0]:.0f} to {den1d_clean['z'].iloc[-1]:.0f} m ({len(den1d_clean)} layers)")

    # Step 3: Extract 3D model bounding box
    x_min, x_max = vel3d_clean['x'].min(), vel3d_clean['x'].max()
    y_min, y_max = vel3d_clean['y'].min(), vel3d_clean['y'].max()
    z_min, z_max = vel3d_clean['z'].min(), vel3d_clean['z'].max()

    if verbose:
        print(f"\nStep 3: 3D velocity model extent")
        print(f"  X: {x_min/1e3:.1f} to {x_max/1e3:.1f} km (span: {(x_max-x_min)/1e3:.1f} km)")
        print(f"  Y: {y_min/1e3:.1f} to {y_max/1e3:.1f} km (span: {(y_max-y_min)/1e3:.1f} km)")
        print(f"  Z: {z_min/1e3:.1f} to {z_max/1e3:.1f} km (span: {(z_max-z_min)/1e3:.1f} km)")
        print(f"  Velocity range: {vel3d_clean['vs'].min():.3f} to {vel3d_clean['vs'].max():.3f} km/s")

    # Step 4: Analyze mesh
    mesh_coords = mesh.coordinates()
    n_nodes = len(mesh_coords)

    if verbose:
        print(f"\nStep 4: Mesh analysis")
        print(f"  Mesh nodes: {n_nodes}")
        print(f"  Mesh X range: {mesh_coords[:, 0].min()/1e3:.1f} to {mesh_coords[:, 0].max()/1e3:.1f} km")
        print(f"  Mesh Y range: {mesh_coords[:, 1].min()/1e3:.1f} to {mesh_coords[:, 1].max()/1e3:.1f} km")
        print(f"  Mesh Z range: {mesh_coords[:, 2].min()/1e3:.1f} to {mesh_coords[:, 2].max()/1e3:.1f} km")

    # Step 5: Create FEniCS function space
    if verbose:
        print(f"\nStep 5: Create FEniCS functions")
    
    if CG_mu_degree != 1:
        print(f"Using CG function space of degree {CG_mu_degree} instead of default 1")

    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_degree)
    vs_function = dl.Function(CG_mu)
    density_function = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)

    # Get DOF coordinates
    V_coords = CG_mu.tabulate_dof_coordinates()
    n_dofs = len(V_coords)

    if verbose:
        print(f"  Function space: CG{CG_mu_degree} with {n_dofs} DOFs")

    # Step 6: Apply 1D models to ALL DOFs (background) using linear interpolation
    if verbose:
        print(f"\nStep 6: Apply 1D background models (linear interpolation)")

    vs_at_dofs = get_layered_1d_value_interp(
        V_coords[:, 2],
        vel1d_clean['z'].values,
        vel1d_clean['vs'].values
    )

    density_at_dofs = get_layered_1d_value_interp(
        V_coords[:, 2],
        den1d_clean['z'].values,
        den1d_clean['den'].values
    )

    if verbose:
        print(f"  1D velocity: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  1D density:  {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")

    # Step 7: Classify DOFs inside/outside 3D box
    inside_3d_dofs = ((V_coords[:, 0] >= x_min) & (V_coords[:, 0] <= x_max) &
                      (V_coords[:, 1] >= y_min) & (V_coords[:, 1] <= y_max) &
                      (V_coords[:, 2] >= z_min) & (V_coords[:, 2] <= z_max))

    n_inside_dofs = np.sum(inside_3d_dofs)
    n_outside_dofs = np.sum(~inside_3d_dofs)

    if verbose:
        print(f"\nStep 7: Classify DOFs relative to 3D box")
        print(f"  Inside 3D box:  {n_inside_dofs} DOFs ({100*n_inside_dofs/n_dofs:.1f}%)")
        print(f"  Outside 3D box: {n_outside_dofs} DOFs ({100*n_outside_dofs/n_dofs:.1f}%)")

    # Step 8: Apply 3D model inside box
    if n_inside_dofs > 0:
        if verbose:
            print(f"\nStep 8: Apply 3D model inside box")

        # Use griddata linear interpolation
        vs_3d = griddata(
            points=vel3d_clean[['x', 'y', 'z']].values,
            values=vel3d_clean['vs'].values,
            xi=V_coords[inside_3d_dofs],
            method='linear',
            fill_value=np.nan
        )

        # Handle NaN values (sparse points)
        nan_mask = np.isnan(vs_3d)
        n_nan = np.sum(nan_mask)

        if n_nan > 0:
            if use_rbf_sparse:
                if verbose:
                    print(f"  Filling {n_nan} sparse points with RBF ({100*n_nan/n_inside_dofs:.1f}% of inside points)")

                from scipy.interpolate import Rbf

                rbf_interp = Rbf(
                    vel3d_clean['x'].values,
                    vel3d_clean['y'].values,
                    vel3d_clean['z'].values,
                    vel3d_clean['vs'].values,
                    function='thin_plate',
                    smooth=0.0
                )

                nan_coords = V_coords[inside_3d_dofs][nan_mask]
                vs_3d[nan_mask] = rbf_interp(nan_coords[:, 0], nan_coords[:, 1], nan_coords[:, 2])

                # Clip to reasonable range
                vs_min_valid = vel3d_clean['vs'].min() * 0.7
                vs_max_valid = vel3d_clean['vs'].max() * 1.3
                vs_3d[nan_mask] = np.clip(vs_3d[nan_mask], vs_min_valid, vs_max_valid)
            else:
                # Use nearest neighbor (safe, like v1)
                if verbose:
                    print(f"  Filling {n_nan} sparse points with nearest neighbor ({100*n_nan/n_inside_dofs:.1f}% of inside points)")

                vs_3d[nan_mask] = griddata(
                    points=vel3d_clean[['x', 'y', 'z']].values,
                    values=vel3d_clean['vs'].values,
                    xi=V_coords[inside_3d_dofs][nan_mask],
                    method='nearest'
                )

        # Store 3D values before blending
        vs_3d_values = vs_3d.copy()

        if verbose:
            print(f"  3D velocity range: {vs_3d.min():.3f} to {vs_3d.max():.3f} km/s")

        # Step 9: Apply transition zone (if requested)
        if transition_width > 0:
            if verbose:
                print(f"\nStep 9: Apply smooth transition (width = {transition_width/1e3:.1f} km)")

            # Compute signed distance to box
            distances = signed_distance_to_box(
                V_coords,
                x_min, x_max, y_min, y_max, z_min, z_max
            )

            # Compute smooth weights
            weights = smooth_transition_weight(distances, transition_width)

            # Count DOFs in each zone
            n_pure_3d = np.sum(weights >= 0.99)
            n_pure_1d = np.sum(weights <= 0.01)
            n_transition = n_dofs - n_pure_3d - n_pure_1d

            if verbose:
                print(f"  Pure 3D zone (w > 0.99):   {n_pure_3d} DOFs ({100*n_pure_3d/n_dofs:.1f}%)")
                print(f"  Transition zone:            {n_transition} DOFs ({100*n_transition/n_dofs:.1f}%)")
                print(f"  Pure 1D zone (w < 0.01):    {n_pure_1d} DOFs ({100*n_pure_1d/n_dofs:.1f}%)")

            # Apply blending for inside DOFs
            vs_at_dofs[inside_3d_dofs] = (weights[inside_3d_dofs] * vs_3d_values +
                                          (1 - weights[inside_3d_dofs]) * vs_at_dofs[inside_3d_dofs])
        else:
            # Sharp boundary (like v1)
            if verbose:
                print(f"\nStep 9: Apply sharp boundary (no transition)")

            vs_at_dofs[inside_3d_dofs] = vs_3d_values

        if verbose:
            print(f"  Final velocity range: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")

    # Step 10: Assign to FEniCS functions
    if verbose:
        print(f"\nStep 10: Assign values to FEniCS functions")

    vs_function.vector()[:] = vs_at_dofs
    density_function.vector()[:] = density_at_dofs

    # Step 11: Compute shear modulus
    if verbose:
        print(f"\nStep 11: Compute shear modulus (μ = ρ × vs²)")

    vs_ms = vs_at_dofs * 1000.0  # km/s to m/s
    shear_modulus_pa = density_at_dofs * vs_ms**2  # Pa
    shear_modulus_gpa = shear_modulus_pa / 1e9  # GPa

    shear_modulus_function.vector()[:] = shear_modulus_gpa

    if verbose:
        print(f"  Velocity range:      {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  Density range:       {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")
        print(f"  Shear modulus range: {shear_modulus_gpa.min():.2f} to {shear_modulus_gpa.max():.2f} GPa")

    # Step 12: Validation
    if verbose:
        print(f"\nStep 12: Validation")

        # Check for NaN or inf
        if np.any(np.isnan(vs_at_dofs)) or np.any(np.isinf(vs_at_dofs)):
            print(f"  ✗ ERROR: Found NaN or Inf in velocity field!")
        else:
            print(f"  ✓ No NaN or Inf values in velocity")

        if np.any(np.isnan(density_at_dofs)) or np.any(np.isinf(density_at_dofs)):
            print(f"  ✗ ERROR: Found NaN or Inf in density field!")
        else:
            print(f"  ✓ No NaN or Inf values in density")

        if np.any(np.isnan(shear_modulus_gpa)) or np.any(np.isinf(shear_modulus_gpa)):
            print(f"  ✗ ERROR: Found NaN or Inf in shear modulus!")
        else:
            print(f"  ✓ No NaN or Inf values in shear modulus")

    if verbose:
        print("=" * 70)
        print("Processing complete!")
        print("=" * 70)

    return vs_function, density_function, shear_modulus_function, CG_mu


def save_functions_to_file(vs_function, density_function, shear_modulus_function,
                          output_dir="./output/", meshname="mesh", verbose=True):
    """
    Save the computed functions to XDMF files for visualization and HDF5 for storage.
    
    Parameters:
    -----------
    vs_function, density_function, shear_modulus_function : dolfin.Function
        The computed field functions
    output_dir : str
        Directory to save the files
    meshname : str
        Mesh name for file naming
    verbose : bool, default=True
        If True, print progress information
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Saving functions to {output_dir}")
    
    # Save as XDMF files (for visualization in ParaView/VisIt)
    # Velocity
    vs_filename = output_dir + f'vs_{meshname}.xdmf'
    vs_file = dl.XDMFFile(vs_filename)
    vs_function.rename('shear velocity', 'shear velocity')
    vs_file.write(vs_function)
    vs_file.close()
    
    # Density  
    density_filename = output_dir + f'density_{meshname}.xdmf'
    density_file = dl.XDMFFile(density_filename)
    density_function.rename('density', 'density')
    density_file.write(density_function)
    density_file.close()
    
    # Shear modulus
    mu_filename = output_dir + f'shear_modulus_{meshname}.xdmf'
    mu_file = dl.XDMFFile(mu_filename)
    shear_modulus_function.rename('shear modulus', 'shear modulus')
    mu_file.write(shear_modulus_function)
    mu_file.close()
    
    # Also save as HDF5 files (for loading back into FEniCS)
    vs_h5_file = dl.HDF5File(vs_function.function_space().mesh().mpi_comm(), 
                            output_dir + f"vs_{meshname}.h5", "w")
    vs_h5_file.write(vs_function, "vs")
    vs_h5_file.close()
    
    density_h5_file = dl.HDF5File(density_function.function_space().mesh().mpi_comm(), 
                                 output_dir + f"density_{meshname}.h5", "w")
    density_h5_file.write(density_function, "density") 
    density_h5_file.close()
    
    mu_h5_file = dl.HDF5File(shear_modulus_function.function_space().mesh().mpi_comm(), 
                            output_dir + f"shear_modulus_{meshname}.h5", "w")
    mu_h5_file.write(shear_modulus_function, "shear_modulus")
    mu_h5_file.close()
    
    if verbose:
        print("Functions saved successfully!")
        print(f"XDMF files (for visualization):")
        print(f"  {vs_filename}")
        print(f"  {density_filename}") 
        print(f"  {mu_filename}")


def test_layered_model(vel1d, den1d, test_depths, verbose=True, side='right'):
    """
    Test function to verify layered model implementation.
    
    Parameters:
    -----------
    vel1d, den1d : pd.DataFrame
        1D models to test
    test_depths : array
        Depths to test
    verbose : bool
        Print detailed results
    """
    
    if verbose:
        print("Testing layered model implementation:")
        print("1D Velocity Model layers:")
        for i in range(len(vel1d)):
            if i == len(vel1d) - 1:
                # Last layer extends to negative infinity
                print(f"  Layer {i}: z ≤ {vel1d.iloc[i]['z']:.0f} m (deepest) → vs = {vel1d.iloc[i]['vs']:.3f} km/s")
            else:
                # Regular layer boundaries
                current_z = vel1d.iloc[i]['z']
                next_z = vel1d.iloc[i+1]['z']
                print(f"  Layer {i}: {next_z:.0f} m < z ≤ {current_z:.0f} m → vs = {vel1d.iloc[i]['vs']:.3f} km/s")
        
        print("\n1D Density Model layers:")
        for i in range(len(den1d)):
            if i == len(den1d) - 1:
                # Last layer extends to negative infinity
                print(f"  Layer {i}: z ≤ {den1d.iloc[i]['z']:.0f} m (deepest) → density = {den1d.iloc[i]['den']:.1f} kg/m³")
            else:
                # Regular layer boundaries
                current_z = den1d.iloc[i]['z']
                next_z = den1d.iloc[i+1]['z']
                print(f"  Layer {i}: {next_z:.0f} m < z ≤ {current_z:.0f} m → density = {den1d.iloc[i]['den']:.1f} kg/m³")
    
    # Test velocity lookup
    vs_test  = get_layered_1d_value(test_depths, vel1d['z'].values, vel1d['vs'].values,  side=side)
    den_test = get_layered_1d_value(test_depths, den1d['z'].values, den1d['den'].values, side=side)

    if verbose:
        print(f"\nTest Results:")
        for i, depth in enumerate(test_depths):
            print(f"  Depth {depth:.0f} m → vs = {vs_test[i]:.3f} km/s, density = {den_test[i]:.1f} kg/m³")

    return vs_test, den_test


def visualize_field_slices(function, mesh, field_name="Field", z_levels=None):
    """
    Create simple visualization of field at different depth slices.
    """
    
    try:
        import matplotlib.pyplot as plt
        
        if z_levels is None:
            mesh_coords = mesh.coordinates()
            z_min, z_max = mesh_coords[:, 2].min(), mesh_coords[:, 2].max()
            z_levels = np.linspace(z_min, z_max, 5)  # 5 slices
        
        print(f"Creating visualization slices for {field_name} at z = {z_levels}")
        print("For detailed visualization, save functions and use ParaView:")
        print("  paraview vs_meshname.xdmf density_meshname.xdmf shear_modulus_meshname.xdmf")
        
    except ImportError:
        print("Matplotlib not available. Use ParaView for visualization:")
        print("  paraview vs_meshname.xdmf density_meshname.xdmf shear_modulus_meshname.xdmf")


    # Example usage - complete workflow:
    """
    # Step 1: Process velocity models with correct layered implementation
    vs_func, den_func, mu_func, CG_mu = process_velocity_models(vel3d, vel1d, den1d, mesh, verbose=True)

    # Step 2: Test the layered model (optional)
    test_depths = np.array([-1000, -5000, -50000, -150000, -800000])  # Test various depths
    test_layered_model(vel1d, den1d, test_depths, verbose=True)

    # Step 3: Use shear modulus directly in your PDE
    mtrue_mu_fun = mu_func  # Contains shear modulus in GPa

    # Step 4: Save functions
    save_functions_to_file(vs_func, den_func, mu_func, 
                        output_dir="./velocity_fields/", 
                        meshname="nicoya2")

    # Step 5: Use in PDE workflow (no changes needed)
    pde_varf = PDEVarf(mtrue_mu_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    """


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

        detected_nodes = inv_vals >= threshold * true_vals
        fraction = detected_nodes.mean()

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
    def summary(self, mask, col='s_dip', type='pattern'):
        detection_frac, detection_pass = self.detection_rate(mask, col=col)
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
        print(mask.sum())
        
        detected = p[mask] >= threshold * t[mask]
        frac = detected.mean()
        return frac, frac >= min_fraction

    def summary(self, col='s_dip'):
        zf = self.loc_f['zf'].to_numpy()
        det_frac, det_pass = self.detection_rate(col=col)

        metrics = {
            "NumNodes": len(self.loc_f),
            "DepthRange": f"{zf.min():.2f} – {zf.max():.2f}",
            "RMSE": self.rmse(col=col),
            "DetectionRate": det_frac,
            "DetectionPass": det_pass
        }

        return pd.DataFrame(metrics, index=[0]).T.rename(columns={0: "Value"})
    

def rot_xy(x, y, rot):
    x = np.asarray(x)
    y = np.asarray(y)
    cos_rot = np.cos(np.radians(rot))
    sin_rot = np.sin(np.radians(rot))
    x_rot = x * cos_rot + y * sin_rot
    y_rot = -x * sin_rot + y * cos_rot

    return x_rot, y_rot

def rot_xyerror(x_err, y_err, rot):
    x_err = np.asarray(x_err)
    y_err = np.asarray(y_err)
    cos_rot = np.cos(np.radians(rot))
    sin_rot = np.sin(np.radians(rot))
    x_err_rot = np.sqrt(x_err**2 * cos_rot**2 + y_err**2 * sin_rot**2)
    y_err_rot = np.sqrt(x_err**2 * sin_rot**2 + y_err**2 * cos_rot**2)

    # Even if the original error are not correlated, they are afterwards
    # **Correlation after rotation:**
    # The correlation coefficient is: ρ_x'y' = sin(θ)cos(θ)(σ_N² - σ_E²)/(σ_x'·σ_y')
    # For 30°: ρ_x'y' = (√3/4)(σ_N² - σ_E²)/(σ_x'·σ_y')
    # For 45°: ρ_x'y' = (1/2)(σ_N² - σ_E²)/(σ_N² + σ_E²)

    return x_err_rot, y_err_rot

def compute_rotated_weights(sigma_e_sq, sigma_n_sq, rotation_angle_deg):
    """
    Compute weights for rotated coordinate system.
    
    Parameters:
    - sigma_e_sq: East variance (array or scalar)
    - sigma_n_sq: North variance (array or scalar)
    - rotation_angle_deg: CCW rotation angle in degrees from E-N to x-y
    
    Returns:
    - weight_x, weight_y: Diagonal weights for rotated system
    """
    theta = np.radians(rotation_angle_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Transformed variances after rotation
    sigma_x_sq = (sigma_e_sq * cos_theta**2 + sigma_n_sq * sin_theta**2)
    sigma_y_sq = (sigma_e_sq * sin_theta**2 + sigma_n_sq * cos_theta**2)
    sigma_xy = (sigma_n_sq - sigma_e_sq) * sin_theta * cos_theta
    
    # Initialize with diagonal approximation
    weight_x = 1.0 / sigma_x_sq
    weight_y = 1.0 / sigma_y_sq
    
    # Apply correction for correlation where significant
    correlation_coeff = sigma_xy / np.sqrt(sigma_x_sq * sigma_y_sq)
    
    # Find points where correlation is significant
    significant_corr = np.abs(correlation_coeff) > 0.1
    
    if np.any(significant_corr):
        # Use true inverse covariance diagonal elements for these points
        det_C = sigma_x_sq * sigma_y_sq - sigma_xy**2
        inv_C11 = sigma_y_sq / det_C
        inv_C22 = sigma_x_sq / det_C
        
        # Apply correction only where needed
        weight_x = np.where(significant_corr, inv_C11, weight_x)
        weight_y = np.where(significant_corr, inv_C22, weight_y)
        
        print(f"Applied correlation correction to {np.sum(significant_corr)} out of {len(correlation_coeff)} points")
        print(f"Max correlation coefficient: {np.max(np.abs(correlation_coeff)):.3f}")
    
    return weight_x, weight_y


def process_velocity_models_hull(vel3d, vel1d, den1d, mesh, CG_mu_degree=1, verbose=True, side='right'):
    """
    Process 3D and 1D velocity models with density and project to FEniCS mesh.

    Uses CONVEX HULL instead of rectangular bounding box to determine the 3D model domain.
    This correctly handles rotated grids (e.g., 45° rotation) that would otherwise have
    ~50% empty corners in a rectangular bounding box.

    Key difference from process_velocity_models():
    - Uses scipy.spatial.Delaunay to build convex hull of 3D data points
    - DOFs inside the convex hull use 3D interpolated velocity
    - DOFs outside the convex hull use 1D layered velocity
    - No nearest-neighbor fallback needed (all hull-interior points have valid interpolation)

    Parameters:
    -----------
    vel3d : pd.DataFrame
        3D velocity model with columns: x, y, z (m), vs (km/s)
    vel1d : pd.DataFrame
        1D velocity model with columns: z (m), vs (km/s)
    den1d : pd.DataFrame
        1D density model with columns: z (m), den (kg/m³)
    mesh : dolfin.Mesh
        FEniCS mesh object
    CG_mu_degree : int, default=1
        Degree of CG function space for shear modulus
    verbose : bool, default=True
        If True, print detailed progress information

    Returns:
    --------
    vs_function : dolfin.Function
        Shear velocity field on mesh (km/s)
    density_function : dolfin.Function
        Density field on mesh (kg/m³)
    shear_modulus_function : dolfin.Function
        Shear modulus field on mesh (GPa)
    CG_mu : dolfin.FunctionSpace
        Function space used
    """
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    if verbose:
        print("=" * 70)
        print("Velocity Model Processing (Convex Hull Method)")
        print("=" * 70)

    # Step 1: Clean the data - remove any NaN values
    vel3d_clean = vel3d.dropna(subset=['x', 'y', 'z', 'vs']).copy()
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).copy()
    den1d_clean = den1d.dropna(subset=['z', 'den']).copy()

    if verbose:
        print(f"\nStep 1: Data cleaning")
        print(f"  3D model: {len(vel3d_clean)} points")
        print(f"  1D velocity: {len(vel1d_clean)} layers")
        print(f"  1D density: {len(den1d_clean)} layers")

    # Check if we have enough data
    if len(vel3d_clean) == 0:
        raise ValueError("No valid data points in 3D model after removing NaN values")
    if len(vel1d_clean) == 0:
        raise ValueError("No valid data points in 1D velocity model after removing NaN values")
    if len(den1d_clean) == 0:
        raise ValueError("No valid data points in 1D density model after removing NaN values")

    # Step 2: Sort 1D models by depth (shallow to deep: 0, -5000, -9000, ...)
    vel1d_clean = vel1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d_clean.sort_values('z', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\nStep 2: Sort 1D models by depth (shallow to deep)")
        print(f"  Velocity: {vel1d_clean['z'].iloc[0]:.0f} to {vel1d_clean['z'].iloc[-1]:.0f} m")
        print(f"  Density: {den1d_clean['z'].iloc[0]:.0f} to {den1d_clean['z'].iloc[-1]:.0f} m")

    # Step 3: Build Delaunay triangulation (convex hull) from 3D data points
    if verbose:
        print(f"\nStep 3: Build Delaunay triangulation for convex hull test")

    data_points = vel3d_clean[['x', 'y', 'z']].values
    try:
        hull = Delaunay(data_points)
        if verbose:
            print(f"  Delaunay triangulation built with {len(hull.simplices)} simplices")
    except Exception as e:
        raise ValueError(f"Failed to build Delaunay triangulation: {e}")

    # Also compute bounding box for reference
    x_min, x_max = vel3d_clean['x'].min(), vel3d_clean['x'].max()
    y_min, y_max = vel3d_clean['y'].min(), vel3d_clean['y'].max()
    z_min, z_max = vel3d_clean['z'].min(), vel3d_clean['z'].max()

    if verbose:
        print(f"\nStep 4: 3D model extent")
        print(f"  X: {x_min/1e3:.1f} to {x_max/1e3:.1f} km")
        print(f"  Y: {y_min/1e3:.1f} to {y_max/1e3:.1f} km")
        print(f"  Z: {z_min/1e3:.1f} to {z_max/1e3:.1f} km")

    # Step 5: Create FEniCS function space and functions
    if verbose:
        print(f"\nStep 5: Create FEniCS functions")

    if CG_mu_degree != 1:
        print(f"  Using CG function space of degree {CG_mu_degree}")

    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_degree)

    vs_function = dl.Function(CG_mu)
    density_function = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)

    # Get DOF coordinates
    V_coords = CG_mu.tabulate_dof_coordinates()
    n_dofs = len(V_coords)

    if verbose:
        print(f"  Function space CG{CG_mu_degree} has {n_dofs} DOFs")

    # Step 6: Apply 1D layered models to ALL DOFs (background)
    if verbose:
        print(f"\nStep 6: Apply 1D layered models to all DOFs (background)")

    vs_at_dofs = get_layered_1d_value(
        V_coords[:, 2],
        vel1d_clean['z'].values,
        vel1d_clean['vs'].values,
        side=side
    )

    # NOTE: den1d['den'] must be in kg/m³ (caller is responsible for unit conversion).
    # Calling script converts in-place: den1d['den'] = den1d['den'] * 1e3 (g/cm³ → kg/m³).
    density_at_dofs = get_layered_1d_value(
        V_coords[:, 2],
        den1d_clean['z'].values,
        den1d_clean['den'].values,
        side=side
    )

    if verbose:
        print(f"  1D velocity range: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  1D density range: {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")

    # Verification: horizontal slices at same depth should be laterally homogeneous
    if verbose:
        print(f"\n  Verification: checking lateral homogeneity of 1D assignment")
        # Group DOFs by z-coordinate (round to 0.1 m to handle floating-point)
        z_rounded = np.round(V_coords[:, 2], 1)
        unique_z = np.unique(z_rounded)

        # Check max variance within each z-slice
        max_vs_var = 0.0
        max_den_var = 0.0
        worst_z_vs = None
        worst_z_den = None

        for z in unique_z:
            mask = (z_rounded == z)
            vs_var = np.var(vs_at_dofs[mask])
            den_var = np.var(density_at_dofs[mask])
            if vs_var > max_vs_var:
                max_vs_var = vs_var
                worst_z_vs = z
            if den_var > max_den_var:
                max_den_var = den_var
                worst_z_den = z

        print(f"    Checked {len(unique_z)} unique depth levels")
        print(f"    Max vs variance within z-slices: {max_vs_var:.2e} (at z={worst_z_vs} m)")
        print(f"    Max density variance within z-slices: {max_den_var:.2e} (at z={worst_z_den} m)")

        if max_vs_var < 1e-10 and max_den_var < 1e-10:
            print(f"    ✓ PASSED: 1D assignment is laterally homogeneous")
        else:
            print(f"    ✗ WARNING: Non-zero variance detected - check layer boundaries")

    # Step 7: Determine which DOFs are inside the CONVEX HULL of 3D data
    # This is the key difference from the original function!
    if verbose:
        print(f"\nStep 7: Classify DOFs using convex hull (NOT bounding box)")

    # find_simplex returns -1 for points outside the convex hull
    inside_3d_dofs = hull.find_simplex(V_coords) >= 0

    n_inside_dofs = np.sum(inside_3d_dofs)
    n_outside_dofs = np.sum(~inside_3d_dofs)

    # Compare with what bounding box would give
    inside_bbox = ((V_coords[:, 0] >= x_min) & (V_coords[:, 0] <= x_max) &
                   (V_coords[:, 1] >= y_min) & (V_coords[:, 1] <= y_max) &
                   (V_coords[:, 2] >= z_min) & (V_coords[:, 2] <= z_max))
    n_inside_bbox = np.sum(inside_bbox)

    if verbose:
        print(f"  DOFs inside convex hull: {n_inside_dofs} ({100*n_inside_dofs/n_dofs:.1f}%)")
        print(f"  DOFs outside convex hull: {n_outside_dofs}")
        print(f"  (For comparison, bounding box would include {n_inside_bbox} DOFs)")
        print(f"  Convex hull excludes {n_inside_bbox - n_inside_dofs} DOFs from empty corners")

    # Step 8: Override with 3D model where inside convex hull
    if n_inside_dofs > 0:
        if verbose:
            print(f"\nStep 8: Apply 3D interpolated model inside convex hull")

        # Interpolate 3D velocity using LinearNDInterpolator, passing the already-built
        # Delaunay triangulation (hull) to avoid rebuilding it from scratch.
        interp_vs = LinearNDInterpolator(hull, vel3d_clean['vs'].values)
        vs_3d = interp_vs(V_coords[inside_3d_dofs])

        # Check for any NaN values (should be very few if any, since we're inside hull)
        nan_mask = np.isnan(vs_3d)
        n_nan = np.sum(nan_mask)

        if n_nan > 0:
            if verbose:
                print(f"  Warning: {n_nan} NaN values inside hull ({100*n_nan/n_inside_dofs:.2f}%)")
                print(f"  These will retain 1D background values (no nearest-neighbor fallback)")
            # DOFs with NaN keep their 1D values - we don't use nearest neighbor
            vs_3d[nan_mask] = vs_at_dofs[inside_3d_dofs][nan_mask]

        # Override 1D values with 3D interpolated values
        vs_at_dofs[inside_3d_dofs] = vs_3d

        if verbose:
            print(f"  3D velocity range (inside hull): {vs_3d[~nan_mask].min():.3f} to {vs_3d[~nan_mask].max():.3f} km/s")

    # Step 9: Assign values to FEniCS functions
    vs_function.vector()[:] = vs_at_dofs
    density_function.vector()[:] = density_at_dofs

    # Step 10: Compute shear modulus (μ = ρ × vs²)
    if verbose:
        print(f"\nStep 10: Compute shear modulus field (μ = ρ × vs²)")

    vs_ms = vs_at_dofs * 1000.0  # km/s to m/s
    shear_modulus_pa = density_at_dofs * vs_ms**2  # kg/m³ × (m/s)² = Pa
    shear_modulus_gpa = shear_modulus_pa / 1e9  # Pa to GPa

    shear_modulus_function.vector()[:] = shear_modulus_gpa

    # Step 11: Summary
    if verbose:
        print(f"\n" + "=" * 70)
        print("Processing complete! Summary:")
        print(f"  Final velocity: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  Final density: {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")
        print(f"  Final shear modulus: {shear_modulus_gpa.min():.2f} to {shear_modulus_gpa.max():.2f} GPa")
        print("=" * 70)

    return vs_function, density_function, shear_modulus_function, CG_mu

def process_velocity_models_hull_interp(vel3d, vel1d, den1d, mesh, CG_mu_degree=1, verbose=True):
    """
    Variant of process_velocity_models_hull that uses LINEAR INTERPOLATION for
    the 1D background model, consistent with how SIMULPS13Q parameterizes velocity.

    Key difference from process_velocity_models_hull():
    - 1D model: np.interp() (linear interpolation between nodes) instead of
                get_layered_1d_value() (step/constant-layer lookup)
    - 3D model: identical — scipy.griddata(method='linear') trilinear interpolation
                inside the convex hull of the 3D data points

    Motivation (DeShon et al. 2006 / How.md):
        "The depths in Table 1 are grid node positions at which velocity is
        specified, with linear interpolation between adjacent nodes."
        SIMULPS13Q never holds a constant velocity within a depth interval;
        every point is a linear blend of its two bounding nodes.

    Parameters:
    -----------
    vel3d : pd.DataFrame
        3D velocity model with columns: x, y, z (m), vs (km/s)
    vel1d : pd.DataFrame
        1D velocity model with columns: z (m), vs (km/s)
        Depths must be negative (below sea level) and sorted shallow-to-deep.
    den1d : pd.DataFrame
        1D density model with columns: z (m), den (kg/m³)
    mesh : dolfin.Mesh
        FEniCS mesh object
    CG_mu_degree : int, default=1
        Degree of CG function space for shear modulus
    verbose : bool, default=True
        If True, print detailed progress information

    Returns:
    --------
    vs_function : dolfin.Function
        Shear velocity field on mesh (km/s)
    density_function : dolfin.Function
        Density field on mesh (kg/m³)
    shear_modulus_function : dolfin.Function
        Shear modulus field on mesh (GPa)
    CG_mu : dolfin.FunctionSpace
        Function space used
    """
    from scipy.spatial import Delaunay

    if verbose:
        print("=" * 70)
        print("Velocity Model Processing (Convex Hull + Linear Interp for 1D)")
        print("=" * 70)

    # Step 1: Clean the data - remove any NaN values
    vel3d_clean = vel3d.dropna(subset=['x', 'y', 'z', 'vs']).copy()
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).copy()
    den1d_clean = den1d.dropna(subset=['z', 'den']).copy()

    if verbose:
        print(f"\nStep 1: Data cleaning")
        print(f"  3D model: {len(vel3d_clean)} points")
        print(f"  1D velocity: {len(vel1d_clean)} layers")
        print(f"  1D density: {len(den1d_clean)} layers")

    if len(vel3d_clean) == 0:
        raise ValueError("No valid data points in 3D model after removing NaN values")
    if len(vel1d_clean) == 0:
        raise ValueError("No valid data points in 1D velocity model after removing NaN values")
    if len(den1d_clean) == 0:
        raise ValueError("No valid data points in 1D density model after removing NaN values")

    # Step 2: Sort 1D models by depth (shallow to deep: 0, -5000, -9000, ...)
    vel1d_clean = vel1d_clean.sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d_clean.sort_values('z', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\nStep 2: Sort 1D models by depth (shallow to deep)")
        print(f"  Velocity: {vel1d_clean['z'].iloc[0]:.0f} to {vel1d_clean['z'].iloc[-1]:.0f} m")
        print(f"  Density: {den1d_clean['z'].iloc[0]:.0f} to {den1d_clean['z'].iloc[-1]:.0f} m")

    # Step 3: Build Delaunay triangulation (convex hull) from 3D data points
    if verbose:
        print(f"\nStep 3: Build Delaunay triangulation for convex hull test")

    data_points = vel3d_clean[['x', 'y', 'z']].values
    try:
        hull = Delaunay(data_points)
        if verbose:
            print(f"  Delaunay triangulation built with {len(hull.simplices)} simplices")
    except Exception as e:
        raise ValueError(f"Failed to build Delaunay triangulation: {e}")

    x_min, x_max = vel3d_clean['x'].min(), vel3d_clean['x'].max()
    y_min, y_max = vel3d_clean['y'].min(), vel3d_clean['y'].max()
    z_min, z_max = vel3d_clean['z'].min(), vel3d_clean['z'].max()

    if verbose:
        print(f"\nStep 4: 3D model extent")
        print(f"  X: {x_min/1e3:.1f} to {x_max/1e3:.1f} km")
        print(f"  Y: {y_min/1e3:.1f} to {y_max/1e3:.1f} km")
        print(f"  Z: {z_min/1e3:.1f} to {z_max/1e3:.1f} km")

    # Step 5: Create FEniCS function space and functions
    if verbose:
        print(f"\nStep 5: Create FEniCS functions")

    if CG_mu_degree != 1:
        print(f"  Using CG function space of degree {CG_mu_degree}")

    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_degree)

    vs_function            = dl.Function(CG_mu)
    density_function       = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)

    V_coords = CG_mu.tabulate_dof_coordinates()
    n_dofs   = len(V_coords)

    if verbose:
        print(f"  Function space CG{CG_mu_degree} has {n_dofs} DOFs")

    # Step 6: Apply 1D models to ALL DOFs using LINEAR INTERPOLATION
    # np.interp requires xp to be increasing, so reverse the shallow-to-deep
    # sorted arrays (descending z) to make them ascending.
    # Values outside the node range are clamped to the boundary node values,
    # matching SIMULPS13Q's fixed boundary-node behaviour.
    if verbose:
        print(f"\nStep 6: Apply 1D models via linear interpolation (SIMULPS13Q convention)")

    z_nodes_vel = vel1d_clean['z'].values[::-1]   # ascending (deep → shallow)
    vs_nodes    = vel1d_clean['vs'].values[::-1]
    z_nodes_den = den1d_clean['z'].values[::-1]
    den_nodes   = den1d_clean['den'].values[::-1]

    vs_at_dofs      = np.interp(V_coords[:, 2], z_nodes_vel, vs_nodes)
    density_at_dofs = np.interp(V_coords[:, 2], z_nodes_den, den_nodes)

    if verbose:
        print(f"  1D velocity range: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  1D density range: {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")

    # Verification: horizontal slices at same depth should be laterally homogeneous
    if verbose:
        print(f"\n  Verification: checking lateral homogeneity of 1D assignment")
        z_rounded = np.round(V_coords[:, 2], 1)
        unique_z  = np.unique(z_rounded)

        max_vs_var  = 0.0
        max_den_var = 0.0
        worst_z_vs  = None
        worst_z_den = None

        for z in unique_z:
            mask    = (z_rounded == z)
            vs_var  = np.var(vs_at_dofs[mask])
            den_var = np.var(density_at_dofs[mask])
            if vs_var > max_vs_var:
                max_vs_var = vs_var
                worst_z_vs = z
            if den_var > max_den_var:
                max_den_var = den_var
                worst_z_den = z

        print(f"    Checked {len(unique_z)} unique depth levels")
        print(f"    Max vs variance within z-slices: {max_vs_var:.2e} (at z={worst_z_vs} m)")
        print(f"    Max density variance within z-slices: {max_den_var:.2e} (at z={worst_z_den} m)")

        if max_vs_var < 1e-10 and max_den_var < 1e-10:
            print(f"    ✓ PASSED: 1D assignment is laterally homogeneous")
        else:
            print(f"    ✗ WARNING: Non-zero variance detected - check layer boundaries")

    # Step 7: Classify DOFs inside the convex hull of the 3D data
    if verbose:
        print(f"\nStep 7: Classify DOFs using convex hull (NOT bounding box)")

    inside_3d_dofs = hull.find_simplex(V_coords) >= 0
    n_inside_dofs  = np.sum(inside_3d_dofs)
    n_outside_dofs = np.sum(~inside_3d_dofs)

    inside_bbox = ((V_coords[:, 0] >= x_min) & (V_coords[:, 0] <= x_max) &
                   (V_coords[:, 1] >= y_min) & (V_coords[:, 1] <= y_max) &
                   (V_coords[:, 2] >= z_min) & (V_coords[:, 2] <= z_max))
    n_inside_bbox = np.sum(inside_bbox)

    if verbose:
        print(f"  DOFs inside convex hull: {n_inside_dofs} ({100*n_inside_dofs/n_dofs:.1f}%)")
        print(f"  DOFs outside convex hull: {n_outside_dofs}")
        print(f"  (For comparison, bounding box would include {n_inside_bbox} DOFs)")
        print(f"  Convex hull excludes {n_inside_bbox - n_inside_dofs} DOFs from empty corners")

    # Step 8: Override with 3D model where inside convex hull
    # (identical to process_velocity_models_hull — trilinear interpolation)
    if n_inside_dofs > 0:
        if verbose:
            print(f"\nStep 8: Apply 3D interpolated model inside convex hull")

        vs_3d = griddata(
            points=data_points,
            values=vel3d_clean['vs'].values,
            xi=V_coords[inside_3d_dofs],
            method='linear',
            fill_value=np.nan
        )

        nan_mask = np.isnan(vs_3d)
        n_nan    = np.sum(nan_mask)

        if n_nan > 0:
            if verbose:
                print(f"  Warning: {n_nan} NaN values inside hull ({100*n_nan/n_inside_dofs:.2f}%)")
                print(f"  These will retain 1D background values (no nearest-neighbor fallback)")
            vs_3d[nan_mask] = vs_at_dofs[inside_3d_dofs][nan_mask]

        vs_at_dofs[inside_3d_dofs] = vs_3d

        if verbose:
            print(f"  3D velocity range (inside hull): {vs_3d[~nan_mask].min():.3f} to {vs_3d[~nan_mask].max():.3f} km/s")

    # Step 9: Assign values to FEniCS functions
    vs_function.vector()[:]      = vs_at_dofs
    density_function.vector()[:] = density_at_dofs

    # Step 10: Compute shear modulus (mu = rho x vs^2)
    if verbose:
        print(f"\nStep 10: Compute shear modulus field (mu = rho x vs^2)")

    vs_ms             = vs_at_dofs * 1000.0         # km/s to m/s
    shear_modulus_pa  = density_at_dofs * vs_ms**2  # kg/m^3 x (m/s)^2 = Pa
    shear_modulus_gpa = shear_modulus_pa / 1e9      # Pa to GPa

    shear_modulus_function.vector()[:] = shear_modulus_gpa

    # Step 11: Summary
    if verbose:
        print(f"\n" + "=" * 70)
        print("Processing complete! Summary:")
        print(f"  Final velocity: {vs_at_dofs.min():.3f} to {vs_at_dofs.max():.3f} km/s")
        print(f"  Final density: {density_at_dofs.min():.1f} to {density_at_dofs.max():.1f} kg/m³")
        print(f"  Final shear modulus: {shear_modulus_gpa.min():.2f} to {shear_modulus_gpa.max():.2f} GPa")
        print("=" * 70)

    return vs_function, density_function, shear_modulus_function, CG_mu


def scale_shear_modulus_by_1d(mu_func, vel1d, den1d, contrast_factor, min_mu=None, verbose=True, side='right'):
    """
    Scale a shear modulus FEniCS Function by amplifying lateral anomalies
    relative to the depth-dependent 1D reference model.

    For each DOF at depth z:
        mu_ref(z)  = den_1d(z) * (vs_1d(z) * 1e3)^2 / 1e9   [GPa]
        mu_scaled  = mu_ref(z) + contrast_factor * (mu_func - mu_ref(z))
        if min_mu is not None: mu_scaled = max(mu_scaled, min_mu)

    The 1D reference uses a step-function (constant-layer) lookup, NOT
    interpolation.  At exact layer boundaries the deeper layer value is used
    (consistent with get_layered_1d_value).

    Compatible with any mu_func origin (process_velocity_models_hull, etc.).

    Parameters
    ----------
    mu_func : dolfin.Function
        Original shear modulus field in GPa.
    vel1d : pd.DataFrame
        1D velocity model. Columns: z (m, negative downward), vs (km/s).
        Sorted shallow to deep (e.g. z = 0, -5000, -9000, ...).
    den1d : pd.DataFrame
        1D density model. Columns: z (m, negative downward), den (kg/m³).
        Sorted shallow to deep.
    contrast_factor : float
        Amplification factor for lateral anomalies (1.0 = no change).
    min_mu : float or None, optional
        Minimum physical shear modulus in GPa (floor after scaling).
        Default None (no clipping).
    verbose : bool, optional
        Print summary statistics. Default True.

    Returns
    -------
    mu_scaled : dolfin.Function
        Scaled shear modulus field in GPa, in the same function space as mu_func.
    """
    import numpy as np
    import dolfin as dl

    # ── DOF coordinates ───────────────────────────────────────────────────────
    V      = mu_func.function_space()
    coords = V.tabulate_dof_coordinates()   # (N_dofs, 3), metres
    z_dofs = coords[:, 2]                   # negative downward

    # ── 1D reference mu at each DOF depth ────────────────────────────────────
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).reset_index(drop=True)
    den1d_clean = den1d.dropna(subset=['z', 'den']).reset_index(drop=True)

    vs_1d_at_dofs  = get_layered_1d_value(z_dofs,
                                          vel1d_clean['z'].values,
                                          vel1d_clean['vs'].values,
                                          side=side)
    den_1d_at_dofs = get_layered_1d_value(z_dofs,
                                          den1d_clean['z'].values,
                                          den1d_clean['den'].values,
                                          side=side)

    # mu_ref(z) = rho x Vs^2  (Pa -> GPa)
    mu_ref_z = den_1d_at_dofs * (vs_1d_at_dofs * 1e3) ** 2 / 1e9

    # ── Scale ─────────────────────────────────────────────────────────────────
    mu_orig = mu_func.vector()[:]
    mu_new  = mu_ref_z + contrast_factor * (mu_orig - mu_ref_z)
    if min_mu is not None:
        mu_new = np.maximum(mu_new, min_mu)

    # ── Assign to new FEniCS Function ─────────────────────────────────────────
    mu_scaled = dl.Function(V)
    mu_scaled.vector()[:] = mu_new

    if verbose:
        anomaly = mu_orig - mu_ref_z
        print(f"scale_shear_modulus_by_1d: contrast_factor = {contrast_factor}")
        print(f"  Original mu      : {mu_orig.min():.2f} - {mu_orig.max():.2f} GPa")
        print(f"  1D ref mu(z)     : {mu_ref_z.min():.2f} - {mu_ref_z.max():.2f} GPa")
        print(f"  Lateral anomaly  : {anomaly.min():.2f} - {anomaly.max():.2f} GPa")
        print(f"  Scaled mu        : {mu_new.min():.2f} - {mu_new.max():.2f} GPa")
        if min_mu is not None:
            n_clipped = int(np.sum(mu_new == min_mu))
            if n_clipped:
                print(f"  Clipped to min_mu={min_mu} GPa: {n_clipped} DOFs")

    return mu_scaled


def scale_shear_modulus_by_1d_interp(mu_func, vel1d, den1d, contrast_factor, min_mu=None, verbose=True):
    """
    Scale a shear modulus FEniCS Function by amplifying lateral anomalies
    relative to a **linearly-interpolated** depth-dependent 1D reference model.

    Intended to be paired with process_velocity_models_hull_interp(), which also
    uses linear interpolation for the 1D background.  This ensures that DOFs
    outside the 3D hull (whose values equal the linear-interp 1D value) have
    zero anomaly and are therefore unaffected by scaling.

    For each DOF at depth z:
        mu_ref(z)  = den_1d_interp(z) * (vs_1d_interp(z) * 1e3)^2 / 1e9   [GPa]
        mu_scaled  = mu_ref(z) + contrast_factor * (mu_func - mu_ref(z))
        mu_scaled  = max(mu_scaled, min_mu)

    The 1D reference uses linear interpolation between tabulated nodes
    (via get_layered_1d_value_interp / np.interp), consistent with
    process_velocity_models_hull_interp().

    Compare with scale_shear_modulus_by_1d() which uses a step-function lookup
    and is paired with process_velocity_models_hull().

    Parameters
    ----------
    mu_func : dolfin.Function
        Original shear modulus field in GPa (output of process_velocity_models_hull_interp).
    vel1d : pd.DataFrame
        1D velocity model. Columns: z (m, negative downward), vs (km/s).
        Sorted shallow to deep (e.g. z = 0, -5000, -9000, ...).
    den1d : pd.DataFrame
        1D density model. Columns: z (m, negative downward), den (kg/m³).
        Sorted shallow to deep.
    contrast_factor : float
        Amplification factor for lateral anomalies (1.0 = no change).
    min_mu : float or None, optional
        Minimum physical shear modulus in GPa (floor after scaling).
        Default None (no clipping).
    verbose : bool, optional
        Print summary statistics. Default True.

    Returns
    -------
    mu_scaled : dolfin.Function
        Scaled shear modulus field in GPa, in the same function space as mu_func.
    """
    import numpy as np
    import dolfin as dl

    # ── DOF coordinates ───────────────────────────────────────────────────────
    V      = mu_func.function_space()
    coords = V.tabulate_dof_coordinates()   # (N_dofs, 3), metres
    z_dofs = coords[:, 2]                   # negative downward

    # ── 1D reference mu at each DOF depth (linear interpolation) ─────────────
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).reset_index(drop=True)
    den1d_clean = den1d.dropna(subset=['z', 'den']).reset_index(drop=True)

    vs_1d_at_dofs  = get_layered_1d_value_interp(z_dofs,
                                                     vel1d_clean['z'].values,
                                                     vel1d_clean['vs'].values)
    den_1d_at_dofs = get_layered_1d_value_interp(z_dofs,
                                                     den1d_clean['z'].values,
                                                     den1d_clean['den'].values)

    # mu_ref(z) = rho x Vs^2  (Pa -> GPa)
    mu_ref_z = den_1d_at_dofs * (vs_1d_at_dofs * 1e3) ** 2 / 1e9

    # ── Scale ─────────────────────────────────────────────────────────────────
    mu_orig = mu_func.vector()[:]
    mu_new  = mu_ref_z + contrast_factor * (mu_orig - mu_ref_z)
    if min_mu is not None:
        mu_new = np.maximum(mu_new, min_mu)

    # ── Assign to new FEniCS Function ─────────────────────────────────────────
    mu_scaled = dl.Function(V)
    mu_scaled.vector()[:] = mu_new

    if verbose:
        anomaly = mu_orig - mu_ref_z
        print(f"scale_shear_modulus_by_1d_interp: contrast_factor = {contrast_factor}")
        print(f"  Original mu      : {mu_orig.min():.2f} - {mu_orig.max():.2f} GPa")
        print(f"  1D ref mu(z)     : {mu_ref_z.min():.2f} - {mu_ref_z.max():.2f} GPa")
        print(f"  Lateral anomaly  : {anomaly.min():.2f} - {anomaly.max():.2f} GPa")
        print(f"  Scaled mu        : {mu_new.min():.2f} - {mu_new.max():.2f} GPa")
        if min_mu is not None:
            n_clipped = int(np.sum(mu_new == min_mu))
            if n_clipped:
                print(f"  Clipped to min_mu={min_mu} GPa: {n_clipped} DOFs")

    return mu_scaled


def build_1d_shear_modulus(vel1d, den1d, mesh, CG_mu_degree=1, method='step', verbose=True):
    """
    Build a purely 1D depth-layered shear modulus model on the FE mesh.

    Every DOF gets the 1D reference value at its depth — no 3D spatial variation.
    Use this as a baseline for comparing with heterogeneous model results.

    Paired methods (match the construction of the 3D model being compared):
        method='step'   -> consistent with process_velocity_models_hull
        method='interp' -> consistent with process_velocity_models_hull_interp

    Parameters
    ----------
    vel1d : pd.DataFrame
        1D velocity model. Columns: z (m, negative downward), vs (km/s).
    den1d : pd.DataFrame
        1D density model. Columns: z (m, negative downward), den (kg/m3).
    mesh : dolfin.Mesh
        FEniCS mesh object.
    CG_mu_degree : int, optional
        Degree of CG function space. Default 1.
    method : str, optional
        1D lookup method: 'step' (constant per layer, default) or 'interp'
        (linear interpolation between nodes).
    verbose : bool, optional
        Print summary statistics. Default True.

    Returns
    -------
    vs_function : dolfin.Function
        Shear velocity field (km/s) - laterally homogeneous.
    density_function : dolfin.Function
        Density field (kg/m3) - laterally homogeneous.
    shear_modulus_function : dolfin.Function
        Shear modulus field (GPa) - laterally homogeneous.
    CG_mu : dolfin.FunctionSpace
        Function space used.
    """
    import numpy as np
    import dolfin as dl

    # Clean and sort 1D models (shallow to deep)
    vel1d_clean = vel1d.dropna(subset=['z', 'vs']).sort_values('z', ascending=False).reset_index(drop=True)
    den1d_clean = den1d.dropna(subset=['z', 'den']).sort_values('z', ascending=False).reset_index(drop=True)

    # Build function space
    CG_mu  = dl.FunctionSpace(mesh, "CG", CG_mu_degree)
    coords = CG_mu.tabulate_dof_coordinates()   # (N_dofs, 3), metres
    z_dofs = coords[:, 2]                        # negative downward
    n_dofs = len(z_dofs)

    if verbose:
        print(f"build_1d_shear_modulus: method='{method}', CG{CG_mu_degree}, {n_dofs} DOFs")

    # Assign 1D values to every DOF
    if method == 'step':
        vs_at_dofs      = get_layered_1d_value(z_dofs,
                                               vel1d_clean['z'].values,
                                               vel1d_clean['vs'].values,
                                               side='right')
        density_at_dofs = get_layered_1d_value(z_dofs,
                                               den1d_clean['z'].values,
                                               den1d_clean['den'].values,
                                               side='right')
    else:  # 'interp'
        vs_at_dofs      = get_layered_1d_value_interp(z_dofs,
                                                      vel1d_clean['z'].values,
                                                      vel1d_clean['vs'].values)
        density_at_dofs = get_layered_1d_value_interp(z_dofs,
                                                      den1d_clean['z'].values,
                                                      den1d_clean['den'].values)

    # mu = rho * vs^2  (GPa)
    mu_at_dofs = density_at_dofs * (vs_at_dofs * 1e3) ** 2 / 1e9

    # Assign to FEniCS functions
    vs_function            = dl.Function(CG_mu)
    density_function       = dl.Function(CG_mu)
    shear_modulus_function = dl.Function(CG_mu)

    vs_function.vector()[:]            = vs_at_dofs
    density_function.vector()[:]       = density_at_dofs
    shear_modulus_function.vector()[:] = mu_at_dofs

    if verbose:
        print(f"  Vs  : {vs_at_dofs.min():.3f} - {vs_at_dofs.max():.3f} km/s")
        print(f"  rho : {density_at_dofs.min():.1f} - {density_at_dofs.max():.1f} kg/m3")
        print(f"  mu  : {mu_at_dofs.min():.2f} - {mu_at_dofs.max():.2f} GPa")

    return vs_function, density_function, shear_modulus_function, CG_mu


# ─────────────────────────────────────────────────────────────────────────────
# compute_fault_basis_per_vertex  (copied from Nankai utils.py, 2026-04-28)
# ─────────────────────────────────────────────────────────────────────────────
# Adapted to match Nicoya's `dir_strike(n)` formula (NO eps guard) so the
# returned basis vectors are sign-consistent with Nicoya's PDE weak form.
# Nankai's version had an eps guard for near-horizontal facets at the trench;
# Nicoya's mesh has no near-horizontal facets at the fault, so the eps is
# unnecessary and removing it ensures byte-perfect alignment with the
# inversion's (m_strike, m_dip) values.
#
# The returned arrays are row-aligned with the m_s_fault extraction used by
# the inversion scripts:
#       bc2 = DirichletBC(Vm, (99, 99), boundaries, fault); um2 = Function(Vm); bc2.apply(um2.vector())
#       m_s_fault = m[um2.vector() == 99]; m_sx_fault = m_s_fault[0::2]
# so row i of (coords, strike, dip) corresponds to (m_sx_fault[i], m_sy_fault[i]).
def compute_fault_basis_per_vertex(mesh, boundaries, Vm, fault_tag,
                                    coherence_tol=0.5, verbose=True):
    """Unit strike and dip direction vectors at each fault vertex.

    Uses UFL assembly with `FacetNormal(mesh)('+')` — identical to the
    `dir_strike(n)` / `dir_dip(n)` expressions in the inversion's weak form —
    so the returned basis is guaranteed sign-consistent with the stored
    `m_strike` and `m_dip`.  Reconstructing slip is a straight superposition:

        slip_xyz = m_strike * strike_hat + m_dip * dip_hat

    Per-vertex direction is the area-weighted average of the per-facet
    UFL expressions (CG1 test-function assembly gives A_f/3 weight per
    vertex per adjacent facet), then renormalized to unit length.

    Output ordering matches the `m_s_fault` extraction used by the
    inversion scripts:
        bc = DirichletBC(Vm, (99, 99), boundaries, fault_tag)
        slip_fault = phys_slip.vector()[um.vector() == 99]
        m_sx_fault = slip_fault[0::2]
    so row i of the returned arrays corresponds to `m_sx_fault[i]`,
    `m_sy_fault[i]`.

    Degenerate vertices — those whose average direction coherence
    `|Σ (A_f/3)·dir_f| / Σ (A_f/3)` falls below `coherence_tol` — are
    zeroed out and counted in the summary.  Expect coherence ≈ 1
    everywhere on a smooth slab.

    Parameters
    ----------
    mesh          : dolfin.Mesh
    boundaries    : MeshFunction of facet tags
    Vm            : VectorFunctionSpace (CG1, dim=2) used for the slip parameter
    fault_tag     : int, value in `boundaries` identifying fault facets
    coherence_tol : threshold below which a vertex is flagged degenerate
    verbose       : print a summary line

    Returns
    -------
    coords  : (M, 3) float64 — fault vertex (x, y, z) in mesh units (m)
    strike  : (M, 3) float64 — unit strike direction per vertex
    dip     : (M, 3) float64 — unit dip direction per vertex
    """
    import ufl

    # UFL per-facet directions, identical to Nicoya's weak-form dir_strike / dir_dip.
    # NOTE: Nicoya's dir_strike has NO eps guard (unlike Nankai). We match that here
    # so the basis is sign- and value-identical to what the PDE assembled against.
    n_plus = dl.FacetNormal(mesh)('+')
    z_dir  = dl.Constant((0., 0., 1.))
    n_x_z  = ufl.cross(n_plus, z_dir)
    strike_p = n_x_z / ufl.sqrt(ufl.dot(n_x_z, n_x_z))
    dip_p    = ufl.cross(strike_p, n_plus)

    dS_ = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)

    # Scalar CG1 test function drives per-vertex accumulation (mass = A_f/3)
    V1 = dl.FunctionSpace(mesh, "CG", 1)
    v1 = dl.TestFunction(V1)
    v2d_V1 = dl.vertex_to_dof_map(V1)  # length N; v2d_V1[v] = DOF idx at vertex v

    def _accum(restricted_scalar):
        """Per-vertex area-weighted sum of an already-restricted UFL scalar."""
        return dl.assemble(v1('+') * restricted_scalar * dS_(fault_tag)
                           ).get_local()[v2d_V1]

    strike_sum = np.stack([_accum(strike_p[i]) for i in range(3)], axis=1)  # (N, 3)
    dip_sum    = np.stack([_accum(dip_p[i])    for i in range(3)], axis=1)
    weight_sum = dl.assemble(v1('+') * dS_(fault_tag)).get_local()[v2d_V1]  # (N,)

    # Normalize per-vertex accumulators → unit vectors
    s_mags = np.linalg.norm(strike_sum, axis=1)
    d_mags = np.linalg.norm(dip_sum,    axis=1)
    strike_unit = np.zeros_like(strike_sum)
    dip_unit    = np.zeros_like(dip_sum)
    is_fault_vert = weight_sum > 1e-20
    for v in np.where(is_fault_vert)[0]:
        if s_mags[v] > 1e-12:
            strike_unit[v] = strike_sum[v] / s_mags[v]
        if d_mags[v] > 1e-12:
            dip_unit[v]    = dip_sum[v]    / d_mags[v]

    # Recover output order by replaying the m_s_fault extraction path
    bc    = dl.DirichletBC(Vm, dl.Constant((99., 99.)), boundaries, fault_tag)
    probe = dl.Function(Vm);  bc.apply(probe.vector())
    mask  = probe.vector().get_local() == 99.
    fault_dof_indices = np.where(mask)[0]        # 2M ascending DOF indices
    strike_dofs       = fault_dof_indices[0::2]  # M strike DOFs per m_sx_fault convention
    # dof_to_vertex_map on a VectorFunctionSpace returns scaled indices
    # (vertex * value_size + component); divide to recover mesh vertex id.
    d_scale           = Vm.num_sub_spaces()      # 2 for (strike, dip)
    vertex_order      = dl.dof_to_vertex_map(Vm)[strike_dofs] // d_scale

    coords_out = mesh.coordinates()[vertex_order]
    strike_out = strike_unit[vertex_order].copy()
    dip_out    = dip_unit[vertex_order].copy()

    # Coherence diagnostic + degenerate handling
    w = weight_sum[vertex_order]
    coh_s = np.where(w > 0, s_mags[vertex_order] / np.maximum(w, 1e-30), 0.)
    coh_d = np.where(w > 0, d_mags[vertex_order] / np.maximum(w, 1e-30), 0.)
    bad   = (coh_s < coherence_tol) | (coh_d < coherence_tol)
    strike_out[bad] = 0.0
    dip_out[bad]    = 0.0

    if verbose:
        print(f"Fault basis: {len(vertex_order)} vertices; "
              f"strike coherence min/max {coh_s.min():.3f}/{coh_s.max():.3f}, "
              f"dip coherence min/max {coh_d.min():.3f}/{coh_d.max():.3f}; "
              f"{int(bad.sum())} degenerate (zeroed, tol={coherence_tol}).")

    return coords_out, strike_out, dip_out
