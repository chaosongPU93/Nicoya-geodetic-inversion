"""
Smooth PyVista Visualization Functions (Option C - Part B)

These functions re-interpolate mesh-projected data onto regular grids
to remove triangular mesh artifacts while preserving heterogeneity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_horizontal_slice_pyvista_smooth(grid, field_name, depth_m, ax=None,
                                         vmin=None, vmax=None, cmap='gist_rainbow_r',
                                         add_colorbar=True, resolution=500,
                                         xlim=None, ylim=None):
    """
    Plot horizontal slice with smooth re-interpolation to remove mesh artifacts

    Parameters:
    -----------
    grid : pv.UnstructuredGrid
        PyVista grid with velocity data
    field_name : str
        Name of the field to plot
    depth_m : float
        Depth in meters (negative value)
    ax : matplotlib axes
        Axes to plot on
    vmin, vmax : float
        Color limits
    cmap : str
        Colormap name
    add_colorbar : bool
        Whether to add colorbar
    resolution : int
        Grid resolution for smooth interpolation (default 500)
    xlim : tuple (xmin, xmax)
        X-axis limits in km
    ylim : tuple (ymin, ymax)
        Y-axis limits in km

    Returns:
    --------
    CS : contourf object
    cbar : colorbar object or None
    """
    # Slice the grid
    normal = [0., 0., 1.]
    origin = [0., 0., depth_m]
    slice_mesh = grid.slice(normal=normal, origin=origin)

    if len(slice_mesh.points) == 0:
        print(f"No data at depth {depth_m/1e3:.1f} km")
        return None, None

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        standalone = True
    else:
        standalone = False

    # Extract coordinates and values from mesh
    x_mesh = slice_mesh.points[:, 0] / 1e3  # km
    y_mesh = slice_mesh.points[:, 1] / 1e3  # km

    if field_name not in slice_mesh.array_names:
        raise ValueError(f"Field '{field_name}' not found")

    values_mesh = slice_mesh[field_name]

    # Determine unit
    if field_name == 'shear modulus':
        unit = 'GPa'
    elif field_name == 'shear velocity Vs':
        unit = 'km/s'
    elif field_name == 'density':
        unit = 'kg/m³'
    else:
        unit = ''

    # Set color limits
    if vmin is None:
        vmin = values_mesh.min()
    if vmax is None:
        vmax = values_mesh.max()

    # Create regular grid for smooth visualization
    if xlim is not None:
        x_min, x_max = xlim
    else:
        x_min, x_max = x_mesh.min(), x_mesh.max()

    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_min, y_max = y_mesh.min(), y_mesh.max()

    x_regular = np.linspace(x_min, x_max, resolution)
    y_regular = np.linspace(y_min, y_max, resolution)
    X_grid, Y_grid = np.meshgrid(x_regular, y_regular)

    # Re-interpolate mesh data onto regular grid using cubic interpolation
    print(f"  Re-interpolating {len(values_mesh)} mesh points onto {resolution}x{resolution} regular grid...")
    Values_smooth = griddata(
        points=np.column_stack((x_mesh, y_mesh)),
        values=values_mesh,
        xi=(X_grid, Y_grid),
        method='cubic',  # Smooth cubic interpolation
        fill_value=np.nan
    )

    # Fill any remaining NaNs with linear interpolation
    nan_mask = np.isnan(Values_smooth)
    if np.any(nan_mask):
        Values_smooth_linear = griddata(
            points=np.column_stack((x_mesh, y_mesh)),
            values=values_mesh,
            xi=(X_grid, Y_grid),
            method='linear',
            fill_value=np.nan
        )
        Values_smooth[nan_mask] = Values_smooth_linear[nan_mask]

    # Plot smooth contours
    contour_levels = np.linspace(vmin, vmax, 256)
    CS = ax.contourf(X_grid, Y_grid, Values_smooth,
                     levels=contour_levels, cmap=cmap,
                     vmin=vmin, vmax=vmax, extend='both')

    # Add colorbar
    cbar = None
    if add_colorbar:
        cbar = plt.colorbar(CS, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f'{field_name} ({unit})', fontsize=12)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))

    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title(f'Horizontal Slice of {field_name} at Depth = {-depth_m/1e3:.1f} km (Smooth)\n({len(values_mesh)} points → {resolution}² grid)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if standalone:
        plt.tight_layout()
        plt.show()

    print(f"  {field_name} range: {np.nanmin(Values_smooth):.2f} - {np.nanmax(Values_smooth):.2f} {unit}")

    return CS, cbar


def plot_vertical_slice_pyvista_smooth(grid, field_name, position_m, direction='y',
                                       ax=None, vmin=None, vmax=None, cmap='gist_rainbow_r',
                                       add_colorbar=True, resolution=500):
    """
    Plot vertical slice with smooth re-interpolation

    Parameters:
    -----------
    grid : pv.UnstructuredGrid
        PyVista grid with velocity data
    field_name : str
        Name of the field to plot
    position_m : float
        Position in meters along slicing direction
    direction : str
        'x' or 'y' - direction perpendicular to slice
    ax : matplotlib axes
        Axes to plot on
    vmin, vmax : float
        Color limits
    cmap : str
        Colormap name
    add_colorbar : bool
        Whether to add colorbar
    resolution : int
        Grid resolution for smooth interpolation

    Returns:
    --------
    CS : contourf object
    cbar : colorbar object or None
    """
    # Define slice
    if direction == 'y':
        normal = [0., 1., 0.]
        origin = [0., position_m, 0.]
        coord_label = 'X'
        coord_idx = 0
    else:
        normal = [1., 0., 0.]
        origin = [position_m, 0., 0.]
        coord_label = 'Y'
        coord_idx = 1

    # Slice the grid
    slice_mesh = grid.slice(normal=normal, origin=origin)

    if len(slice_mesh.points) == 0:
        print(f"No data at {direction} = {position_m/1e3:.1f} km")
        return None, None

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        standalone = True
    else:
        standalone = False

    # Extract coordinates and values
    coord_mesh = slice_mesh.points[:, coord_idx] / 1e3  # km
    z_mesh = slice_mesh.points[:, 2] / 1e3  # km

    if field_name not in slice_mesh.array_names:
        raise ValueError(f"Field '{field_name}' not found")

    values_mesh = slice_mesh[field_name]

    # Determine unit
    if field_name == 'shear modulus':
        unit = 'GPa'
    elif field_name == 'shear velocity Vs':
        unit = 'km/s'
    elif field_name == 'density':
        unit = 'kg/m³'
    else:
        unit = ''

    # Set color limits
    if vmin is None:
        vmin = values_mesh.min()
    if vmax is None:
        vmax = values_mesh.max()

    # Create regular grid
    coord_min, coord_max = coord_mesh.min(), coord_mesh.max()
    z_min, z_max = z_mesh.min(), z_mesh.max()

    coord_regular = np.linspace(coord_min, coord_max, resolution)
    z_regular = np.linspace(z_min, z_max, resolution)
    Coord_grid, Z_grid = np.meshgrid(coord_regular, z_regular)

    # Re-interpolate
    print(f"  Re-interpolating {len(values_mesh)} mesh points onto {resolution}x{resolution} regular grid...")
    Values_smooth = griddata(
        points=np.column_stack((coord_mesh, z_mesh)),
        values=values_mesh,
        xi=(Coord_grid, Z_grid),
        method='cubic',
        fill_value=np.nan
    )

    # Fill NaNs with linear
    nan_mask = np.isnan(Values_smooth)
    if np.any(nan_mask):
        Values_smooth_linear = griddata(
            points=np.column_stack((coord_mesh, z_mesh)),
            values=values_mesh,
            xi=(Coord_grid, Z_grid),
            method='linear',
            fill_value=np.nan
        )
        Values_smooth[nan_mask] = Values_smooth_linear[nan_mask]

    # Plot
    contour_levels = np.linspace(vmin, vmax, 256)
    CS = ax.contourf(Coord_grid, Z_grid, Values_smooth,
                     levels=contour_levels, cmap=cmap,
                     vmin=vmin, vmax=vmax, extend='both')

    # Add colorbar
    cbar = None
    if add_colorbar:
        cbar = plt.colorbar(CS, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f'{field_name} ({unit})', fontsize=12)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))

    ax.set_xlabel(f'{coord_label} (km)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title(f'Vertical Slice of {field_name} at {direction.upper()} = {position_m/1e3:.1f} km (Smooth)\n({len(values_mesh)} points → {resolution}² grid)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

    print(f"  {field_name} range: {np.nanmin(Values_smooth):.2f} - {np.nanmax(Values_smooth):.2f} {unit}")

    return CS, cbar


if __name__ == "__main__":
    print("Smooth PyVista visualization functions loaded!")
    print("Functions:")
    print("  - plot_horizontal_slice_pyvista_smooth()")
    print("  - plot_vertical_slice_pyvista_smooth()")
