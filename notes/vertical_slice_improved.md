# Improved Vertical Slicing - Using Matplotlib Triangulation

The issue with the PyGMT approach is that `xyz2grd` creates a sparse grid from scattered slice points.
Instead, use matplotlib's triangulation like in `plot_vertical_slice_pyvista`.

---

## Updated Cell: Along-strike vertical slices with matplotlib

```python
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Define x positions for along-strike slices (in km, then convert to m)
x_strike_km = [-60, -30, 0, 30, 60]  # 5 slices
x_strike_values = [x * 1e3 for x in x_strike_km]  # Convert to meters

# Create matplotlib figure with subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300, sharey=True)

# Define region for all panels (y-z space)
y_min, y_max = -120, 120  # km
z_min, z_max = -80, 5     # km

# Colormap limits
mu_min = 0
mu_max = 120  # GPa

for i, (x_km, x_val) in enumerate(zip(x_strike_km, x_strike_values)):
    ax = axes[i]

    # Extract slice
    normal = [1., 0., 0.]  # Normal in x direction
    origin = [x_val, 0., 0.]

    slice_mesh = mu_3d_grid.slice(normal=normal, origin=origin)

    if len(slice_mesh.points) == 0:
        print(f"Warning: Empty slice at x = {x_km} km")
        continue

    # Extract coordinates and mu values
    y_slice = slice_mesh.points[:, 1] / 1e3  # y in km
    z_slice = slice_mesh.points[:, 2] / 1e3  # z in km
    mu_slice = slice_mesh['shear modulus']

    # Create triangulation for scattered data
    triang = mtri.Triangulation(y_slice, z_slice)

    # Plot filled contour
    cs = ax.tricontourf(triang, mu_slice, levels=256, cmap='viridis',
                        vmin=mu_min, vmax=mu_max, extend='both')

    # Overlay plate interface
    tolerance = 5e3  # 5 km tolerance in meters
    interface_mask = np.abs(df_plate_interface['x_local'] - x_val) < tolerance

    if interface_mask.sum() > 0:
        interface_y_km = df_plate_interface.loc[interface_mask, 'y_local'].values / 1e3
        interface_z_km = df_plate_interface.loc[interface_mask, 'z_local'].values / 1e3

        # Sort by y for proper line plotting
        sort_idx = np.argsort(interface_y_km)
        interface_y_km = interface_y_km[sort_idx]
        interface_z_km = interface_z_km[sort_idx]

        ax.plot(interface_y_km, interface_z_km, 'w--', linewidth=2, label='Plate interface')

    # Set limits and labels
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('Along-strike (km)', fontsize=10)
    ax.set_title(f'x = {x_km} km', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    if i == 0:
        ax.set_ylabel('Depth (km)', fontsize=10)

# Add colorbar
cbar = fig.colorbar(cs, ax=axes, orientation='horizontal',
                    pad=0.08, aspect=40, shrink=0.8)
cbar.set_label('Shear Modulus μ (GPa)', fontsize=11)

plt.tight_layout()
output_file = resultpath + 'mu_vertical_slices_strike_matplotlib.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nAlong-strike slices saved to: {output_file}")
plt.show()
```

---

## Updated Cell: Along-dip vertical slices with matplotlib

```python
# Define y positions for along-dip slices (in km, then convert to m)
y_dip_km = [-90, -45, 0, 45, 90]  # 5 slices
y_dip_values = [y * 1e3 for y in y_dip_km]  # Convert to meters

# Create matplotlib figure with subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300, sharey=True)

# Define region for all panels (x-z space)
x_min, x_max = -60, 120  # km
z_min, z_max = -80, 5    # km

for i, (y_km, y_val) in enumerate(zip(y_dip_km, y_dip_values)):
    ax = axes[i]

    # Extract slice
    normal = [0., 1., 0.]  # Normal in y direction
    origin = [0., y_val, 0.]

    slice_mesh = mu_3d_grid.slice(normal=normal, origin=origin)

    if len(slice_mesh.points) == 0:
        print(f"Warning: Empty slice at y = {y_km} km")
        continue

    # Extract coordinates and mu values
    x_slice = slice_mesh.points[:, 0] / 1e3  # x in km
    z_slice = slice_mesh.points[:, 2] / 1e3  # z in km
    mu_slice = slice_mesh['shear modulus']

    # Create triangulation for scattered data
    triang = mtri.Triangulation(x_slice, z_slice)

    # Plot filled contour
    cs = ax.tricontourf(triang, mu_slice, levels=256, cmap='viridis',
                        vmin=mu_min, vmax=mu_max, extend='both')

    # Overlay plate interface
    tolerance = 5e3  # 5 km tolerance in meters
    interface_mask = np.abs(df_plate_interface['y_local'] - y_val) < tolerance

    if interface_mask.sum() > 0:
        interface_x_km = df_plate_interface.loc[interface_mask, 'x_local'].values / 1e3
        interface_z_km = df_plate_interface.loc[interface_mask, 'z_local'].values / 1e3

        # Sort by x for proper line plotting
        sort_idx = np.argsort(interface_x_km)
        interface_x_km = interface_x_km[sort_idx]
        interface_z_km = interface_z_km[sort_idx]

        ax.plot(interface_x_km, interface_z_km, 'w--', linewidth=2, label='Plate interface')

    # Set limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('Along-dip (km)', fontsize=10)
    ax.set_title(f'y = {y_km} km', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    if i == 0:
        ax.set_ylabel('Depth (km)', fontsize=10)

# Add colorbar
cbar = fig.colorbar(cs, ax=axes, orientation='horizontal',
                    pad=0.08, aspect=40, shrink=0.8)
cbar.set_label('Shear Modulus μ (GPa)', fontsize=11)

plt.tight_layout()
output_file = resultpath + 'mu_vertical_slices_dip_matplotlib.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nAlong-dip slices saved to: {output_file}")
plt.show()
```

---

## Alternative: Dense interpolation for PyGMT (if you prefer PyGMT style)

```python
# If you prefer PyGMT style, use denser interpolation
# This creates a regular grid from scattered slice points

from scipy.interpolate import griddata

x_strike_km = [-60, -30, 0, 30, 60]
x_strike_values = [x * 1e3 for x in x_strike_km]

fig = pygmt.Figure()

y_min, y_max = -120, 120
z_min, z_max = -80, 5
region = [y_min, y_max, z_min, z_max]

# Create dense regular grid for interpolation
y_grid = np.linspace(y_min, y_max, 400)  # Increased from 200
z_grid = np.linspace(z_min, z_max, 300)  # Increased from 150
yy, zz = np.meshgrid(y_grid, z_grid)

for i, (x_km, x_val) in enumerate(zip(x_strike_km, x_strike_values)):
    normal = [1., 0., 0.]
    origin = [x_val, 0., 0.]

    slice_mesh = mu_3d_grid.slice(normal=normal, origin=origin)

    if len(slice_mesh.points) == 0:
        continue

    # Extract slice data
    y_slice = slice_mesh.points[:, 1] / 1e3
    z_slice = slice_mesh.points[:, 2] / 1e3
    mu_slice = slice_mesh['shear modulus']

    # Interpolate to DENSE regular grid
    mu_grid = griddata((y_slice, z_slice), mu_slice, (yy, zz),
                       method='linear')  # Use linear for smoother result

    if i > 0:
        fig.shift_origin(xshift='12c')

    fig.basemap(region=region, projection="X10c/8c",
                frame=['WSne', 'xaf+l"Along-strike (km)"', 'yaf+l"Depth (km)"'])

    # Create DataFrame for xyz2grd
    y_flat = yy.ravel()
    z_flat = zz.ravel()
    mu_flat = mu_grid.ravel()

    # Remove NaN values
    valid = ~np.isnan(mu_flat)

    # Create grid with MUCH smaller spacing
    grid_3d = pygmt.xyz2grd(x=y_flat[valid], y=z_flat[valid], z=mu_flat[valid],
                            region=region, spacing=[0.5, 0.3])  # Smaller spacing

    fig.grdimage(grid=grid_3d, cmap='viridis', nan_transparent=True)

    # Overlay plate interface
    tolerance = 5e3
    interface_mask = np.abs(df_plate_interface['x_local'] - x_val) < tolerance

    if interface_mask.sum() > 0:
        interface_y_km = df_plate_interface.loc[interface_mask, 'y_local'].values / 1e3
        interface_z_km = df_plate_interface.loc[interface_mask, 'z_local'].values / 1e3
        sort_idx = np.argsort(interface_y_km)
        fig.plot(x=interface_y_km[sort_idx], y=interface_z_km[sort_idx], pen="2p,white,-")

    fig.text(text=f"x = {x_km} km", position="TC", offset="0c/0.3c",
             font="12p,Helvetica-Bold")

    if i == len(x_strike_km) - 1:
        fig.colorbar(position="JMR+o1c/0c+w8c/0.4c", frame=[f'af+l"@~m@~ (GPa)"'])

output_file = resultpath + 'mu_vertical_slices_strike_pygmt_dense.png'
fig.savefig(output_file, dpi=300)
print(f"\nDense PyGMT slices saved to: {output_file}")
fig.show()
```

---

## Key Insights from `plot_vertical_slice_pyvista`:

1. **Triangulation is better for scattered data**: Your function uses `mtri.Triangulation` which creates a mesh connecting nearby points, avoiding gaps

2. **Direct plotting without gridding**: `tricontourf` plots directly on the triangulated mesh without needing to interpolate to a regular grid first

3. **More efficient**: No intermediate gridding step means faster plotting and better preservation of original data

4. **Better for PyVista slices**: Since PyVista slices return scattered points (not regular grids), triangulation handles them naturally

The matplotlib approach will give you smooth, gap-free plots similar to what you have in `plt_test3dmodel.ipynb`.
