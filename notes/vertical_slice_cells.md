# Vertical Slice Cells for plt_slip_mu_relation.ipynb

Copy these cells into your notebook. They use your existing `df_plate` data structure.

---

## Cell 1: Markdown

```markdown
## 13. Vertical Slices

Visualize μ structure on vertical cross-sections along strike and dip directions,
with plate interface overlay for geological context.
```

---

## Cell 2: Code - Prepare plate interface in rotated coordinates

```python
# Prepare plate interface in rotated mesh coordinates
# Note: df_plate already exists with columns ['x', 'y', 'z', 'lon', 'lat']
# where x, y are in meters (original unrotated coordinates)

df_plate_interface = df_plate.copy()

# Apply rotation to convert to mesh coordinate system
df_plate_interface['x_local'] = df_plate_interface['x'].values * np.cos(np.radians(rot)) + \
                                  df_plate_interface['y'].values * np.sin(np.radians(rot))
df_plate_interface['y_local'] = -df_plate_interface['x'].values * np.sin(np.radians(rot)) + \
                                  df_plate_interface['y'].values * np.cos(np.radians(rot))
df_plate_interface['z_local'] = df_plate_interface['z'].values * 1e3  # Convert km to meters

print(f"Plate interface: {len(df_plate_interface)} points")
print(f"X range: [{df_plate_interface['x_local'].min()/1e3:.1f}, {df_plate_interface['x_local'].max()/1e3:.1f}] km")
print(f"Y range: [{df_plate_interface['y_local'].min()/1e3:.1f}, {df_plate_interface['y_local'].max()/1e3:.1f}] km")
print(f"Z range: [{df_plate_interface['z_local'].min()/1e3:.1f}, {df_plate_interface['z_local'].max()/1e3:.1f}] km")
```

---

## Cell 3: Code - Helper function for vertical slicing

```python
def extract_vertical_slice_pygmt(mu_grid, normal, origin, slice_axis='y-z',
                                   field_name='shear modulus'):
    """
    Extract vertical slice from PyVista grid and prepare for PyGMT plotting.

    Parameters:
    -----------
    mu_grid : pyvista.UnstructuredGrid
        The 3D grid with mu values
    normal : list
        Normal vector for the slice plane [nx, ny, nz]
    origin : list
        Origin point for the slice plane [x, y, z]
    slice_axis : str
        Which axes to plot: 'y-z' for along-strike, 'x-z' for along-dip
    field_name : str
        Name of the field to extract from the grid

    Returns:
    --------
    slice_df : pd.DataFrame
        DataFrame with columns [coord1, z, mu] ready for PyGMT
    """
    # Slice the grid
    slice_obj = mu_grid.slice(normal=normal, origin=origin)

    if len(slice_obj.points) == 0:
        print(f"Warning: Empty slice at origin {origin}")
        return None

    # Extract coordinates and mu values
    points = slice_obj.points  # [N, 3] array
    mu_vals = slice_obj[field_name]

    # Select appropriate coordinates based on slice direction
    if slice_axis == 'y-z':
        # Along-strike slice (constant x)
        coord1 = points[:, 1] / 1e3  # y in km
        coord1_name = 'y_km'
    elif slice_axis == 'x-z':
        # Along-dip slice (constant y)
        coord1 = points[:, 0] / 1e3  # x in km
        coord1_name = 'x_km'
    else:
        raise ValueError(f"Unknown slice_axis: {slice_axis}")

    z_km = points[:, 2] / 1e3  # z in km

    # Create DataFrame
    slice_df = pd.DataFrame({
        coord1_name: coord1,
        'z_km': z_km,
        'mu': mu_vals
    })

    return slice_df
```

---

## Cell 4: Code - Verify PyVista grids

```python
# Ensure mu_3d_grid and mu_hom_grid are PyVista objects
# If they were overwritten, reload them

if not hasattr(mu_3d_grid, 'slice'):
    print("Reloading mu_3d_grid as PyVista object...")
    mu_3d_grid = load_xdmf_as_pyvista(mu_3d_file)

if not hasattr(mu_hom_grid, 'slice'):
    print("Reloading mu_hom_grid as PyVista object...")
    mu_hom_grid = load_xdmf_as_pyvista(mu_hom_file)

print(f"mu_3d_grid type: {type(mu_3d_grid)}")
print(f"mu_hom_grid type: {type(mu_hom_grid)}")
print("Ready for slicing!")
```

---

## Cell 5: Code - Along-strike vertical slices (constant x)

```python
# Define x positions for along-strike slices (in km, then convert to m)
x_strike_km = [-60, -30, 0, 30, 60]  # 5 slices
x_strike_values = [x * 1e3 for x in x_strike_km]  # Convert to meters

# Create figure with PyGMT
n_slices = len(x_strike_values)
fig = pygmt.Figure()

# Define region for all panels (y-z space)
y_min, y_max = -120, 120  # km
z_min, z_max = -80, 5     # km
region = [y_min, y_max, z_min, z_max]

# Colormap limits
mu_min = 0
mu_max = 120  # GPa

for i, (x_km, x_val) in enumerate(zip(x_strike_km, x_strike_values)):
    # Extract slices
    normal = [1., 0., 0.]  # Normal in x direction
    origin = [x_val, 0., 0.]

    # Get 3D heterogeneous slice
    slice_3d = extract_vertical_slice_pygmt(mu_3d_grid, normal, origin,
                                             slice_axis='y-z',
                                             field_name='shear modulus')

    if slice_3d is None:
        continue

    # Shift figure position
    if i > 0:
        fig.shift_origin(xshift='12c')

    # Plot the slice
    fig.basemap(region=region, projection="X10c/8c",
                frame=['WSne', 'xaf+l"Along-strike (km)"', 'yaf+l"Depth (km)"'])

    # Create grid for plotting
    grid_3d = pygmt.xyz2grd(x=slice_3d['y_km'], y=slice_3d['z_km'], z=slice_3d['mu'],
                             region=region, spacing=[1, 0.5])

    fig.grdimage(grid=grid_3d, cmap='viridis', nan_transparent=True)

    # Extract and plot plate interface on this slice
    tolerance = 5e3  # 5 km tolerance in meters
    interface_mask = np.abs(df_plate_interface['x_local'] - x_val) < tolerance

    if interface_mask.sum() > 0:
        interface_y_km = df_plate_interface.loc[interface_mask, 'y_local'].values / 1e3
        interface_z_km = df_plate_interface.loc[interface_mask, 'z_local'].values / 1e3

        # Sort by y for proper line plotting
        sort_idx = np.argsort(interface_y_km)
        interface_y_km = interface_y_km[sort_idx]
        interface_z_km = interface_z_km[sort_idx]

        fig.plot(x=interface_y_km, y=interface_z_km, pen="2p,white,-")

    # Add title
    fig.text(text=f"x = {x_km} km", position="TC", offset="0c/0.3c",
             font="12p,Helvetica-Bold")

    # Add colorbar on last panel
    if i == n_slices - 1:
        fig.colorbar(position="JMR+o1c/0c+w8c/0.4c", frame=[f'af+l"@~m@~ (GPa)"'])

# Save figure
output_file = resultpath + 'mu_vertical_slices_strike.png'
fig.savefig(output_file, dpi=300)
print(f"\nAlong-strike slices saved to: {output_file}")
fig.show()
```

---

## Cell 6: Code - Along-dip vertical slices (constant y)

```python
# Define y positions for along-dip slices (in km, then convert to m)
y_dip_km = [-90, -45, 0, 45, 90]  # 5 slices
y_dip_values = [y * 1e3 for y in y_dip_km]  # Convert to meters

# Create figure with PyGMT
fig = pygmt.Figure()

# Define region for all panels (x-z space)
x_min, x_max = -60, 120  # km
z_min, z_max = -80, 5    # km
region = [x_min, x_max, z_min, z_max]

for i, (y_km, y_val) in enumerate(zip(y_dip_km, y_dip_values)):
    # Extract slices
    normal = [0., 1., 0.]  # Normal in y direction
    origin = [0., y_val, 0.]

    # Get 3D heterogeneous slice
    slice_3d = extract_vertical_slice_pygmt(mu_3d_grid, normal, origin,
                                             slice_axis='x-z',
                                             field_name='shear modulus')

    if slice_3d is None:
        continue

    # Shift figure position
    if i > 0:
        fig.shift_origin(xshift='12c')

    # Plot the slice
    fig.basemap(region=region, projection="X10c/8c",
                frame=['WSne', 'xaf+l"Along-dip (km)"', 'yaf+l"Depth (km)"'])

    # Create grid for plotting
    grid_3d = pygmt.xyz2grd(x=slice_3d['x_km'], y=slice_3d['z_km'], z=slice_3d['mu'],
                             region=region, spacing=[1, 0.5])

    fig.grdimage(grid=grid_3d, cmap='viridis', nan_transparent=True)

    # Extract and plot plate interface on this slice
    tolerance = 5e3  # 5 km tolerance in meters
    interface_mask = np.abs(df_plate_interface['y_local'] - y_val) < tolerance

    if interface_mask.sum() > 0:
        interface_x_km = df_plate_interface.loc[interface_mask, 'x_local'].values / 1e3
        interface_z_km = df_plate_interface.loc[interface_mask, 'z_local'].values / 1e3

        # Sort by x for proper line plotting
        sort_idx = np.argsort(interface_x_km)
        interface_x_km = interface_x_km[sort_idx]
        interface_z_km = interface_z_km[sort_idx]

        fig.plot(x=interface_x_km, y=interface_z_km, pen="2p,white,-")

    # Add title
    fig.text(text=f"y = {y_km} km", position="TC", offset="0c/0.3c",
             font="12p,Helvetica-Bold")

    # Add colorbar on last panel
    if i == n_slices - 1:
        fig.colorbar(position="JMR+o1c/0c+w8c/0.4c", frame=[f'af+l"@~m@~ (GPa)"'])

# Save figure
output_file = resultpath + 'mu_vertical_slices_dip.png'
fig.savefig(output_file, dpi=300)
print(f"\nAlong-dip slices saved to: {output_file}")
fig.show()
```

---

## Cell 7 (Optional): Code - Single slice comparison (μ_3D vs μ anomaly)

```python
# Create a 2-panel figure showing both mu_3d and mu_anomaly on a single vertical slice
fig = pygmt.Figure()

# Along-strike slice at x=0 km showing both mu_3d and mu_anomaly
x_val = 0.0
normal = [1., 0., 0.]
origin = [x_val, 0., 0.]

# Extract slices
slice_3d = extract_vertical_slice_pygmt(mu_3d_grid, normal, origin,
                                         slice_axis='y-z',
                                         field_name='shear modulus')
slice_hom = extract_vertical_slice_pygmt(mu_hom_grid, normal, origin,
                                          slice_axis='y-z',
                                          field_name='shear modulus')

# Merge and compute anomaly
from scipy.interpolate import griddata
slice_merged = slice_3d.copy()
slice_merged['mu_hom'] = griddata((slice_hom['y_km'].values, slice_hom['z_km'].values),
                                   slice_hom['mu'].values,
                                   (slice_3d['y_km'].values, slice_3d['z_km'].values),
                                   method='nearest')
slice_merged['mu_anomaly'] = (slice_merged['mu'] - slice_merged['mu_hom']) / slice_merged['mu_hom'] * 100

# Create 2-panel figure
region = [-120, 120, -80, 5]

# Panel 1: μ_3D
fig.basemap(region=region, projection="X10c/8c",
            frame=['WSne', 'xaf+l"Along-strike (km)"', 'yaf+l"Depth (km)"'])
grid_3d = pygmt.xyz2grd(x=slice_merged['y_km'], y=slice_merged['z_km'], z=slice_merged['mu'],
                        region=region, spacing=[1, 0.5])
fig.grdimage(grid=grid_3d, cmap='viridis')

# Overlay plate interface
tolerance = 5e3
interface_mask = np.abs(df_plate_interface['x_local'] - x_val) < tolerance
if interface_mask.sum() > 0:
    interface_y_km = df_plate_interface.loc[interface_mask, 'y_local'].values / 1e3
    interface_z_km = df_plate_interface.loc[interface_mask, 'z_local'].values / 1e3
    sort_idx = np.argsort(interface_y_km)
    fig.plot(x=interface_y_km[sort_idx], y=interface_z_km[sort_idx], pen="2p,white,-")

fig.colorbar(position="JML+o-1c/0c+w8c/0.4c", frame=[f'af+l"@~m@~@-3D@- (GPa)"'])
fig.text(text="(a) 3D μ Structure", position="TL", offset="0.2c/-0.2c",
         font="12p,Helvetica-Bold", fill="white")

# Panel 2: μ anomaly
fig.shift_origin(xshift='12c')
fig.basemap(region=region, projection="X10c/8c",
            frame=['wSne', 'xaf+l"Along-strike (km)"', 'yaf'])
vmax = np.nanpercentile(np.abs(slice_merged['mu_anomaly']), 95)
grid_anom = pygmt.xyz2grd(x=slice_merged['y_km'], y=slice_merged['z_km'],
                          z=slice_merged['mu_anomaly'],
                          region=region, spacing=[1, 0.5])
pygmt.makecpt(cmap='polar', series=[-vmax, vmax, vmax/10])
fig.grdimage(grid=grid_anom, cmap=True)

# Overlay plate interface
if interface_mask.sum() > 0:
    fig.plot(x=interface_y_km[sort_idx], y=interface_z_km[sort_idx], pen="2p,black,-")

fig.colorbar(position="JMR+o1c/0c+w8c/0.4c", frame=[f'af+l"@~m@~ Anomaly (%)"'])
fig.text(text="(b) μ Anomaly", position="TL", offset="0.2c/-0.2c",
         font="12p,Helvetica-Bold", fill="white")

output_file = resultpath + 'mu_vertical_slice_comparison.png'
fig.savefig(output_file, dpi=300)
print(f"\nComparison slice saved to: {output_file}")
fig.show()
```

---

## Notes

1. These cells use your existing `df_plate` dataframe
2. The rotation is applied to create `df_plate_interface` with `x_local`, `y_local`, `z_local` columns
3. All cells assume the following variables already exist in your notebook:
   - `rot` (rotation angle, degrees)
   - `mu_3d_grid`, `mu_hom_grid` (PyVista grids)
   - `resultpath` (output directory)
   - `load_xdmf_as_pyvista()` function

4. The slicing uses PyVista's native `.slice()` method which is robust and efficient

5. Output files:
   - `mu_vertical_slices_strike.png` - 5 along-strike slices
   - `mu_vertical_slices_dip.png` - 5 along-dip slices
   - `mu_vertical_slice_comparison.png` - 2-panel comparison at x=0
