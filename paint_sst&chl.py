import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.image import imread


def plot_correlations_tif_files(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter files to only include those with "correlations" in the filename and a ".tif" extension
    tif_files = [f for f in files if "correlations" in f and f.endswith(".tif")]

    # If no tif files are found, raise an error
    if not tif_files:
        raise ValueError("No tif files found in folder")

    # Loop through each tif file and plot it on a map
    for tif_file in tif_files:
        tif_path = './0_cartopy_data/NE1_50M_SR_W.tif'
        # Read the tif file using xarray
        data = xr.open_rasterio(os.path.join(folder_path, tif_file))

        # Get the extent of the data
        extent = [data.x[0], data.x[-1], data.y[-1], data.y[0]]

        # Create a plot using cartopy
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        ax.imshow(
            imread(tif_path),
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90]
        )

        # Plot the data
        ax.imshow(data[0], origin="upper", extent=extent, transform=ccrs.PlateCarree())

        # Set the title of the plot to the filename
        ax.set_title(tif_file)

        # Show the plot
        plt.show()


if __name__ == '__main__':
    plot_correlations_tif_files('./06_bohai_result_file/')
