#!/usr/bin/env python3
"""
Generate visualizations for a single specific day
================================================

This script generates PNGs and GIFs for a single day from the LES analysis.
Useful for examining specific days like 2024-06-23 that may have interesting characteristics.

Author: Max Aragon Cerecedes
"""
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from pyproj import Transformer, Geod
from pathlib import Path
from datetime import datetime
import imageio
from tqdm import tqdm
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    "center_coords": {"lat": 48.1374, "lon": 11.5755}, # LMU Munich
    "aoi_radius_km": 20,
    "france_extent": {"lon": [-5, 10], "lat": [42, 52]},
    "data_glob_pattern": "S_NWC_CT_MSG*_globeM-VISIR_*.nc",
    "geos_proj_str": "+proj=geos +a=6378137.0 +b=6356752.3 +lon_0=0.0 +h=35785863.0 +sweep=y",
    "cloud_categories": {
        "Other": [1, 2, 3, 4, 255],
        "Warm": [5, 6, 7, 8, 9, 10],  # Fog + Thick clouds
        "Cold": [11, 12, 13, 14, 15],  # Thin clouds
    },
    "plot_categories": {
        0: {"name": "Clear/Other", "color": "white"},
        1: {"name": "Warm", "color": "red"},      # Fog + Thick clouds
        2: {"name": "Cold", "color": "blue"},     # Thin clouds
    }
}

_GEOD = Geod(ellps='WGS84')

def _haversine_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Calculate great-circle distance in kilometers using pyproj."""
    _, _, dist_m = _GEOD.inv(lon0, lat0, lon1, lat1)
    return dist_m / 1000

class SingleDayVisualizer:
    """Generates visualizations for a single specific day."""

    def __init__(self, data_root: Path, out_root: Path, date_str: str, score: float = None):
        self.data_root = data_root
        self.out_root = out_root
        self.date_str = date_str
        self.score = score

        # Create output directories
        self.png_dir = self.out_root / "single_day_png"
        self.gif_dir = self.out_root / "single_day_gif"
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.gif_dir.mkdir(parents=True, exist_ok=True)

        # Setup plotting
        self._setup_plotting()

        # Internal state for grid information
        self._lat2d, self._lon2d = None, None
        self._fr_slice, self._aoi_slice = (), ()

    def _setup_plotting(self):
        """Setup matplotlib for plotting."""
        plt.style.use('fast')
        plt.rcParams.update({
            'figure.dpi': 100, 'savefig.dpi': 100, 'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1, 'axes.edgecolor': 'black'
        })
        cat_map = CONFIG["plot_categories"]
        colors = [cat_map[i]["color"] for i in sorted(cat_map.keys())]
        self.cmap = ListedColormap(colors)

    def _setup_grid(self, ds: xr.Dataset):
        """Initialize geographic grid and calculate data slices."""
        if self._lat2d is not None:
            return

        logging.info("Initializing geographic grid...")
        proj_str = ds.attrs.get('gdal_projection', CONFIG["geos_proj_str"])
        transformer = Transformer.from_crs(proj_str, "epsg:4326", always_xy=True)
        
        nx_grid, ny_grid = np.meshgrid(ds.nx.values, ds.ny.values)
        self._lon2d, self._lat2d = transformer.transform(nx_grid, ny_grid)

        # Find SIRTA location
        center_lat, center_lon = CONFIG["center_coords"]["lat"], CONFIG["center_coords"]["lon"]
        dist_sq = (self._lat2d - center_lat)**2 + (self._lon2d - center_lon)**2
        y0, x0 = np.unravel_index(np.nanargmin(dist_sq), dist_sq.shape)

        # Calculate resolution and AOI slice
        res_y = _haversine_km(self._lat2d[y0 - 1, x0], self._lon2d[y0 - 1, x0], 
                             self._lat2d[y0 + 1, x0], self._lon2d[y0 + 1, x0]) / 2
        res_x = _haversine_km(self._lat2d[y0, x0 - 1], self._lon2d[y0, x0 - 1], 
                             self._lat2d[y0, x0 + 1], self._lon2d[y0, x0 + 1]) / 2
        avg_pixel_res_km = (res_x + res_y) / 2

        n_pixels = int(np.ceil(CONFIG["aoi_radius_km"] / avg_pixel_res_km))
        y_start, y_end = max(0, y0 - n_pixels), min(self._lat2d.shape[0], y0 + n_pixels + 1)
        x_start, x_end = max(0, x0 - n_pixels), min(self._lat2d.shape[1], x0 + n_pixels + 1)
        self._aoi_slice = (slice(y_start, y_end), slice(x_start, x_end))

        # France extent slice
        fr_lon_min, fr_lon_max = CONFIG["france_extent"]["lon"]
        fr_lat_min, fr_lat_max = CONFIG["france_extent"]["lat"]
        
        fr_mask = ((self._lon2d >= fr_lon_min) & (self._lon2d <= fr_lon_max) & 
                   (self._lat2d >= fr_lat_min) & (self._lat2d <= fr_lat_max))
        
        if np.any(fr_mask):
            fr_indices = np.where(fr_mask)
            y_start_fr, y_end_fr = fr_indices[0].min(), fr_indices[0].max() + 1
            x_start_fr, x_end_fr = fr_indices[1].min(), fr_indices[1].max() + 1
            self._fr_slice = (slice(y_start_fr, y_end_fr), slice(x_start_fr, x_end_fr))
        else:
            self._fr_slice = self._aoi_slice

        logging.info(f"AOI slice: {self._aoi_slice}")
        logging.info(f"France slice: {self._fr_slice}")

    def _map_cloud_types(self, ct_raw: np.ndarray) -> np.ndarray:
        """Map raw cloud type values to simplified categories."""
        ct_mapped = np.zeros_like(ct_raw, dtype=int)
        
        # Map warm clouds (5-10) to category 1
        for warm_type in CONFIG["cloud_categories"]["Warm"]:
            ct_mapped[ct_raw == warm_type] = 1
            
        # Map cold clouds (11-15) to category 2
        for cold_type in CONFIG["cloud_categories"]["Cold"]:
            ct_mapped[ct_raw == cold_type] = 2
            
        return ct_mapped

    def _calculate_percentages(self, arr: np.ndarray) -> dict:
        """Calculate percentages of each cloud category."""
        flat = arr[~np.isnan(arr)].astype(int)
        total_pixels = flat.size if flat.size > 0 else 1
        unique, counts = np.unique(flat, return_counts=True)
        return {k: v * 100 / total_pixels for k, v in zip(unique, counts)}

    def _plot_frame(self, ts: datetime, ct_fr: np.ndarray, ct_aoi: np.ndarray, pct_aoi: dict,
                   day_rank: int, day_score: float) -> str:
        """Create a single plot frame and return the file path."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # France overview
        fr_ext = CONFIG["france_extent"]
        ax1.set_extent(fr_ext["lon"] + fr_ext["lat"], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='gray')
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

        im = ax1.pcolormesh(self._lon2d[self._fr_slice], self._lat2d[self._fr_slice], ct_fr,
                           cmap=self.cmap, vmin=-0.5, vmax=2.5)

        # Add AOI rectangle
        aoi_lon, aoi_lat = self._lon2d[self._aoi_slice], self._lat2d[self._aoi_slice]
        rect = Rectangle((np.nanmin(aoi_lon), np.nanmin(aoi_lat)), np.nanmax(aoi_lon) - np.nanmin(aoi_lon),
                        np.nanmax(aoi_lat) - np.nanmin(aoi_lat),
                        linewidth=2, edgecolor='black', facecolor='none',
                        label='AOI', transform=ccrs.PlateCarree())
        ax1.add_patch(rect)
        ax1.set_title(f'France - {ts.strftime("%Y-%m-%d %H:%M UTC")} - Rank #{day_rank} (Score: {day_score:.2f})')
        ax1.legend()

        # AOI detail
        padding_deg = 0.05
        ax2.set_extent([np.nanmin(aoi_lon) - padding_deg, np.nanmax(aoi_lon) + padding_deg,
                       np.nanmin(aoi_lat) - padding_deg, np.nanmax(aoi_lat) + padding_deg])
        ax2.pcolormesh(aoi_lon, aoi_lat, ct_aoi, cmap=self.cmap, vmin=-0.5, vmax=2.5)

        gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                          color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter(zero_direction_label=True, number_format='.2f')
        gl.yformatter = LatitudeFormatter(number_format='.2f')

        # **FIX**: Percentages are now based on mapped categories (0, 1, 2)
        warm_pct = pct_aoi.get(1, 0.0)
        cold_pct = pct_aoi.get(2, 0.0)
        ax2.set_title(f'AOI Detail\nWarm: {warm_pct:.1f}%, Cold: {cold_pct:.1f}%')

        # Colorbar
        cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.7, pad=0.03)
        cbar.set_label('Cloud Type')
        plot_cats = CONFIG["plot_categories"]
        # **FIX**: Sort keys to ensure labels and ticks are correctly ordered
        ticks = sorted(plot_cats.keys())
        labels = [plot_cats[k]["name"] for k in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)

        # Save frame
        frame_path = self.png_dir / f"frame_{ts.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return str(frame_path)

    def process_day(self):
        """Process the single day and create GIF."""
        logging.info(f"Processing day {self.date_str}")

        date_obj = datetime.strptime(self.date_str, "%Y-%m-%d")
        day_dir = self.data_root / f"{date_obj.strftime('%Y_%m_%d')}"

        if not day_dir.exists():
            logging.warning(f"Directory not found: {day_dir}")
            return

        ct_files = sorted(day_dir.glob(CONFIG["data_glob_pattern"]))
        if not ct_files:
            logging.warning(f"No CT files found in {day_dir}")
            return

        with xr.open_dataset(ct_files[0]) as ds:
            self._setup_grid(ds)

        frame_paths = []

        for file_path in tqdm(ct_files, desc=f"Day {self.date_str}"):
            try:
                ts = datetime.strptime(file_path.stem.split('_')[-1], "%Y%m%dT%H%M%SZ")

                if 6 <= ts.hour < 18:
                    with xr.open_dataset(file_path) as ds:
                        # Get raw data
                        ct_aoi_raw = ds.ct.isel(ny=self._aoi_slice[0], nx=self._aoi_slice[1]).values
                        ct_fr_raw = ds.ct.isel(ny=self._fr_slice[0], nx=self._fr_slice[1]).values

                        # Map raw data to simplified categories (0, 1, 2)
                        ct_aoi_mapped = self._map_cloud_types(ct_aoi_raw)
                        ct_fr_mapped = self._map_cloud_types(ct_fr_raw)
                        
                        # **FIX**: Calculate percentages from the MAPPED data for consistency
                        pct_aoi_mapped = self._calculate_percentages(ct_aoi_mapped)
                        
                        # Pass mapped data and its corresponding percentages to the plot function
                        frame_path = self._plot_frame(ts, ct_fr_mapped, ct_aoi_mapped, pct_aoi_mapped, 1, 0.0)
                        frame_paths.append(frame_path)

            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}", exc_info=False)

        if frame_paths:
            gif_path = self.gif_dir / f"{self.date_str}_daytime.gif"
            logging.info(f"Creating GIF: {gif_path}")

            images = [imageio.imread(frame_path) for frame_path in frame_paths]
            imageio.mimsave(gif_path, images, duration=200, loop=0)  # 200ms duration (faster), loop forever

            logging.info(f"GIF created: {gif_path}")

            for frame_path in frame_paths:
                Path(frame_path).unlink()
        else:
            logging.warning(f"No frames generated for {self.date_str}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate visualizations for a single day')
    parser.add_argument('date', help='Date in YYYY-MM-DD format (e.g., 2024-06-23)')
    parser.add_argument('--data-root', default='/mnt/m0/y-m.saint-drenan/data/NWCSAF_CloudType/2024/',
                       help='Root directory for cloud type data')
    parser.add_argument('--output-dir', default='./output_single_day',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    visualizer = SingleDayVisualizer(data_root, out_root, args.date)
    visualizer.process_day()

if __name__ == "__main__":
    main() 