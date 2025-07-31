#!/usr/bin/env python3
"""
Generate visualizations for top-ranked LES days
===============================================

This script generates PNGs and GIFs for the top-ranked days from the LES analysis.
It's much faster than processing all 30 days and focuses on the most relevant results.

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    "center_coords": {"lat": 48.71725, "lon": 2.20884}, # SIRTA
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

class TopDayVisualizer:
    """Generates visualizations for top-ranked LES days."""

    def __init__(self, data_root: Path, out_root: Path, ranking_csv: Path, top_n: int = 5):
        self.data_root = data_root
        self.out_root = out_root
        self.ranking_csv = ranking_csv
        self.top_n = top_n

        # Create output directories
        self.png_dir = self.out_root / "top_days_png"
        self.gif_dir = self.out_root / "top_days_gif"
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.gif_dir.mkdir(parents=True, exist_ok=True)

        # Setup plotting
        self._setup_plotting()

        # Load ranking data
        self.ranking_df = pd.read_csv(ranking_csv)
        self.top_days = self.ranking_df.head(top_n)

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

        # Clean non-finite values
        self._lon2d[~np.isfinite(self._lon2d)] = np.nan
        self._lat2d[~np.isfinite(self._lat2d)] = np.nan

        center_lat, center_lon = CONFIG["center_coords"]["lat"], CONFIG["center_coords"]["lon"]
        dist_sq = (self._lat2d - center_lat)**2 + (self._lon2d - center_lon)**2
        y0, x0 = np.unravel_index(np.nanargmin(dist_sq), dist_sq.shape)

        res_y = _haversine_km(self._lat2d[y0 - 1, x0], self._lon2d[y0 - 1, x0], self._lat2d[y0 + 1, x0], self._lon2d[y0 + 1, x0]) / 2
        res_x = _haversine_km(self._lat2d[y0, x0 - 1], self._lon2d[y0, x0 - 1], self._lat2d[y0, x0 + 1], self._lon2d[y0, x0 + 1]) / 2
        avg_pixel_res_km = (res_x + res_y) / 2

        n_pixels = int(np.ceil(CONFIG["aoi_radius_km"] / avg_pixel_res_km))
        y_start, y_end = max(0, y0 - n_pixels), min(self._lat2d.shape[0], y0 + n_pixels + 1)
        x_start, x_end = max(0, x0 - n_pixels), min(self._lat2d.shape[1], x0 + n_pixels + 1)
        self._aoi_slice = (slice(y_start, y_end), slice(x_start, x_end))

        fr_ext = CONFIG["france_extent"]
        fr_mask = ((self._lat2d >= fr_ext["lat"][0]) & (self._lat2d <= fr_ext["lat"][1]) &
                   (self._lon2d >= fr_ext["lon"][0]) & (self._lon2d <= fr_ext["lon"][1]))
        ys, xs = np.where(fr_mask)
        if ys.size > 0:
            self._fr_slice = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

    def _map_cloud_types(self, ct_raw: np.ndarray) -> np.ndarray:
        """Map cloud types to simplified categories."""
        cat_map = CONFIG["cloud_categories"]
        ct_mapped = np.zeros_like(ct_raw, dtype=np.int8)

        # Map warm clouds (categories 5-10) to value 1
        warm_mask = np.isin(ct_raw, cat_map["Warm"])
        ct_mapped[warm_mask] = 1

        # Map cold clouds (categories 11-15) to value 2
        cold_mask = np.isin(ct_raw, cat_map["Cold"])
        ct_mapped[cold_mask] = 2

        return ct_mapped

    def _calculate_percentages(self, arr: np.ndarray) -> dict:
        """Calculate cloud type percentages from a data array."""
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

    def process_day(self, date_str: str, rank: int, score: float):
        """Process a single day and create GIF."""
        logging.info(f"Processing day {date_str} (Rank #{rank}, Score: {score:.2f})")

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
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

        for file_path in tqdm(ct_files, desc=f"Day {date_str}"):
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
                        frame_path = self._plot_frame(ts, ct_fr_mapped, ct_aoi_mapped, pct_aoi_mapped, rank, score)
                        frame_paths.append(frame_path)

            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}", exc_info=False)

        if frame_paths:
            gif_path = self.gif_dir / f"rank_{rank:02d}_{date_str}_score_{score:.2f}.gif"
            logging.info(f"Creating GIF: {gif_path}")

            images = [imageio.imread(frame_path) for frame_path in frame_paths]
            imageio.mimsave(gif_path, images, duration=200, loop=0)  # 200ms duration (faster), loop forever

            logging.info(f"GIF created: {gif_path}")

            for frame_path in frame_paths:
                Path(frame_path).unlink()
        else:
            logging.warning(f"No frames generated for {date_str}")

    def run(self):
        """Process all top-ranked days."""
        logging.info(f"Starting visualization generation for top {self.top_n} days")

        for idx, row in self.top_days.iterrows():
            date_str = row['date']
            rank = idx + 1
            score = row['les_score']
            self.process_day(date_str, rank, score)

        logging.info("Visualization generation complete!")

def main():
    """Main function."""
    data_root = Path("/mnt/m0/y-m.saint-drenan/data/NWCSAF_CloudType/2024/")
    
    # Find the most recent ranking CSV in output directories
    output_dirs = list(Path(".").glob("output_*"))
    if not output_dirs:
        logging.error("No output directories found")
        return
    
    # Sort by modification time (most recent first)
    output_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    out_root = output_dirs[0]
    
    # Find ranking CSV in the output directory
    ranking_csvs = list(out_root.glob("les_suitability_ranking_*.csv"))
    if not ranking_csvs:
        logging.error(f"No ranking CSV found in {out_root}")
        return
    
    ranking_csv = ranking_csvs[0]
    logging.info(f"Using ranking CSV: {ranking_csv}")
    
    visualizer = TopDayVisualizer(data_root, out_root, ranking_csv, top_n=3)
    visualizer.run()

if __name__ == "__main__":
    main()