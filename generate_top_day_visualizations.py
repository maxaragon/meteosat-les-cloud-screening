#!/usr/bin/env python3
"""
Generate visualizations for top-ranked LES days
===============================================

This script generates PNGs and GIFs for the top-ranked days from the LES analysis.
It combines a flexible, multi-site configuration engine with a clear and visually
appealing pcolormesh-based plotting routine.

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
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Site Configurations (Extensible and Centralized) ---
SITE_CONFIGS = {
    # Original sites
    "MUNICH": {"center_coords": {"lat": 48.148, "lon": 11.573}},
    "PALAISEAU": {"center_coords": {"lat": 48.717, "lon": 2.209}},
    "CABAUW": {"center_coords": {"lat": 51.968, "lon": 4.927}},
    "LINDENBERG": {"center_coords": {"lat": 52.208, "lon": 14.118}},
    # CloudNet sites
    "BUCHAREST": {"center_coords": {"lat": 44.344, "lon": 26.012}},
    "CHILBOLTON": {"center_coords": {"lat": 51.144, "lon": -1.439}},
    "CLUJ": {"center_coords": {"lat": 46.768, "lon": 23.540}},
    "GALATI": {"center_coords": {"lat": 45.435, "lon": 28.037}},
    "GRANADA": {"center_coords": {"lat": 37.164, "lon": -3.605}},
    "HYYTIALA": {"center_coords": {"lat": 61.844, "lon": 24.287}},
    "JUELICH": {"center_coords": {"lat": 50.908, "lon": 6.413}},
    "KENTTAROVA": {"center_coords": {"lat": 67.987, "lon": 24.243}},
    "LAMPEDUSA": {"center_coords": {"lat": 35.520, "lon": 12.630}},
    "LEIPZIG": {"center_coords": {"lat": 51.353, "lon": 12.435}},
    "LEIPZIG-LIM": {"center_coords": {"lat": 51.333, "lon": 12.389}},
    "LIMASSOL": {"center_coords": {"lat": 34.677, "lon": 33.038}},
    "MACE-HEAD": {"center_coords": {"lat": 53.326, "lon": -9.900}},
    "MAIDO": {"center_coords": {"lat": -21.079, "lon": 55.383}},
    "MINDELO": {"center_coords": {"lat": 16.878, "lon": -24.995}},
    "NEUMAYER": {"center_coords": {"lat": -70.660, "lon": -8.284}},
    "NORUNDA": {"center_coords": {"lat": 60.086, "lon": 17.479}},
    "NY-ALESUND": {"center_coords": {"lat": 78.923, "lon": 11.922}},
    "PAYERNE": {"center_coords": {"lat": 46.813, "lon": 6.944}},
    "POTENZA": {"center_coords": {"lat": 40.601, "lon": 15.724}},
    "RZECIN": {"center_coords": {"lat": 52.758, "lon": 16.310}},
    "SCHNEEFERNERHAUS": {"center_coords": {"lat": 47.417, "lon": 10.977}},
    "WARSAW": {"center_coords": {"lat": 52.210, "lon": 20.980}},
}

# --- General Configuration ---
CONFIG = {
    "aoi_radius_km": 20,
    "data_glob_pattern": "S_NWC_CT_MSG*_globeM-VISIR_*.nc",
    "geos_proj_str": "+proj=geos +a=6378137.0 +b=6356752.3 +lon_0=0.0 +h=35785863.0 +sweep=y",
    "cloud_categories": {
        "Other": [0, 1, 2, 3, 4, 255], # Includes clear sky (0)
        "Warm": [5, 6, 7, 8, 9, 10],   # Fog + Thick low/mid-level clouds
        "Cold": [11, 12, 13, 14, 15],  # Thin/Thick high-level clouds
    },
    "plot_categories": {
        0: {"name": "Clear/Other", "color": "white"},
        1: {"name": "Warm", "color": "red"},
        2: {"name": "Cold", "color": "blue"},
    }
}

_GEOD = Geod(ellps='WGS84')

def _haversine_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Calculate great-circle distance in kilometers using pyproj."""
    _, _, dist_m = _GEOD.inv(lon0, lat0, lon1, lat1)
    return dist_m / 1000

def _determine_region_info(lat: float, lon: float) -> dict:
    """Determine map extent using a default 10x10 degree box around the site."""
    padding = 5.0  # degrees padding
    return {"lon": [lon - padding, lon + padding], "lat": [lat - padding, lat + padding], "name": "Region"}


class TopDayVisualizer:
    """Generates visualizations for top-ranked LES days."""

    def __init__(self, data_root: Path, out_root: Path, ranking_csv: Path, site: str, top_n: int = 5):
        self.data_root = data_root
        self.out_root = out_root
        self.ranking_csv = ranking_csv
        self.site_name = site
        self.top_n = top_n

        if self.site_name not in SITE_CONFIGS:
            raise ValueError(f"Unknown site '{self.site_name}'. Please add it to SITE_CONFIGS.")
        self.site_config = SITE_CONFIGS[self.site_name]
        logging.info(f"Using configuration for site: {self.site_name}")

        self.png_dir = self.out_root / f"top_days_png_{self.site_name}"
        self.gif_dir = self.out_root / f"top_days_gif_{self.site_name}"
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.gif_dir.mkdir(parents=True, exist_ok=True)

        self._setup_plotting()
        self.ranking_df = pd.read_csv(ranking_csv)
        
        # Filter for only Likely and Probable days
        suitable_days = self.ranking_df[self.ranking_df['LES_suitable'].isin(['Likely', 'Probable'])]
        self.top_days = suitable_days.head(top_n)
        
        logging.info(f"Found {len(suitable_days)} suitable days (Likely/Probable) out of {len(self.ranking_df)} total days")
        logging.info(f"Will visualize top {len(self.top_days)} suitable days")

        self._lat2d, self._lon2d = None, None
        self._region_slice, self._aoi_slice = (), ()
        self.region_info = {}

    def _setup_plotting(self):
        """Setup matplotlib for plotting."""
        plt.style.use('fast')
        plt.rcParams.update({
            'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
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

        # Convert to regular numpy arrays and handle any masked values
        if hasattr(self._lon2d, 'mask'):
            self._lon2d = np.where(self._lon2d.mask, np.nan, self._lon2d.data)
        if hasattr(self._lat2d, 'mask'):
            self._lat2d = np.where(self._lat2d.mask, np.nan, self._lat2d.data)
        
        # Ensure we have regular numpy arrays
        self._lon2d = np.asarray(self._lon2d)
        self._lat2d = np.asarray(self._lat2d)
        
        # Replace non-finite values with NaN
        self._lon2d[~np.isfinite(self._lon2d)] = np.nan
        self._lat2d[~np.isfinite(self._lat2d)] = np.nan

        center_lat = self.site_config["center_coords"]["lat"]
        center_lon = self.site_config["center_coords"]["lon"]
        dist_sq = (self._lat2d - center_lat)**2 + (self._lon2d - center_lon)**2
        y0, x0 = np.unravel_index(np.nanargmin(dist_sq), dist_sq.shape)

        res_y = _haversine_km(self._lat2d[y0 - 1, x0], self._lon2d[y0 - 1, x0], self._lat2d[y0 + 1, x0], self._lon2d[y0 + 1, x0]) / 2
        res_x = _haversine_km(self._lat2d[y0, x0 - 1], self._lon2d[y0, x0 - 1], self._lat2d[y0, x0 + 1], self._lon2d[y0, x0 + 1]) / 2
        avg_pixel_res_km = (res_x + res_y) / 2

        n_pixels = int(np.ceil(CONFIG["aoi_radius_km"] / avg_pixel_res_km))
        y_start, y_end = max(0, y0 - n_pixels), min(self._lat2d.shape[0], y0 + n_pixels + 1)
        x_start, x_end = max(0, x0 - n_pixels), min(self._lat2d.shape[1], x0 + n_pixels + 1)
        self._aoi_slice = (slice(y_start, y_end), slice(x_start, x_end))

        self.region_info = _determine_region_info(center_lat, center_lon)
        region_ext = self.region_info
        region_mask = ((self._lat2d >= region_ext["lat"][0]) & (self._lat2d <= region_ext["lat"][1]) &
                       (self._lon2d >= region_ext["lon"][0]) & (self._lon2d <= region_ext["lon"][1]))
        ys, xs = np.where(region_mask)
        if ys.size > 0:
            self._region_slice = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        else:
            logging.warning("Region slice could not be determined, using AOI as region.")
            self._region_slice = self._aoi_slice


    def _map_cloud_types(self, ct_raw: np.ndarray) -> np.ndarray:
        """Map cloud types to simplified categories."""
        ct_mapped = np.zeros_like(ct_raw, dtype=np.int8)
        ct_mapped[np.isin(ct_raw, CONFIG["cloud_categories"]["Warm"])] = 1
        ct_mapped[np.isin(ct_raw, CONFIG["cloud_categories"]["Cold"])] = 2
        return ct_mapped

    def _calculate_percentages(self, arr: np.ndarray) -> dict:
        """Calculate cloud type percentages from a data array."""
        total_pixels = arr.size if arr.size > 0 else 1
        unique, counts = np.unique(arr, return_counts=True)
        return {k: v * 100 / total_pixels for k, v in zip(unique, counts)}

    def _plot_frame(self, ts: datetime, ct_region: np.ndarray, ct_aoi: np.ndarray, pct_aoi: dict,
                   day_rank: int, day_score: float, day_score_norm: float) -> str:
        """Create a single plot frame using pcolormesh and return the file path."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

        # --- Region Overview Plot (Left) ---
        region_ext = self.region_info
        ax1.set_extent(region_ext["lon"] + region_ext["lat"], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='gray')
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        # Handle masked arrays and non-finite values for region plotting
        lon_region = self._lon2d[self._region_slice]
        lat_region = self._lat2d[self._region_slice]
        
        # Debug: Check shapes and types
        logging.debug(f"lon_region shape: {lon_region.shape}, type: {type(lon_region)}")
        logging.debug(f"lat_region shape: {lat_region.shape}, type: {type(lat_region)}")
        logging.debug(f"ct_region shape: {ct_region.shape}, type: {type(ct_region)}")
        
        # Convert masked arrays to regular arrays with NaN for masked values
        if hasattr(lon_region, 'mask'):
            lon_region = np.where(lon_region.mask, np.nan, lon_region.data)
        if hasattr(lat_region, 'mask'):
            lat_region = np.where(lat_region.mask, np.nan, lat_region.data)
        
        # Replace non-finite values with NaN
        lon_region = np.where(np.isfinite(lon_region), lon_region, np.nan)
        lat_region = np.where(np.isfinite(lat_region), lat_region, np.nan)
        
        # Ensure all arrays have the same shape
        if lon_region.shape != ct_region.shape:
            logging.warning(f"Shape mismatch: lon_region {lon_region.shape} vs ct_region {ct_region.shape}")
            # Try to match shapes by using the smaller one
            min_shape = (min(lon_region.shape[0], ct_region.shape[0]), 
                        min(lon_region.shape[1], ct_region.shape[1]))
            lon_region = lon_region[:min_shape[0], :min_shape[1]]
            lat_region = lat_region[:min_shape[0], :min_shape[1]]
            ct_region = ct_region[:min_shape[0], :min_shape[1]]
        
        # Try pcolormesh first, fallback to imshow if it fails
        try:
            im = ax1.pcolormesh(lon_region, lat_region, ct_region,
                               cmap=self.cmap, vmin=-0.5, vmax=2.5, shading='auto')
        except (ValueError, TypeError) as e:
            logging.warning(f"pcolormesh failed, using imshow: {e}")
            im = ax1.imshow(ct_region, cmap=self.cmap, vmin=-0.5, vmax=2.5, 
                           extent=[np.nanmin(lon_region), np.nanmax(lon_region), 
                                  np.nanmin(lat_region), np.nanmax(lat_region)],
                           aspect='auto', origin='lower')

        aoi_lon, aoi_lat = self._lon2d[self._aoi_slice], self._lat2d[self._aoi_slice]
        rect = Rectangle((np.nanmin(aoi_lon), np.nanmin(aoi_lat)), 
                         np.nanmax(aoi_lon) - np.nanmin(aoi_lon),
                         np.nanmax(aoi_lat) - np.nanmin(aoi_lat),
                         linewidth=2, edgecolor='black', facecolor='none', label='AOI')
        ax1.add_patch(rect)
        ax1.set_title(f"{self.region_info['name']} Overview - {ts.strftime('%H:%M UTC')}")
        ax1.legend(loc="upper right")

        # --- AOI Detail Plot (Right) ---
        padding_deg = 0.05
        ax2.set_extent([np.nanmin(aoi_lon) - padding_deg, np.nanmax(aoi_lon) + padding_deg,
                       np.nanmin(aoi_lat) - padding_deg, np.nanmax(aoi_lat) + padding_deg])
        # Handle masked arrays and non-finite values for AOI plotting
        aoi_lon_clean = aoi_lon.copy()
        aoi_lat_clean = aoi_lat.copy()
        
        # Debug: Check shapes and types
        logging.debug(f"aoi_lon shape: {aoi_lon_clean.shape}, type: {type(aoi_lon_clean)}")
        logging.debug(f"aoi_lat shape: {aoi_lat_clean.shape}, type: {type(aoi_lat_clean)}")
        logging.debug(f"ct_aoi shape: {ct_aoi.shape}, type: {type(ct_aoi)}")
        
        # Convert masked arrays to regular arrays with NaN for masked values
        if hasattr(aoi_lon_clean, 'mask'):
            aoi_lon_clean = np.where(aoi_lon_clean.mask, np.nan, aoi_lon_clean.data)
        if hasattr(aoi_lat_clean, 'mask'):
            aoi_lat_clean = np.where(aoi_lat_clean.mask, np.nan, aoi_lat_clean.data)
        
        # Replace non-finite values with NaN
        aoi_lon_clean = np.where(np.isfinite(aoi_lon_clean), aoi_lon_clean, np.nan)
        aoi_lat_clean = np.where(np.isfinite(aoi_lat_clean), aoi_lat_clean, np.nan)
        
        # Ensure all arrays have the same shape
        if aoi_lon_clean.shape != ct_aoi.shape:
            logging.warning(f"Shape mismatch: aoi_lon {aoi_lon_clean.shape} vs ct_aoi {ct_aoi.shape}")
            # Try to match shapes by using the smaller one
            min_shape = (min(aoi_lon_clean.shape[0], ct_aoi.shape[0]), 
                        min(aoi_lon_clean.shape[1], ct_aoi.shape[1]))
            aoi_lon_clean = aoi_lon_clean[:min_shape[0], :min_shape[1]]
            aoi_lat_clean = aoi_lat_clean[:min_shape[0], :min_shape[1]]
            ct_aoi = ct_aoi[:min_shape[0], :min_shape[1]]
        
        # Try pcolormesh first, fallback to imshow if it fails
        try:
            ax2.pcolormesh(aoi_lon_clean, aoi_lat_clean, ct_aoi, cmap=self.cmap, vmin=-0.5, vmax=2.5, shading='auto')
        except (ValueError, TypeError) as e:
            logging.warning(f"AOI pcolormesh failed, using imshow: {e}")
            ax2.imshow(ct_aoi, cmap=self.cmap, vmin=-0.5, vmax=2.5,
                      extent=[np.nanmin(aoi_lon_clean), np.nanmax(aoi_lon_clean),
                             np.nanmin(aoi_lat_clean), np.nanmax(aoi_lat_clean)],
                      aspect='auto', origin='lower')

        gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter(zero_direction_label=True, number_format='.2f')
        gl.yformatter = LatitudeFormatter(number_format='.2f')

        warm_pct = pct_aoi.get(1, 0.0)
        cold_pct = pct_aoi.get(2, 0.0)
        ax2.set_title(f'AOI Detail\nWarm Clouds: {warm_pct:.1f}%, Cold Clouds: {cold_pct:.1f}%')

        # --- Overall Figure Settings ---
        fig.suptitle(f'Site: {self.site_name} | Date: {ts.strftime("%Y-%m-%d")} | Rank #{day_rank} (Score: {day_score_norm:.1f})', fontsize=16)
        
        cbar = fig.colorbar(im, ax=[ax1, ax2], orientation='horizontal', shrink=0.6, pad=0.08)
        cbar.set_label('Cloud Type')
        plot_cats = CONFIG["plot_categories"]
        ticks = sorted(plot_cats.keys())
        labels = [plot_cats[k]["name"] for k in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)

        frame_path = self.png_dir / f"frame_{ts.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(frame_path)
        plt.close(fig)
        return str(frame_path)

    def process_day(self, date_str: str, rank: int, score: float, score_norm: float):
        """Process a single day and create a GIF."""
        logging.info(f"Processing Day: {date_str} (Rank #{rank}, Score Norm: {score_norm:.1f})")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day_dir = self.data_root / date_obj.strftime('%Y_%m_%d')

        if not day_dir.exists():
            logging.warning(f"Data directory not found, skipping: {day_dir}")
            return

        ct_files = sorted(day_dir.glob(CONFIG["data_glob_pattern"]))
        if not ct_files:
            logging.warning(f"No NetCDF files found in {day_dir}, skipping.")
            return

        with xr.open_dataset(ct_files[0]) as ds:
            self._setup_grid(ds)

        frame_paths = []
        for file_path in tqdm(ct_files, desc=f"Day {date_str}", unit="frame"):
            try:
                ts_str = file_path.stem.split('_')[-1]
                ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ")

                if 6 <= ts.hour < 18: # Process daytime hours only
                    with xr.open_dataset(file_path) as ds:
                        ct_raw = ds.ct.values
                        ct_region_raw = ct_raw[self._region_slice]
                        ct_aoi_raw = ct_raw[self._aoi_slice]

                        ct_region_mapped = self._map_cloud_types(ct_region_raw)
                        ct_aoi_mapped = self._map_cloud_types(ct_aoi_raw)
                        
                        pct_aoi = self._calculate_percentages(ct_aoi_mapped)
                        
                        frame_path = self._plot_frame(ts, ct_region_mapped, ct_aoi_mapped, pct_aoi, rank, score, score_norm)
                        frame_paths.append(frame_path)
            except Exception as e:
                logging.error(f"Failed to process frame {file_path.name}: {e}", exc_info=False)

        if frame_paths:
            gif_path = self.gif_dir / f"rank_{rank:02d}_{self.site_name}_{date_str}_score_{score_norm:.1f}.gif"
            logging.info(f"Creating GIF with {len(frame_paths)} frames...")
            images = [imageio.v2.imread(p) for p in frame_paths]
            imageio.mimsave(gif_path, images, duration=200, loop=0) # 200ms per frame
            logging.info(f"Successfully created GIF: {gif_path}")

            for p in frame_paths:
                Path(p).unlink() # Clean up PNG frames
        else:
            logging.warning(f"No frames were generated for {date_str}, GIF not created.")

    def run(self):
        """Process all top-ranked days."""
        logging.info(f"Starting visualization for Top {self.top_n} days for site '{self.site_name}'")
        for idx, row in self.top_days.iterrows():
            self.process_day(
                date_str=row['date'],
                rank=idx + 1,
                score=row['les_score'],
                score_norm=row['les_score_norm']
            )
        logging.info("Visualization generation complete!")

def main():
    """Main execution function."""
    # --- Required: Set the root for satellite data ---
    data_root = Path("/mnt/m0/y-m.saint-drenan/data/NWCSAF_CloudType/2024/")
    if not data_root.exists():
        logging.error(f"Data root directory not found: {data_root}")
        return

    # --- Use environment variables for flexibility (e.g., in a batch script) ---
    site = os.environ.get('SITE') # SITE must be set
    if not site:
        logging.error("SITE environment variable is required. Please set it before running the script.")
        return
    month = os.environ.get('MONTH', None)      # e.g., '2024-05'
    top_n = int(os.environ.get('TOP_N', 3))    # Number of top days to visualize
    
    logging.info(f"SITE set to: {site}")
    if month:
        logging.info(f"MONTH specified: {month}")

    # --- Locate the correct output directory and ranking file ---
    # The script now assumes output files are in a directory named after the site
    # e.g., ./PALAISEAU/output_2024-05/les_suitability_ranking_....csv
    site_dir = Path(site)
    if not site_dir.exists():
        logging.error(f"Site directory '{site_dir}' not found. Please run analysis first.")
        return

    if month:
        out_root = site_dir / f"output_{month}"
        if not out_root.exists():
            logging.error(f"Specified output directory not found: {out_root}")
            return
    else:
        # Find the most recent output directory if month is not specified
        output_dirs = sorted([d for d in site_dir.glob("output_*") if d.is_dir()], 
                             key=lambda x: x.stat().st_mtime, reverse=True)
        if not output_dirs:
            logging.error(f"No 'output_*' directories found in {site_dir}")
            return
        out_root = output_dirs[0]
        logging.info(f"Found most recent output directory: {out_root}")

    # Find the ranking CSV file within the chosen output directory
    ranking_csvs = list(out_root.glob("les_suitability_ranking_*.csv"))
    if not ranking_csvs:
        logging.error(f"No 'les_suitability_ranking_*.csv' file found in {out_root}")
        return
    
    ranking_csv = ranking_csvs[0]
    logging.info(f"Using ranking file: {ranking_csv}")
    
    visualizer = TopDayVisualizer(data_root, out_root, ranking_csv, site=site, top_n=top_n)
    visualizer.run()

if __name__ == "__main__":
    # For this script to work, you need a 'countries.geojson' file in the same
    # directory as the script. You can download one from:
    # https://github.com/datasets/geo-countries/blob/master/data/countries.geojson
    main()