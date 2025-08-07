#!/usr/bin/env python3
"""
Monthly LES Cloud Screening and Ranking Processor
=================================================

This script analyzes an entire month of NWCSAF cloud type data to identify
and rank the most suitable days for Large Eddy Simulations (LES). It processes
all files for a given month, calculates hourly cloud statistics, and then scores
each day based on a specialized LES suitability metric.

Core Features:
- Processes a full month of data (e.g., "2024-09").
- Ranks days using an LES Suitability Score based on:
  - Daytime (06-18h UTC) thick cloud presence.
  - Cloud 'breakability' (variability of thick cloud cover).
  - Penalty for thin clouds.
- Generates a final ranked CSV file for easy day selection.
- Retains optional, on-demand generation of daily PNGs.

Author: Max Aragon Cerecedes
Revised: July 29, 2025

Requirements:
- Python 3.8+
- xarray, pandas, numpy, matplotlib, cartopy, pyproj, tqdm
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Transformer
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "target_month": "2024-04", # Target month to process
    "center_coords": None, # Will be set via command line arguments
    "aoi_radius_km": 20,
    "france_extent": {"lon": [-5, 10], "lat": [42, 52]},
    "data_glob_pattern": "S_NWC_CT_MSG*_globeM-VISIR_*.nc",
    "geos_proj_str": "+proj=geos +a=6378137.0 +b=6356752.3 +lon_0=0.0 +h=35785863.0 +sweep=y",

    "cloud_categories": {
        "Other": [1, 2, 3, 4, 255], 
        "Warm": [5, 6, 7, 8, 9, 10],  # Fog + Thick clouds
        "Cold": [11, 12, 13, 14, 15],  # Thin clouds
    },

    "les_scoring": {
        "daytime_hours": (6, 18), # Start and end hour (UTC) for daytime analysis
        "min_thick_cloud_presence_pct": 10.0, # Reduced min avg thick cloud % for a day to be considered
        "breakability_norm_factor": 15.0, # Reduced normalization for std dev of thick clouds (more sensitive to variability)
        "persistence_penalty_factor": 0.3, # Factor to penalize persistent warm clouds (lower = more penalty)
        "breakability_weight": 0.75, # Weight for breakability score (75% as per user preference)
        "traditional_weight": 0.25, # Weight for traditional criteria (25% as per user preference)
        "max_score": 142.5 # Theoretical maximum LES score
    }
}

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_GEOD = Geod(ellps='WGS84')

def _haversine_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Calculate great-circle distance in kilometers using pyproj."""
    _, _, dist_m = _GEOD.inv(lon0, lat0, lon1, lat1)
    return dist_m / 1000

class MonthlyLESAnalysis:
    """Orchestrates the monthly cloud screening and ranking process."""

    def __init__(self, data_root: Path, out_root: Path, config: Dict):
        self.data_root = data_root
        self.out_root = out_root
        self.config = config
        self.out_root.mkdir(parents=True, exist_ok=True)

        # Internal state for grid information
        self._lat2d, self._lon2d = None, None
        self._fr_slice, self._aoi_slice = (), ()
        self.aoi_dims_km = (0.0, 0.0)



    def _list_days_in_month(self) -> List[Path]:
        """Finds all daily data directories for the configured month."""
        month_str = self.config["target_month"].replace("-", "_")
        day_dirs = sorted(self.data_root.glob(f"{month_str}_*"))
        logging.info(f"Found {len(day_dirs)} data directories for month {self.config['target_month']}.")
        return day_dirs

    def _setup_grid(self, ds: xr.Dataset):
        """Initializes the geographic grid and calculates data slices."""
        logging.info("Initializing geographic grid from the first file...")
        proj_str = ds.attrs.get('gdal_projection', self.config["geos_proj_str"])
        transformer = Transformer.from_crs(proj_str, "epsg:4326", always_xy=True)
        nx_grid, ny_grid = np.meshgrid(ds.nx.values, ds.ny.values)
        self._lon2d, self._lat2d = transformer.transform(nx_grid, ny_grid)

        # Clean non-finite values from coordinate arrays
        self._lon2d[~np.isfinite(self._lon2d)] = np.nan
        self._lat2d[~np.isfinite(self._lat2d)] = np.nan

        center_lat, center_lon = self.config["center_coords"]["lat"], self.config["center_coords"]["lon"]
        dist_sq = (self._lat2d - center_lat)**2 + (self._lon2d - center_lon)**2
        y0, x0 = np.unravel_index(np.nanargmin(dist_sq), dist_sq.shape)

        res_y = _haversine_km(self._lat2d[y0 - 1, x0], self._lon2d[y0 - 1, x0], self._lat2d[y0 + 1, x0], self._lon2d[y0 + 1, x0]) / 2
        res_x = _haversine_km(self._lat2d[y0, x0 - 1], self._lon2d[y0, x0 - 1], self._lat2d[y0, x0 + 1], self._lon2d[y0, x0 + 1]) / 2
        avg_pixel_res_km = (res_x + res_y) / 2

        n_pixels = int(np.ceil(self.config["aoi_radius_km"] / avg_pixel_res_km))
        y_start, y_end = max(0, y0 - n_pixels), min(self._lat2d.shape[0], y0 + n_pixels + 1)
        x_start, x_end = max(0, x0 - n_pixels), min(self._lat2d.shape[1], x0 + n_pixels + 1)
        self._aoi_slice = (slice(y_start, y_end), slice(x_start, x_end))
        
        fr_ext = self.config["france_extent"]
        fr_mask = ((self._lat2d >= fr_ext["lat"][0]) & (self._lat2d <= fr_ext["lat"][1]) &
                   (self._lon2d >= fr_ext["lon"][0]) & (self._lon2d <= fr_ext["lon"][1]))
        ys, xs = np.where(fr_mask)
        if ys.size > 0:
            self._fr_slice = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

        aoi_lat, aoi_lon = self._lat2d[self._aoi_slice], self._lon2d[self._aoi_slice]
        self.aoi_dims_km = (_haversine_km(aoi_lat[0,0], aoi_lon[0,0], aoi_lat[0,-1], aoi_lon[0,-1]),
                            _haversine_km(aoi_lat[0,0], aoi_lon[0,0], aoi_lat[-1,0], aoi_lon[-1,0]))
        logging.info(f"Grid setup complete. AOI is {self.aoi_dims_km[0]:.1f} x {self.aoi_dims_km[1]:.1f} km.")

    def _process_file(self, file_path: Path) -> Optional[Dict]:
        """Processes a single NetCDF file and returns a dictionary of metrics."""
        try:
            ts = datetime.strptime(file_path.stem.split('_')[-1], "%Y%m%dT%H%M%SZ")
            with xr.open_dataset(file_path) as ds:
                if self._lat2d is None: self._setup_grid(ds)

                ct_aoi_raw = ds.ct.isel(ny=self._aoi_slice[0], nx=self._aoi_slice[1]).values
                pct_aoi = self._calculate_percentages(ct_aoi_raw)

            output = {"timestamp": ts}
            output.update({f"aoi_ct_{k}": v for k, v in pct_aoi.items()})

            return output
        except Exception as e:
            logging.error(f"Failed to process {file_path.name}: {e}", exc_info=False)
            return None

    def _calculate_les_score(self, day_df: pd.DataFrame) -> Dict:
        """Calculates the LES suitability score and related metrics for a single day."""
        params = self.config["les_scoring"]
        start_hr, end_hr = params["daytime_hours"]

        daytime_df = day_df[(day_df['timestamp'].dt.hour >= start_hr) & (day_df['timestamp'].dt.hour < end_hr)]
        if daytime_df.empty: return {"les_score": 0}

        warm_cols = [f'aoi_ct_{i}' for i in self.config["cloud_categories"]["Warm"]]
        cold_cols = [f'aoi_ct_{i}' for i in self.config["cloud_categories"]["Cold"]]

        warm_pct = daytime_df[[c for c in warm_cols if c in daytime_df.columns]].sum(axis=1)
        cold_pct = daytime_df[[c for c in cold_cols if c in daytime_df.columns]].sum(axis=1)

        mean_warm = warm_pct.mean()
        std_warm = warm_pct.std()
        max_warm = warm_pct.max()  # Add maximum warm cloud percentage
        min_warm = warm_pct.min()  # Add minimum warm cloud percentage
        mean_cold = cold_pct.mean()
        
        # Calculate hourly changes to detect extreme swings vs gradual transitions
        hourly_changes = warm_pct.diff().abs()
        max_hourly_change = hourly_changes.max() if len(hourly_changes) > 0 else 0
        mean_hourly_change = hourly_changes.mean() if len(hourly_changes) > 0 else 0

        if pd.isna(mean_warm) or mean_warm < params["min_thick_cloud_presence_pct"]:
            score = 0
        else:
            # Breakability score (variability of warm clouds) - higher is better for cumulus
            # Add stronger penalty for low variability
            breakability_score = min(1.0, std_warm / params["breakability_norm_factor"])
            
            # Penalty for low cloud variability (stable cloud cover is bad for LES)
            if std_warm < 15:  # Very low variability - strong penalty
                low_variability_penalty = 0.2  # Very strong penalty for stable clouds
            elif std_warm < 25:  # Low variability - moderate penalty
                low_variability_penalty = 0.4  # Strong penalty
            elif std_warm < 35:  # Moderate variability - light penalty
                low_variability_penalty = 0.7  # Moderate penalty
            elif std_warm > 45:  # Extreme variability - penalize too much swing
                low_variability_penalty = 0.5  # Penalty for extreme variability
            else:
                low_variability_penalty = 1.0  # Good variability - no penalty
            
            # Penalty for extreme hourly cloud transitions (bad for LES)
            if max_hourly_change > 50:  # Extreme hourly swing (e.g., 50%+ change in one hour)
                extreme_transition_penalty = 0.2  # Very strong penalty for extreme swings
            elif max_hourly_change > 35:  # Large hourly swing
                extreme_transition_penalty = 0.4  # Strong penalty
            elif max_hourly_change > 25:  # Moderate hourly swing
                extreme_transition_penalty = 0.6  # Moderate penalty
            elif max_hourly_change > 15:  # Small hourly swing
                extreme_transition_penalty = 0.8  # Light penalty
            else:
                extreme_transition_penalty = 1.0  # Gradual transitions - no penalty
            
            # Penalty for persistent cloud cover (stable clouds are bad for LES)
            # Lower std_warm relative to mean_warm indicates more persistent clouds
            if mean_warm > 0:
                persistence_ratio = std_warm / mean_warm
                persistent_cloud_penalty = min(1.0, persistence_ratio / params["persistence_penalty_factor"])
            else:
                persistent_cloud_penalty = 0
            
            # Penalty for non-optimal warm cloud coverage (sweet spot: 45-80%)
            if mean_warm > 90:  # Very high cloud cover - strong penalty
                warm_coverage_penalty = 0.1  # Very strong penalty for overcast conditions
            elif mean_warm > 80:  # High cloud cover - moderate penalty
                warm_coverage_penalty = 0.2  # Moderate penalty
            elif mean_warm < 45:  # Low cloud cover - penalty for insufficient clouds
                warm_coverage_penalty = 1.0  # No penalty for insufficient clouds
            elif mean_warm < 25:  # Very low cloud cover - strong penalty
                warm_coverage_penalty = 0.1  # Very strong penalty for clear conditions
            else:
                warm_coverage_penalty = 1.0  # Sweet spot (45-80%) - no penalty
            
            # Cloud Cover Duration Metrics - Compound Penalty
            # Calculate duration of different cloud cover levels throughout the day
            total_measurements = len(warm_pct)
            
            # Duration of overcast conditions (>90% cloud cover)
            overcast_duration = (warm_pct > 90).sum() / total_measurements
            # Duration of high cloud cover (75-90%)
            high_cloud_duration = ((warm_pct >= 75) & (warm_pct <= 90)).sum() / total_measurements
            # Duration of moderate cloud cover (45-75%)
            moderate_cloud_duration = ((warm_pct >= 45) & (warm_pct < 75)).sum() / total_measurements
            # Duration of low cloud cover (15-45%)
            low_cloud_duration = ((warm_pct >= 15) & (warm_pct < 45)).sum() / total_measurements
            # Duration of clear conditions (<15%)
            clear_duration = (warm_pct < 15).sum() / total_measurements
            
            # Persistent overcast penalty - penalize 100% warm clouds over 10+ consecutive frames
            persistent_overcast_penalty = 1.0
            consecutive_overcast_count = 0
            max_consecutive_overcast = 0
            
            for warm_value in warm_pct:
                if warm_value >= 100.0:  # 100% warm clouds
                    consecutive_overcast_count += 1
                else:
                    if consecutive_overcast_count > max_consecutive_overcast:
                        max_consecutive_overcast = consecutive_overcast_count
                    consecutive_overcast_count = 0
            
            # Check the last sequence
            if consecutive_overcast_count > max_consecutive_overcast:
                max_consecutive_overcast = consecutive_overcast_count
            
            # Apply penalty for persistent overcast
            if max_consecutive_overcast >= 15:  # 15+ consecutive frames (3.75+ hours)
                persistent_overcast_penalty = 0.1  # Very strong penalty (90% reduction)
            elif max_consecutive_overcast >= 12:  # 12+ consecutive frames (3+ hours)
                persistent_overcast_penalty = 0.2  # Strong penalty (80% reduction)
            elif max_consecutive_overcast >= 10:  # 10+ consecutive frames (2.5+ hours)
                persistent_overcast_penalty = 0.3  # Strong penalty (70% reduction)
            elif max_consecutive_overcast >= 8:  # 8+ consecutive frames (2+ hours)
                persistent_overcast_penalty = 0.5  # Moderate penalty (50% reduction)
            elif max_consecutive_overcast >= 6:  # 6+ consecutive frames (1.5+ hours)
                persistent_overcast_penalty = 0.7  # Light penalty (30% reduction)
            
            # Compound penalty/bonus based on cloud cover duration patterns
            # Ideal LES pattern: brief high peaks + good breakability + moderate development time
            cloud_duration_factor = 1.0
            
            # Penalize persistent overcast (bad for LES)
            if overcast_duration > 0.5:  # More than 50% of day overcast
                cloud_duration_factor *= 0.3  # Very strong penalty
            elif overcast_duration > 0.3:  # 30-50% overcast
                cloud_duration_factor *= 0.6  # Strong penalty
            elif overcast_duration > 0.1:  # 10-30% overcast
                cloud_duration_factor *= 0.8  # Moderate penalty
            
            # Bonus for good breakability (clear periods)
            if clear_duration > 0.2:  # More than 20% clear periods
                cloud_duration_factor *= 1.2  # Bonus for breakability
            elif clear_duration > 0.1:  # 10-20% clear periods
                cloud_duration_factor *= 1.1  # Small bonus
            
            # Bonus for moderate development time (not too brief, not too long)
            if 0.2 <= moderate_cloud_duration <= 0.5:  # Good development time
                cloud_duration_factor *= 1.1  # Bonus for good development
            elif moderate_cloud_duration < 0.1:  # Too brief development
                cloud_duration_factor *= 0.8  # Penalty for insufficient development
            
            # Penalize too much time in low cloud cover (insufficient development)
            if low_cloud_duration > 0.6:  # More than 60% in low cloud cover
                cloud_duration_factor *= 0.7  # Penalty for insufficient development
            
            # Ensure factor stays within reasonable bounds
            cloud_duration_factor = max(0.2, min(1.5, cloud_duration_factor))
            
            # For backward compatibility, set max_cloud_penalty and min_cloud_penalty
            # based on the compound factor
            max_cloud_penalty = cloud_duration_factor
            min_cloud_penalty = cloud_duration_factor
            
            # Reduce min cloud penalty for days with good overall characteristics
            if mean_warm > 60 and std_warm > 25 and mean_cold < 5:  # Good overall day
                min_cloud_penalty = min(1.0, min_cloud_penalty * 2.0)  # Reduce penalty by 100% (no penalty)
            
            # Reduce max cloud penalty for days with good variability
            if std_warm > 25:  # High variability - reduce penalty
                max_cloud_penalty = min(1.0, max_cloud_penalty * 1.2)  # Reduce penalty by 20%
            
            # Further reduce max cloud penalty for days with excellent overall characteristics
            if mean_warm > 60 and std_warm > 25 and mean_cold < 5:  # Excellent overall day
                max_cloud_penalty = min(1.0, max_cloud_penalty * 1.5)  # Additional 50% reduction
            
            # Enhanced cold cloud penalty (thin clouds are bad for LES)
            # MUCH stronger penalty for any cold cloud presence - more aggressive thresholds
            if mean_cold > 10:  # High cold cloud cover - extremely strong penalty
                cold_cloud_factor = 0.05  # 95% penalty - almost disqualifying
            elif mean_cold > 5:  # Moderate cold cloud cover - very strong penalty
                cold_cloud_factor = 0.1  # 90% penalty
            elif mean_cold > 2:  # Light cold cloud cover - strong penalty
                cold_cloud_factor = 0.2  # 80% penalty
            elif mean_cold > 0.5:  # Minimal cold cloud cover - moderate penalty
                cold_cloud_factor = 0.4  # 60% penalty
            elif mean_cold > 0.1:  # Very minimal cold cloud cover - light penalty
                cold_cloud_factor = 0.6  # 40% penalty
            else:
                cold_cloud_factor = 1.5  # Zero cold clouds - stronger bonus multiplier (50% boost)
            
            # Combined score with user-specified weights
            # 75% breakability + 25% traditional criteria (mean warm presence)
            breakability_component = breakability_score * persistent_cloud_penalty * cold_cloud_factor * warm_coverage_penalty * max_cloud_penalty * min_cloud_penalty * low_variability_penalty * extreme_transition_penalty * persistent_overcast_penalty
            traditional_component = (mean_warm / 100.0) * cold_cloud_factor * warm_coverage_penalty * max_cloud_penalty * min_cloud_penalty * low_variability_penalty * extreme_transition_penalty * persistent_overcast_penalty
            
            score = (params["breakability_weight"] * breakability_component + 
                    params["traditional_weight"] * traditional_component) * 100
            
            # Calculate normalized score (0-100 scale)
            score_norm = np.clip(100.0 * score / params["max_score"], 0.0, 100.0)
            
            # Determine LES suitability flag based on normalized score
            if score_norm < 2.4:
                les_suitable = "Unlikely"
            elif score_norm < 3.0:
                les_suitable = "Probable"
            else:
                les_suitable = "Likely"

        return {
            "les_score": score,
            "les_score_norm": score_norm if 'score_norm' in locals() else 0.0,
            "LES_suitable": les_suitable if 'les_suitable' in locals() else "Unlikely",
            "mean_warm_pct": mean_warm,
            "std_warm_pct": std_warm,
            "max_warm_pct": max_warm,
            "min_warm_pct": min_warm,
            "mean_cold_pct": mean_cold,
            "max_hourly_change": max_hourly_change,
            "mean_hourly_change": mean_hourly_change,
            "breakability_score": min(1.0, std_warm / params["breakability_norm_factor"]) if not pd.isna(std_warm) else 0,
            "persistent_cloud_penalty": persistent_cloud_penalty if 'persistent_cloud_penalty' in locals() else 0,
            "warm_coverage_penalty": warm_coverage_penalty if 'warm_coverage_penalty' in locals() else 1.0,
            "max_cloud_penalty": max_cloud_penalty if 'max_cloud_penalty' in locals() else 1.0,
            "min_cloud_penalty": min_cloud_penalty if 'min_cloud_penalty' in locals() else 1.0,
            "low_variability_penalty": low_variability_penalty if 'low_variability_penalty' in locals() else 1.0,
            "extreme_transition_penalty": extreme_transition_penalty if 'extreme_transition_penalty' in locals() else 1.0,
            "cold_cloud_factor": cold_cloud_factor if 'cold_cloud_factor' in locals() else 1.0,
            "cloud_duration_factor": cloud_duration_factor if 'cloud_duration_factor' in locals() else 1.0,
            "persistent_overcast_penalty": persistent_overcast_penalty if 'persistent_overcast_penalty' in locals() else 1.0,
            "max_consecutive_overcast": max_consecutive_overcast if 'max_consecutive_overcast' in locals() else 0,
            "overcast_duration": overcast_duration if 'overcast_duration' in locals() else 0.0,
            "clear_duration": clear_duration if 'clear_duration' in locals() else 0.0,
            "moderate_cloud_duration": moderate_cloud_duration if 'moderate_cloud_duration' in locals() else 0.0
        }

    def run(self):
        """Executes the full monthly analysis and ranking pipeline."""
        logging.info(f"🚀 Starting Monthly LES Analysis for {self.config['target_month']}")
        day_dirs = self._list_days_in_month()
        if not day_dirs: return

        all_results = []
        for day_path in day_dirs:
            daily_files = sorted(day_path.glob(self.config["data_glob_pattern"]))
            if not daily_files: continue
            
            logging.info(f"--- Processing {len(daily_files)} files for {day_path.name} ---")
            for f in tqdm(daily_files, desc=f"Day {day_path.name[-2:]}"):
                if (res := self._process_file(f)):
                    all_results.append(res)

        if not all_results:
            logging.warning("No data was successfully processed for the entire month.")
            return

        # --- Final Analysis and Ranking ---
        df = pd.DataFrame(all_results).sort_values('timestamp').reset_index(drop=True)

        day_groups = df.groupby(df['timestamp'].dt.date)
        ranking_data = [
            {"date": date, **self._calculate_les_score(group)}
            for date, group in day_groups
        ]

        ranking_df = pd.DataFrame(ranking_data).sort_values("les_score", ascending=False).fillna(0)
        ranking_df = ranking_df[['date', 'LES_suitable', 'les_score', 'les_score_norm', 'mean_warm_pct', 'std_warm_pct', 'max_warm_pct', 'min_warm_pct', 'mean_cold_pct', 'max_hourly_change', 'mean_hourly_change', 'breakability_score', 'persistent_cloud_penalty', 'warm_coverage_penalty', 'max_cloud_penalty', 'min_cloud_penalty', 'low_variability_penalty', 'extreme_transition_penalty', 'cold_cloud_factor', 'cloud_duration_factor', 'persistent_overcast_penalty', 'max_consecutive_overcast', 'overcast_duration', 'clear_duration', 'moderate_cloud_duration']]

        site_suffix = f"_{self.config.get('site', 'unknown')}" if self.config.get('site') else ""
        csv_path = self.out_root / f"les_suitability_ranking_{self.config['target_month']}{site_suffix}.csv"
        ranking_df.to_csv(csv_path, index=False, float_format='%.2f')

        logging.info("\n" + "="*60)
        logging.info(f"✅ Monthly Analysis Complete!")
        logging.info(f"Top 5 Days for LES in {self.config['target_month']}:")
        print(ranking_df.head().to_string(index=False))
        logging.info(f"\nFull ranked results saved to: {csv_path}")

    # --- Helper methods for data processing ---
    @staticmethod
    def _calculate_percentages(arr: np.ndarray) -> Dict[int, float]:
        flat = arr[~np.isnan(arr)].astype(int)
        total_pixels = flat.size if flat.size > 0 else 1
        unique, counts = np.unique(flat, return_counts=True)
        return {k: v * 100 / total_pixels for k, v in zip(unique, counts)}

def main():
    """Main function to parse arguments and run the processor."""
    parser = argparse.ArgumentParser(description="Monthly LES Cloud Screening and Ranking Processor")
    parser.add_argument("--month", default=CONFIG['target_month'], help="Target month in YYYY-MM format.")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/m0/y-m.saint-drenan/data/NWCSAF_CloudType/2024/"), help="Root directory for NWCSAF data.")
    parser.add_argument("--out-root", type=Path, help="Output directory. Defaults to 'output_YYYY-MM'.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of center point")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of center point")
    parser.add_argument("--radius", type=float, default=20, help="AOI radius in km (default: 20)")
    parser.add_argument("--site", type=str, help="Site name (e.g., PALAISEAU, MUNICH)")
    args = parser.parse_args()
    
    CONFIG['target_month'] = args.month
    CONFIG['aoi_radius_km'] = args.radius
    CONFIG['center_coords'] = {"lat": args.lat, "lon": args.lon}
    CONFIG['site'] = args.site
    
    logging.info(f"Using coordinates: {args.lat:.4f}°N, {args.lon:.4f}°E")
    
    out_root = args.out_root or Path(f"output_{args.month}")
    
    analysis = MonthlyLESAnalysis(args.data_root, out_root, CONFIG)
    analysis.run()

if __name__ == "__main__":
    main()