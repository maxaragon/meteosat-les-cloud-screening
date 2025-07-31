#!/usr/bin/env python3
"""
Debug script to examine cloud type data for a specific day
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pyproj import Transformer

# Configuration
CONFIG = {
    "center_coords": {"lat": 48.71725, "lon": 2.20884}, # SIRTA
    "aoi_radius_km": 20,
    "geos_proj_str": "+proj=geos +a=6378137.0 +b=6356752.3 +lon_0=0.0 +h=35785863.0 +sweep=y",
    "cloud_categories": {
        "Other": [1, 2, 3, 4, 255], 
        "Warm": [5, 6, 7, 8, 9, 10],  # Fog + Thick clouds
        "Cold": [11, 12, 13, 14, 15],  # Thin clouds
    }
}

def _haversine_km(lat0, lon0, lat1, lon1):
    """Calculate great-circle distance in kilometers."""
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    _, _, dist_m = geod.inv(lon0, lat0, lon1, lat1)
    return dist_m / 1000

def analyze_day(day_path):
    """Analyze cloud types for a specific day."""
    print(f"Analyzing {day_path}")
    
    # Get all CT files for the day
    ct_files = sorted(day_path.glob("S_NWC_CT_MSG*_globeM-VISIR_*.nc"))
    print(f"Found {len(ct_files)} CT files")
    
    if not ct_files:
        print("No CT files found!")
        return
    
    # Analyze first file to get grid setup
    with xr.open_dataset(ct_files[0]) as ds:
        print(f"Dataset shape: {ds.ct.shape}")
        print(f"Dataset variables: {list(ds.variables.keys())}")
        
        # Setup grid
        proj_str = ds.attrs.get('gdal_projection', CONFIG["geos_proj_str"])
        transformer = Transformer.from_crs(proj_str, "epsg:4326", always_xy=True)
        nx_grid, ny_grid = np.meshgrid(ds.nx.values, ds.ny.values)
        lon2d, lat2d = transformer.transform(nx_grid, ny_grid)
        
        # Find AOI
        center_lat, center_lon = CONFIG["center_coords"]["lat"], CONFIG["center_coords"]["lon"]
        dist_sq = (lat2d - center_lat)**2 + (lon2d - center_lon)**2
        y0, x0 = np.unravel_index(np.nanargmin(dist_sq), dist_sq.shape)
        
        # Calculate resolution and AOI slice
        res_y = _haversine_km(lat2d[y0 - 1, x0], lon2d[y0 - 1, x0], lat2d[y0 + 1, x0], lon2d[y0 + 1, x0]) / 2
        res_x = _haversine_km(lat2d[y0, x0 - 1], lon2d[y0, x0 - 1], lat2d[y0, x0 + 1], lon2d[y0, x0 + 1]) / 2
        avg_pixel_res_km = (res_x + res_y) / 2
        
        n_pixels = int(np.ceil(CONFIG["aoi_radius_km"] / avg_pixel_res_km))
        y_start, y_end = max(0, y0 - n_pixels), min(lat2d.shape[0], y0 + n_pixels + 1)
        x_start, x_end = max(0, x0 - n_pixels), min(lat2d.shape[1], x0 + n_pixels + 1)
        aoi_slice = (slice(y_start, y_end), slice(x_start, x_end))
        
        print(f"AOI slice: {aoi_slice}")
        print(f"AOI size: {y_end-y_start} x {x_end-x_start} pixels")
    
    # Analyze daytime hours (6-18 UTC)
    daytime_results = []
    
    for file_path in ct_files:
        try:
            ts = datetime.strptime(file_path.stem.split('_')[-1], "%Y%m%dT%H%M%SZ")
            
            # Only process daytime hours
            if 6 <= ts.hour < 18:
                with xr.open_dataset(file_path) as ds:
                    ct_aoi = ds.ct.isel(ny=aoi_slice[0], nx=aoi_slice[1]).values
                    
                    # Calculate percentages
                    flat = ct_aoi[~np.isnan(ct_aoi)].astype(int)
                    total_pixels = flat.size if flat.size > 0 else 1
                    unique, counts = np.unique(flat, return_counts=True)
                    percentages = {k: v * 100 / total_pixels for k, v in zip(unique, counts)}
                    
                    # Calculate warm and cold percentages
                    warm_pct = sum(percentages.get(i, 0) for i in CONFIG["cloud_categories"]["Warm"])
                    cold_pct = sum(percentages.get(i, 0) for i in CONFIG["cloud_categories"]["Cold"])
                    
                    daytime_results.append({
                        'timestamp': ts,
                        'warm_pct': warm_pct,
                        'cold_pct': cold_pct,
                        'total_pixels': total_pixels,
                        'percentages': percentages
                    })
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    if not daytime_results:
        print("No daytime data found!")
        return
    
    # Create DataFrame and analyze
    df = pd.DataFrame(daytime_results)
    print(f"\nDaytime data summary ({len(df)} files):")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Mean warm cloud %: {df['warm_pct'].mean():.2f}")
    print(f"Std warm cloud %: {df['warm_pct'].std():.2f}")
    print(f"Mean cold cloud %: {df['cold_pct'].mean():.2f}")
    print(f"Min warm cloud %: {df['warm_pct'].min():.2f}")
    print(f"Max warm cloud %: {df['warm_pct'].max():.2f}")
    
    # Show hourly breakdown
    print("\nHourly breakdown:")
    hourly_stats = df.groupby(df['timestamp'].dt.hour).agg({
        'warm_pct': ['mean', 'std'],
        'cold_pct': 'mean'
    }).round(2)
    print(hourly_stats)
    
    # Show some sample percentages
    print("\nSample cloud type percentages (first 5 files):")
    for i, row in df.head().iterrows():
        print(f"{row['timestamp']}: Warm={row['warm_pct']:.1f}%, Cold={row['cold_pct']:.1f}%")
        # Show top cloud types
        top_types = sorted(row['percentages'].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top cloud types: {top_types}")

if __name__ == "__main__":
    day_path = Path("/mnt/m0/y-m.saint-drenan/data/NWCSAF_CloudType/2024/2024_09_17/")
    analyze_day(day_path) 