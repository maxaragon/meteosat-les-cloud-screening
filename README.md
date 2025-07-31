# METEOSAT-LES: Multi-Month Cloud Screening for Large Eddy Simulations

This repository contains a comprehensive cloud analysis pipeline designed to screen Meteosat satellite data for identifying days suitable for **Large Eddy Simulations (LES)** across multiple months. The system analyzes NWCSAF Cloud Type data to find periods with optimal cumuliform cloud conditions while avoiding high-level cloud contamination.

## 🎯 Purpose

Identify days with optimal conditions for LES case studies at the SIRTA observatory (48.72°N, 2.21°E) by screening for:
- ✅ **High cumuliform cloud coverage** (flags 5-7) ≥ 25% daily average
- ❌ **Low high-level cloud contamination** (flags 8-10) < 1% hourly maximum
- 🎯 **Optimal atmospheric conditions** for LES boundary layer studies
- 📅 **Multi-month analysis** for comprehensive seasonal coverage

## 📁 Repository Structure

```
METEOSAT-LES/
├── 📄 run_multi_month_analysis.sh      # 🚀 Main multi-month analysis script
├── 📄 les-screening-monthly.py         # 📊 Monthly LES cloud screening pipeline
├── 📄 generate_top_day_visualizations.py # 🎬 Top day visualization generator
├── 📄 debug_day.py                     # 🐛 Debug tool for individual days
├── 📄 README.md                        # 📖 This documentation
├── 📄 requirements.txt                 # 📦 Python dependencies
├── 📄 multi_month_progress.txt         # 📈 Progress tracking file
├── 📄 multi_month_analysis_robust.log  # 📝 Detailed execution log
├── 📄 nohup_multi_month.out            # 🔄 Background execution output
└── 📁 output_YYYY-MM/                  # 📊 Generated analysis results per month
    ├── 📄 les_suitability_ranking_YYYY-MM.csv  # 🏆 LES suitability ranking
    ├── 📁 top_days_png/                # 📸 Top day visualizations
    └── 📁 top_days_gif/                # 🎬 Top day GIF animations
```

## 🚀 Quick Start

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Make script executable
chmod +x run_multi_month_analysis.sh
```

### Run Multi-Month Analysis
```bash
# Run analysis for all months (April-September 2024)
./run_multi_month_analysis.sh

# Run in background with nohup
nohup ./run_multi_month_analysis.sh > nohup_multi_month.out 2>&1 &

# Monitor progress
tail -f nohup_multi_month.out
```

### Run Individual Month Analysis
```bash
# Analyze specific month
python les-screening-monthly.py --month 2024-05 --data-root /path/to/data --out-root output_2024-05

# Generate visualizations for top days
python generate_top_day_visualizations.py
```

## 🔧 Multi-Month Analysis Features

### **Robust Processing System:**
- ✅ **Smart Skipping**: Automatically skips months with existing CSV files
- 🔄 **Retry Logic**: Up to 3 attempts for failed processing steps
- 📊 **Progress Tracking**: Real-time progress monitoring and logging
- 🛡️ **Error Recovery**: Continues processing even if individual months fail
- ⏰ **Background Execution**: Runs with nohup for long-term processing

### **Processing Months:**
- **2024-04** (April) - 30 days
- **2024-05** (May) - 31 days  
- **2024-06** (June) - 30 days
- **2024-07** (July) - 31 days
- **2024-08** (August) - 31 days
- **2024-09** (September) - 30 days

### **Output Structure:**
Each month generates:
- `les_suitability_ranking_YYYY-MM.csv` - Complete daily ranking
- `top_days_png/` - PNG visualizations for top 3 days
- `top_days_gif/` - GIF animations for top 3 days

## 📊 Cloud Type Classification

The system analyzes NWCSAF Cloud Type flags:

| Flag | Cloud Type | Color | LES Relevance |
|------|------------|-------|---------------|
| 0-4  | Other (No data, sea, snow/ice) | White | ❌ Not relevant |
| 5    | Very-low cumuliform | Purple | ✅ LES suitable |
| 6    | Low cumuliform | Blue | ✅ LES suitable |
| 7    | Mid-level cumuliform | Green | ✅ LES suitable |
| 8    | High-level cirrus | Red | ❌ LES unsuitable |
| 9    | High-level deep convective | Orange | ❌ LES unsuitable |
| 10   | High-level other | Magenta | ❌ LES unsuitable |

## 🎯 LES Suitability Criteria

### **Primary Criteria:**
A day is considered **LES-suitable** if:
- **Cumuliform clouds** (flags 5-7) ≥ 25% daily average
- **High-level clouds** (flags 8-10) < 1% hourly maximum
- **Optimal temporal distribution** throughout the day

### **Scoring System:**
- **Daily Score** = Cumuliform % - High-level %
- **Ranking** based on highest daily scores
- **Top 3 days** selected for detailed visualization

## 📈 Analysis Results

### **Current Status:**
- ✅ **April 2024**: Completed (CSV exists)
- 🔄 **May 2024**: Currently processing
- ⏳ **June 2024**: Pending
- ⏳ **July 2024**: Pending
- ⏳ **August 2024**: Pending
- ✅ **September 2024**: Completed (CSV exists)

### **Processing Statistics:**
- **Files per day**: 96 (15-minute intervals)
- **Processing speed**: ~5-6 files/second
- **Estimated time per month**: 15-20 minutes
- **Total estimated time**: 1-1.5 hours for remaining months

## 🔧 Technical Features

### Core Capabilities:
- **Native coordinate handling** using 2D lat/lon arrays from NetCDF files
- **Perfect 40×40 km AOI** calculation using haversine distance
- **Memory-efficient processing** with xarray + dask chunking
- **Multiprocessing support** for parallel file processing
- **Custom colormap** for cloud type visualization
- **GIF animations** for daily cloud evolution

### Data Processing Pipeline:
1. **File Discovery** - Glob pattern matching for NetCDF files
2. **Coordinate Setup** - 2D lat/lon grid extraction and bounding box calculation
3. **Data Extraction** - Region-specific data slicing (France + SIRTA AOI)
4. **Cloud Type Mapping** - Remap flags for visualization and analysis
5. **Statistics Calculation** - Percentage computation for each cloud type
6. **Visualization Generation** - Dual-panel PNGs with cartopy mapping
7. **GIF Creation** - Daily animation compilation using PIL
8. **Ranking Analysis** - LES suitability scoring and daily ranking

## 📦 Dependencies

```python
xarray>=2023.1.0      # NetCDF data handling
numpy>=1.24.0         # Numerical computing
matplotlib>=3.7.0     # Visualization
pandas>=2.0.0         # Data analysis
cartopy>=0.21.0       # Geographic plotting
pyproj>=3.5.0         # Coordinate transformations
netCDF4>=1.6.0        # NetCDF file I/O
pillow>=9.5.0         # GIF creation
imageio>=2.31.0       # GIF processing
tqdm>=4.65.0          # Progress bars
```

## 🎓 Scientific Applications

### LES Case Study Planning:
- **Cloud Type Screening** - Identify optimal cumuliform conditions
- **Temporal Analysis** - 15-minute resolution cloud evolution
- **Spatial Analysis** - 40×40 km SIRTA observatory focus
- **Quality Control** - High-level cloud contamination avoidance
- **Seasonal Analysis** - Multi-month coverage for comprehensive studies

### Research Value:
- **Atmospheric Science** - Cloud microphysics and dynamics
- **Remote Sensing** - Satellite data validation and analysis
- **Climate Research** - Cloud type climatology and trends
- **Numerical Modeling** - LES boundary condition optimization

## 📄 Output Files

### Data Files:
- **`les_suitability_ranking_YYYY-MM.csv`** - Complete daily ranking with suitability scores
- **`multi_month_progress.txt`** - Progress tracking for all months
- **`multi_month_analysis_robust.log`** - Detailed execution log

### Visualizations:
- **PNG Files** - Dual-panel maps showing France overview + SIRTA AOI zoom
- **GIF Animations** - Daily cloud evolution videos for top 3 days

## 🔍 Monitoring and Debugging

### Progress Monitoring:
```bash
# Check if script is running
ps aux | grep run_multi_month_analysis

# View latest output
tail -f nohup_multi_month.out

# Check progress file
cat multi_month_progress.txt

# Check detailed log
tail -f multi_month_analysis_robust.log
```

### Debug Tools:
```bash
# Debug individual day
python debug_day.py --date 2024-05-15

# Check specific month output
ls -la output_2024-05/
```

## 🤝 Contributing

This pipeline is designed for atmospheric science research. For questions or improvements, please refer to the scientific context and ensure compatibility with NWCSAF Cloud Type data format.

## 📚 References

- **NWCSAF Cloud Type Product**: https://nwcsaf.smhi.se/ProductDescriptionCloudType.php
- **SIRTA Observatory**: https://sirta.ipsl.polytechnique.fr/
- **Large Eddy Simulation**: Atmospheric boundary layer modeling technique

---

**Author**: Max Aragon Cerecedes  
**Purpose**: Multi-month LES case study planning and cloud condition screening  
**Data Source**: Meteosat NWCSAF Cloud Type Product  
**Last Updated**: August 2025 