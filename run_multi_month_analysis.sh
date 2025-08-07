#!/bin/bash

# Multi-Month LES Analysis and Visualization Script
# ===============================================
# This script processes multiple months of cloud data and generates
# rankings and visualizations for the top 3 LES candidate days per month.
#
# Author: Max Aragon Cerecedes
# Date: July 31, 2025
# Version: 2.0 (Robust)

# Remove set -e to prevent premature exit on errors
# set -e  # Exit on any error

# Configuration
SCRIPT_DIR="/mnt/m1/max.aragon_cerecedes/METEOSAT-LES"
DATA_ROOT="/mnt/m0/y-m.saint-drenan/data/NWCSAF_CloudType/2024"

# Months to process - EDIT THIS ARRAY TO CHANGE PROCESSING MONTHS
MONTHS=("2024-04" "2024-05" "2024-06" "2024-07" "2024-08" "2024-09")

# Function to show help
show_help() {
    echo "Multi-Month LES Analysis and Visualization Script"
    echo "================================================"
    echo ""
    echo "Usage: $0 SITE"
    echo ""
    echo "SITE options:"
    echo "  PALAISEAU  - SIRTA Observatory, France (48.717°N, 2.209°E)"
    echo "  MUNICH     - LMU Munich, Germany (48.148°N, 11.573°E)"
    echo "  CABAUW     - Cabauw Observatory, Netherlands (51.968°N, 4.927°E)"
    echo "  LINDENBERG - Lindenberg Observatory, Germany (52.208°N, 14.118°E)"
    echo ""
    echo "CloudNet Sites:"
    echo "  BUCHAREST  - Bucharest, Romania (44.344°N, 26.012°E)"
    echo "  CHILBOLTON - Chilbolton, UK (51.144°N, 1.439°W)"
    echo "  CLUJ       - Cluj-Napoca, Romania (46.768°N, 23.540°E)"
    echo "  GALATI     - Galați, Romania (45.435°N, 28.037°E)"
    echo "  GRANADA    - Granada, Spain (37.164°N, 3.605°W)"
    echo "  HYYTIALA   - Hyytiälä, Finland (61.844°N, 24.287°E)"
    echo "  JUELICH    - Jülich, Germany (50.908°N, 6.413°E)"
    echo "  KENTTAROVA - Kenttärova, Finland (67.987°N, 24.243°E)"
    echo "  LAMPEDUSA  - Lampedusa, Italy (35.520°N, 12.630°E)"
    echo "  LEIPZIG    - Leipzig, Germany (51.353°N, 12.435°E)"
    echo "  LEIPZIG-LIM- Leipzig LIM, Germany (51.333°N, 12.389°E)"
    echo "  LIMASSOL   - Limassol, Cyprus (34.677°N, 33.038°E)"
    echo "  MACE-HEAD  - Mace Head, Ireland (53.326°N, 9.900°W)"
    echo "  MAIDO      - Maïdo Observatory, Réunion (-21.079°S, 55.383°E)"
    echo "  MINDELO    - Mindelo, Cabo Verde (16.878°N, 24.995°W)"
    echo "  NEUMAYER   - Neumayer Station, Antarctica (-70.660°S, 8.284°W)"
    echo "  NORUNDA    - Norunda, Sweden (60.086°N, 17.479°E)"
    echo "  NY-ALESUND - Ny-Ålesund, Norway (78.923°N, 11.922°E)"
    echo "  PAYERNE    - Payerne, Switzerland (46.813°N, 6.944°E)"
    echo "  POTENZA    - Potenza, Italy (40.601°N, 15.724°E)"
    echo "  RZECIN     - Rzecin, Poland (52.758°N, 16.310°E)"
    echo "  SCHNEEFERNERHAUS - Schneefernerhaus, Germany (47.417°N, 10.977°E)"
    echo "  WARSAW     - Warsaw, Poland (52.210°N, 20.980°E)"
    echo ""
    echo "Examples:"
    echo "  $0 PALAISEAU    # Run analysis for SIRTA site"
    echo "  $0 MUNICH       # Run analysis for LMU Munich site"
    echo "  $0 GRANADA      # Run analysis for Granada site"
    echo ""
    echo "The script will:"
    echo "  1. Process months: 2024-04 to 2024-09"
    echo "  2. Generate LES suitability rankings"
    echo "  3. Create visualizations for top 3 days per month"
    echo "  4. Save results in SITE/output_YYYY-MM/ directories"
    echo ""
}

# Parse command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "help" ]; then
    show_help
    exit 0
fi

# Check if site argument is provided
if [ -z "$1" ]; then
    echo "Error: SITE argument is required"
    echo ""
    show_help
    exit 1
fi

# Get site parameter from command line argument
SITE="$1"

# Validate site parameter
case "$SITE" in
    # Original sites
    "PALAISEAU"|"MUNICH"|"CABAUW"|"LINDENBERG")
        # Valid site
        ;;
    # CloudNet sites
    "BUCHAREST"|"CHILBOLTON"|"CLUJ"|"GALATI"|"GRANADA"|"HYYTIALA"|"JUELICH"|"KENTTAROVA"|"LAMPEDUSA"|"LEIPZIG"|"LEIPZIG-LIM"|"LIMASSOL"|"MACE-HEAD"|"MAIDO"|"MINDELO"|"NEUMAYER"|"NORUNDA"|"NY-ALESUND"|"PAYERNE"|"POTENZA"|"RZECIN"|"SCHNEEFERNERHAUS"|"WARSAW")
        # Valid site
        ;;
    *)
        echo "Error: Invalid site '$SITE'"
        echo ""
        show_help
        exit 1
        ;;
esac

# Progress tracking file
PROGRESS_FILE="multi_month_progress.txt"
LOG_FILE="multi_month_analysis_robust.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to log progress
log_progress() {
    local month=$1
    local status=$2
    echo "$month:$status:$(date '+%Y-%m-%d %H:%M:%S')" >> "$PROGRESS_FILE"
}

# Function to check if month was already completed
is_month_completed() {
    local month=$1
    local output_dir="${SITE}/output_${month}"
    local ranking_csv="$output_dir/les_suitability_ranking_${month}.csv"
    
    # Check if ranking CSV already exists and is valid (not empty)
    if [ -f "$ranking_csv" ] && [ -s "$ranking_csv" ]; then
        print_status "Ranking CSV already exists for $month: $ranking_csv"
        return 0
    fi
    
    # Check if output directory exists but CSV is missing or empty
    if [ -d "$output_dir" ] && [ ! -f "$ranking_csv" ]; then
        print_warning "Output directory exists for $month but no valid CSV found - will reprocess"
        return 1
    fi
    
    # Check progress file as backup
    if grep -q "^$month:completed:" "$PROGRESS_FILE" 2>/dev/null; then
        # Double-check that the CSV actually exists
        if [ -f "$ranking_csv" ] && [ -s "$ranking_csv" ]; then
            return 0
        else
            print_warning "Progress file shows $month as completed but CSV is missing - will reprocess"
            return 1
        fi
    else
        return 1
    fi
}

# Function to activate WUR environment
activate_wur() {
    print_status "Activating WUR environment..."
    if source /home/MINES/maragoncerecedes/virtualenvs/WUR/bin/activate; then
        print_success "WUR environment activated"
        return 0
    else
        print_error "Failed to activate WUR environment"
        return 1
    fi
}

# Get Python path from WUR environment
WUR_PYTHON="/home/MINES/maragoncerecedes/virtualenvs/WUR/bin/python"

# Function to check if data directory exists
check_data_directory() {
    local month=$1
    local month_pattern="${month//-/_}"
    
    # Check if there are any daily directories for this month
    local day_count=$(find "$DATA_ROOT" -maxdepth 1 -type d -name "${month_pattern}_*" 2>/dev/null | wc -l)
    if [ "$day_count" -eq 0 ]; then
        print_warning "No daily data found for month: $month"
        return 1
    fi
    
    print_success "Found $day_count daily directories for $month"
    return 0
}

# Function to process a single month with comprehensive error handling
process_month() {
    local month=$1
    local output_dir="${SITE}/output_${month}"
    local max_retries=3
    local retry_count=0
    
    print_status "Processing month: $month"
    
    # Check if already completed
    if is_month_completed "$month"; then
        print_success "Month $month already completed, skipping..."
        log_progress "$month" "completed"
        return 0
    fi
    
    # Check if data exists
    if ! check_data_directory "$month"; then
        print_warning "Skipping $month - no data available"
        log_progress "$month" "skipped_no_data"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run LES analysis with retry logic
    while [ $retry_count -lt $max_retries ]; do
        print_status "Running LES analysis for $month (attempt $((retry_count + 1))/$max_retries)..."
        cd "$SCRIPT_DIR"
        
        # Get coordinates based on site
        case "$SITE" in
            # Original sites
            "PALAISEAU")
                lat="48.717"
                lon="2.209"
                ;;
            "MUNICH")
                lat="48.148"
                lon="11.573"
                ;;
            "CABAUW")
                lat="51.968"
                lon="4.927"
                ;;
            "LINDENBERG")
                lat="52.208"
                lon="14.118"
                ;;
            # CloudNet sites
            "BUCHAREST")
                lat="44.344"
                lon="26.012"
                ;;
            "CHILBOLTON")
                lat="51.144"
                lon="-1.439"
                ;;
            "CLUJ")
                lat="46.768"
                lon="23.540"
                ;;
            "GALATI")
                lat="45.435"
                lon="28.037"
                ;;
            "GRANADA")
                lat="37.164"
                lon="-3.605"
                ;;
            "HYYTIALA")
                lat="61.844"
                lon="24.287"
                ;;
            "JUELICH")
                lat="50.908"
                lon="6.413"
                ;;
            "KENTTAROVA")
                lat="67.987"
                lon="24.243"
                ;;
            "LAMPEDUSA")
                lat="35.520"
                lon="12.630"
                ;;
            "LEIPZIG")
                lat="51.353"
                lon="12.435"
                ;;
            "LEIPZIG-LIM")
                lat="51.333"
                lon="12.389"
                ;;
            "LIMASSOL")
                lat="34.677"
                lon="33.038"
                ;;
            "MACE-HEAD")
                lat="53.326"
                lon="-9.900"
                ;;
            "MAIDO")
                lat="-21.079"
                lon="55.383"
                ;;
            "MINDELO")
                lat="16.878"
                lon="-24.995"
                ;;
            "NEUMAYER")
                lat="-70.660"
                lon="-8.284"
                ;;
            "NORUNDA")
                lat="60.086"
                lon="17.479"
                ;;
            "NY-ALESUND")
                lat="78.923"
                lon="11.922"
                ;;
            "PAYERNE")
                lat="46.813"
                lon="6.944"
                ;;
            "POTENZA")
                lat="40.601"
                lon="15.724"
                ;;
            "RZECIN")
                lat="52.758"
                lon="16.310"
                ;;
            "SCHNEEFERNERHAUS")
                lat="47.417"
                lon="10.977"
                ;;
            "WARSAW")
                lat="52.210"
                lon="20.980"
                ;;
            *)
                echo "Error: Unknown site '$SITE'"
                exit 1
                ;;
        esac
        
        if "$WUR_PYTHON" les-screening-monthly.py --month "$month" --data-root "$DATA_ROOT" --out-root "$output_dir" --lat "$lat" --lon "$lon" --site "$SITE"; then
            print_success "LES analysis completed for $month"
            break
        else
            ((retry_count++))
            print_error "LES analysis failed for $month (attempt $retry_count/$max_retries)"
            if [ $retry_count -lt $max_retries ]; then
                print_status "Retrying in 30 seconds..."
                sleep 30
            else
                print_error "LES analysis failed for $month after $max_retries attempts"
                log_progress "$month" "failed_les_analysis"
                return 1
            fi
        fi
    done
    
    # Check if ranking CSV was created
    local ranking_csv="$output_dir/les_suitability_ranking_${month}_${SITE}.csv"
    if [ ! -f "$ranking_csv" ]; then
        print_error "Ranking CSV not found: $ranking_csv"
        log_progress "$month" "failed_no_ranking_csv"
        return 1
    fi
    
    print_success "Ranking CSV created: $ranking_csv"
    
    # Generate visualizations for top 3 days with retry logic
    retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        print_status "Generating visualizations for top 3 days in $month (attempt $((retry_count + 1))/$max_retries)..."
        
        if SITE="$SITE" MONTH="$month" "$WUR_PYTHON" generate_top_day_visualizations.py; then
            print_success "Visualizations completed for $month"
            break
        else
            ((retry_count++))
            print_warning "Visualization generation failed for $month (attempt $retry_count/$max_retries)"
            if [ $retry_count -lt $max_retries ]; then
                print_status "Retrying in 30 seconds..."
                sleep 30
            else
                print_warning "Visualization generation failed for $month after $max_retries attempts, but continuing..."
                # Don't fail the entire month if visualizations fail
            fi
        fi
    done
    
    # Mark as completed
    log_progress "$month" "completed"
    return 0
}

# Function to clean up old PNG directories
cleanup_old_pngs() {
    print_status "Cleaning up old PNG directories..."
    
    # Remove old PNG directories from previous runs
    find . -maxdepth 2 -type d -name "png" -exec rm -rf {} + 2>/dev/null || true
    find . -maxdepth 2 -type d -name "top_days_png" -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show progress
show_progress() {
    print_status "Current progress:"
    if [ -f "$PROGRESS_FILE" ]; then
        while IFS=: read -r month status timestamp; do
            case $status in
                "completed")
                    print_success "✅ $month - Completed at $timestamp"
                    ;;
                "failed_les_analysis")
                    print_error "❌ $month - LES analysis failed at $timestamp"
                    ;;
                "failed_no_ranking_csv")
                    print_error "❌ $month - No ranking CSV at $timestamp"
                    ;;
                "skipped_no_data")
                    print_warning "⚠️  $month - Skipped (no data) at $timestamp"
                    ;;
                *)
                    print_warning "❓ $month - Unknown status: $status at $timestamp"
                    ;;
            esac
        done < "$PROGRESS_FILE"
    else
        print_status "No progress file found - starting fresh"
    fi
}

# Function to show processing plan
show_processing_plan() {
    print_status "Processing plan:"
    local to_process=()
    local to_skip=()
    
    for month in "${MONTHS[@]}"; do
        if is_month_completed "$month"; then
            to_skip+=("$month")
        else
            to_process+=("$month")
        fi
    done
    
    if [ ${#to_skip[@]} -gt 0 ]; then
        print_success "Will skip (already completed): ${to_skip[*]}"
    fi
    
    if [ ${#to_process[@]} -gt 0 ]; then
        print_status "Will process: ${to_process[*]}"
    else
        print_success "All months already completed!"
    fi
    
    echo ""
}

# Function to show summary
show_summary() {
    print_status "Generating summary..."
    
    echo ""
    echo "=========================================="
    echo "           ANALYSIS SUMMARY"
    echo "=========================================="
    
    local completed_count=0
    local failed_count=0
    local skipped_count=0
    
    for month in "${MONTHS[@]}"; do
        local output_dir="${SITE}/output_${month}"
        local ranking_csv="$output_dir/les_suitability_ranking_${month}_${SITE}.csv"
        local gif_dir="$output_dir/top_days_gif"
        
        echo ""
        echo "Month: $month"
        echo "  Output directory: $output_dir"
        
        if [ -f "$ranking_csv" ]; then
            echo "  ✅ Ranking CSV: $(basename "$ranking_csv")"
            ((completed_count++))
            
            # Show top 3 days
            if command -v head >/dev/null 2>&1; then
                echo "  Top 3 days:"
                head -4 "$ranking_csv" | tail -3 | while IFS=',' read -r date score rest; do
                    echo "    - $date (Score: $score)"
                done
            fi
        else
            echo "  ❌ Ranking CSV: Not found"
            ((failed_count++))
        fi
        
        if [ -d "$gif_dir" ]; then
            local gif_count=$(find "$gif_dir" -name "*.gif" | wc -l)
            echo "  ✅ GIFs: $gif_count files"
        else
            echo "  ❌ GIFs: Not found"
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "Summary: $completed_count completed, $failed_count failed, $skipped_count skipped"
    echo "=========================================="
}

# Function to handle script interruption
cleanup_on_exit() {
    print_warning "Script interrupted. Progress saved."
    exit 1
}

# Main execution
main() {
    # Setup signal handlers for graceful interruption
    trap cleanup_on_exit SIGINT SIGTERM
    
    print_status "Starting robust multi-month LES analysis and visualization"
    print_status "Processing site: $SITE"
    print_status "Processing months: ${MONTHS[*]}"
    print_status "Progress will be saved to: $PROGRESS_FILE"
    print_status "Log will be saved to: $LOG_FILE"
    
    # Initialize progress file if it doesn't exist
    touch "$PROGRESS_FILE"
    
    # Show current progress
    show_progress
    
    # Show processing plan
    show_processing_plan
    
    # Check if WUR Python exists
    if [ ! -f "$WUR_PYTHON" ]; then
        print_error "WUR Python not found at: $WUR_PYTHON"
        exit 1
    fi
    print_status "Using WUR Python: $WUR_PYTHON"
    
    # Clean up old files
    cleanup_old_pngs
    
    # Process each month
    local success_count=0
    local total_count=${#MONTHS[@]}
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    print_status "Starting processing at: $start_time"
    
    for month in "${MONTHS[@]}"; do
        echo ""
        print_status "=========================================="
        print_status "Processing: $month"
        print_status "=========================================="
        
        local month_start_time=$(date '+%Y-%m-%d %H:%M:%S')
        print_status "Started processing $month at: $month_start_time"
        
        if process_month "$month"; then
            ((success_count++))
            local month_end_time=$(date '+%Y-%m-%d %H:%M:%S')
            print_success "Completed processing $month at: $month_end_time"
        else
            local month_end_time=$(date '+%Y-%m-%d %H:%M:%S')
            print_error "Failed processing $month at: $month_end_time"
        fi
        
        # Show progress after each month
        print_status "Progress: $success_count/$total_count months completed"
    done
    
    # Show final summary
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    print_status "=========================================="
    print_status "ANALYSIS COMPLETE"
    print_status "=========================================="
    print_status "Started: $start_time"
    print_status "Finished: $end_time"
    print_success "Successfully processed $success_count out of $total_count months"
    
    show_summary
    
    print_success "Multi-month analysis completed!"
}

# Run main function
main "$@" 