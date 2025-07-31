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
MONTHS=("2024-04" "2024-05" "2024-06" "2024-07" "2024-08" "2024-09")

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
    local output_dir="output_${month}"
    local ranking_csv="$output_dir/les_suitability_ranking_${month}.csv"
    
    # Check if ranking CSV already exists
    if [ -f "$ranking_csv" ]; then
        print_status "Ranking CSV already exists for $month: $ranking_csv"
        return 0
    fi
    
    # Also check progress file as backup
    if grep -q "^$month:completed:" "$PROGRESS_FILE" 2>/dev/null; then
        return 0
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
    local output_dir="output_${month}"
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
        
        if python les-screening-monthly.py --month "$month" --data-root "$DATA_ROOT" --out-root "$output_dir"; then
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
    local ranking_csv="$output_dir/les_suitability_ranking_${month}.csv"
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
        
        if python generate_top_day_visualizations.py; then
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
        local output_dir="output_${month}"
        local ranking_csv="$output_dir/les_suitability_ranking_${month}.csv"
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
    print_status "Processing months: ${MONTHS[*]}"
    print_status "Progress will be saved to: $PROGRESS_FILE"
    print_status "Log will be saved to: $LOG_FILE"
    
    # Initialize progress file if it doesn't exist
    touch "$PROGRESS_FILE"
    
    # Show current progress
    show_progress
    
    # Show processing plan
    show_processing_plan
    
    # Activate WUR environment
    if ! activate_wur; then
        print_error "Failed to activate WUR environment. Exiting."
        exit 1
    fi
    
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