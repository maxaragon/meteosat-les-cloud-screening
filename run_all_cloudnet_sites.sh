#!/bin/bash

# Run Multi-Month Analysis for All CloudNet Sites
# ==============================================

# All CloudNet sites
SITES=(
    "PALAISEAU"      # SIRTA, France
    "MUNICH"         # LMU Munich, Germany  
    "CABAUW"         # Cabauw, Netherlands
    "LINDENBERG"     # Lindenberg, Germany
    "BUCHAREST"      # Bucharest, Romania
    "CHILBOLTON"     # Chilbolton, UK
    "CLUJ"           # Cluj-Napoca, Romania
    "GALATI"         # Galați, Romania
    "GRANADA"        # Granada, Spain
    "HYYTIALA"       # Hyytiälä, Finland
    "JUELICH"        # Jülich, Germany
    "KENTTAROVA"     # Kenttärova, Finland
    "LAMPEDUSA"      # Lampedusa, Italy
    "LEIPZIG"        # Leipzig, Germany
    "LEIPZIG-LIM"    # Leipzig LIM, Germany
    "LIMASSOL"       # Limassol, Cyprus
    "MACE-HEAD"      # Mace Head, Ireland
    "MAIDO"          # Maïdo Observatory, Réunion
    "MINDELO"        # Mindelo, Cabo Verde
    "NEUMAYER"       # Neumayer Station, Antarctica
    "NORUNDA"        # Norunda, Sweden
    "NY-ALESUND"     # Ny-Ålesund, Norway
    "PAYERNE"        # Payerne, Switzerland
    "POTENZA"        # Potenza, Italy
    "RZECIN"         # Rzecin, Poland
    "SCHNEEFERNERHAUS" # Schneefernerhaus, Germany
    "WARSAW"         # Warsaw, Poland
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run analysis for a single site
run_site_analysis() {
    local site=$1
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    print_status "=========================================="
    print_status "Processing site: $site"
    print_status "Started at: $start_time"
    print_status "=========================================="
    
    if ./run_multi_month_analysis.sh "$site"; then
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        print_success "Completed $site at $end_time"
        return 0
    else
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        print_error "Failed $site at $end_time"
        return 1
    fi
}

# Function to show summary
show_summary() {
    print_status "=========================================="
    print_status "SUMMARY"
    print_status "=========================================="
    
    local total_sites=${#SITES[@]}
    local completed=0
    local failed=0
    
    for site in "${SITES[@]}"; do
        if [ -d "$site" ]; then
            local output_dirs=$(find "$site" -name "output_*" -type d 2>/dev/null | wc -l)
            if [ "$output_dirs" -gt 0 ]; then
                print_success "✅ $site - Completed ($output_dirs output directories)"
                ((completed++))
            else
                print_warning "⚠️  $site - Directory exists but no output found"
                ((failed++))
            fi
        else
            print_error "❌ $site - Not processed"
            ((failed++))
        fi
    done
    
    print_status "=========================================="
    print_status "Total sites: $total_sites"
    print_success "Completed: $completed"
    print_error "Failed: $failed"
    print_status "=========================================="
}

# Main execution
main() {
    print_status "Starting CloudNet Multi-Site Analysis"
    print_status "Total sites to process: ${#SITES[@]}"
    print_status "Sites: ${SITES[*]}"
    print_status "=========================================="
    
    # Check if the main script exists
    if [ ! -f "./run_multi_month_analysis.sh" ]; then
        print_error "run_multi_month_analysis.sh not found!"
        exit 1
    fi
    
    # Make sure it's executable
    chmod +x ./run_multi_month_analysis.sh
    
    # Process each site
    local success_count=0
    local total_count=${#SITES[@]}
    
    for site in "${SITES[@]}"; do
        if run_site_analysis "$site"; then
            ((success_count++))
        fi
        
        # Add a small delay between sites to avoid overwhelming the system
        sleep 5
    done
    
    print_status "=========================================="
    print_status "ALL SITES PROCESSED"
    print_status "=========================================="
    print_success "Successfully processed: $success_count/$total_count sites"
    
    # Show final summary
    show_summary
}

# Handle script interruption
cleanup_on_exit() {
    print_warning "Script interrupted. Progress saved."
    show_summary
    exit 1
}

# Setup signal handlers
trap cleanup_on_exit SIGINT SIGTERM

# Run main function
main "$@" 