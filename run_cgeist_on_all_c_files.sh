#!/bin/bash

# Script to run cgeist on all .c files in the Test directory
# and then run polygeist-opt --raise-affine-to-linalg and --linalg-debufferize on generated files
# Based on the provided command format

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <mode>"
    echo ""
    echo "Modes:"
    echo "  0 - Run all phases (cgeist + polygeist-opt + canonicalize + linalg-debufferize)"
    echo "  1 - Run only cgeist phase"
    echo "  2 - Run only polygeist-opt --raise-affine-to-linalg phase"
    echo "  3 - Run only polygeist-opt --canonicalize phase"
    echo "  4 - Run only polygeist-opt --linalg-debufferize phase"
    echo "  5 - Run --affine-parallelize then --raise-affine-to-linalg-pipeline phase"
    echo ""
    echo "Examples:"
    echo "  $0 0    # Run all four phases"
    echo "  $0 1    # Run only cgeist"
    echo "  $0 2    # Run only raise-affine-to-linalg"
    echo "  $0 3    # Run only canonicalize"
    echo "  $0 4    # Run only linalg-debufferize"
    echo "  $0 5    # Run only raise-affine-to-linalg-pipeline (combines parallelize + raise + canonicalize)"
    exit 1
}

# Check if argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Mode argument is required"
    show_usage
fi

# Parse mode argument
MODE="$1"

# Validate mode argument
if [[ ! "$MODE" =~ ^[012345]$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 0, 1, 2, 3, 4, or 5"
    show_usage
fi

# Configuration
TEST_DIR="/home/arjaiswal/Polygeist/tools/cgeist/Test"
OUTPUT_DIR="/home/arjaiswal/Polygeist/cgeist-output"
LINALG_OUTPUT_DIR="/home/arjaiswal/Polygeist/cgeist-linalg-output"
CANONICALIZE_OUTPUT_DIR="/home/arjaiswal/Polygeist/cgeist-canonicalized-output"
debufferizeD_OUTPUT_DIR="/home/arjaiswal/Polygeist/cgeist-debufferized-output"
PIPELINE_OUTPUT_DIR="/home/arjaiswal/Polygeist/cgeist-pipeline-output"
RESOURCE_DIR="/usr/lib/clang/14"  # Default clang resource dir, adjust if needed
LOG_FILE="cgeist_run.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Display mode information
case $MODE in
    0)
        echo -e "${GREEN}Starting four-phase cgeist + polygeist-opt + canonicalize + linalg-debufferize processing${NC}"
        echo -e "${BLUE}Phase 1: Running cgeist on all .c files in ${TEST_DIR}${NC}"
        echo -e "${BLUE}Phase 2: Running polygeist-opt --raise-affine-to-linalg on generated files${NC}"
        echo -e "${BLUE}Phase 3: Running polygeist-opt --canonicalize on linalg files${NC}"
        echo -e "${BLUE}Phase 4: Running polygeist-opt --linalg-debufferize on canonicalized files${NC}"
        ;;
    1)
        echo -e "${GREEN}Starting cgeist processing only${NC}"
        echo -e "${BLUE}Phase 1: Running cgeist on all .c files in ${TEST_DIR}${NC}"
        ;;
    2)
        echo -e "${GREEN}Starting polygeist-opt --raise-affine-to-linalg processing only${NC}"
        echo -e "${BLUE}Phase 2: Running polygeist-opt --raise-affine-to-linalg on existing files${NC}"
        ;;
    3)
        echo -e "${GREEN}Starting canonicalize processing only${NC}"
        echo -e "${BLUE}Phase 3: Running polygeist-opt --canonicalize on existing linalg files${NC}"
        ;;
    4)
        echo -e "${GREEN}Starting linalg-debufferize processing only${NC}"
        echo -e "${BLUE}Phase 4: Running polygeist-opt --linalg-debufferize on existing canonicalized files${NC}"
        ;;
    5)
        echo -e "${GREEN}Starting raise-affine-to-linalg-pipeline processing only${NC}"
        echo -e "${BLUE}Phase 5: Running --affine-parallelize then --raise-affine-to-linalg-pipeline on existing files${NC}"
        ;;
esac

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LINALG_OUTPUT_DIR"
mkdir -p "$CANONICALIZE_OUTPUT_DIR"
mkdir -p "$debufferizeD_OUTPUT_DIR"
mkdir -p "$PIPELINE_OUTPUT_DIR"

# Initialize log file
echo "Script run started at $(date) - Mode: $MODE" > "$LOG_FILE"

# Initialize global counters
total_files=0
success_count=0
error_count=0
total_mlir_files=0
linalg_success_count=0
linalg_error_count=0
total_canonicalize_files=0
canonicalize_success_count=0
canonicalize_error_count=0
total_debufferized_files=0
debufferized_success_count=0
debufferized_error_count=0
total_pipeline_files=0
pipeline_success_count=0
pipeline_error_count=0

# ====== PHASE 1: CGEIST PROCESSING ======
if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 1 ]; then
    echo -e "\n${PURPLE}=== PHASE 1: CGEIST PROCESSING ===${NC}"

    # Find all .c files and count them
    mapfile -t c_files < <(find "$TEST_DIR" -name "*.c" -type f)
    total_files=${#c_files[@]}

    echo -e "${YELLOW}Found $total_files .c files to process${NC}"

    # Counter for progress
    counter=0

    # Process each .c file
    for c_file in "${c_files[@]}"; do
        counter=$((counter + 1))
        
        # Get relative path from TEST_DIR
        rel_path="${c_file#$TEST_DIR/}"
        
        # Create output filename (replace .c with .mlir and path separators with underscores)
        output_name=$(echo "$rel_path" | sed 's|/|_|g' | sed 's|\.c$|-polygeist-intermediate.mlir|')
        output_path="$OUTPUT_DIR/$output_name"
        
        # Create output directory structure if needed
        output_subdir=$(dirname "$output_path")
        mkdir -p "$output_subdir"
        
        echo -e "${YELLOW}[$counter/$total_files] Processing: $rel_path${NC}"
        
        # Determine if this is a CUDA file (check if it's in CUDA directory or contains CUDA-specific code)
        cuda_flags=""
        if [[ "$c_file" == *"/CUDA/"* ]] || [[ "$c_file" == *"cuda"* ]] || grep -q "cuda\|__global__\|__device__" "$c_file" 2>/dev/null; then
            cuda_flags="--cuda-gpu-arch=sm_35"
        fi
        
        # Run cgeist command
        cgeist_cmd="cgeist \"$c_file\" --function=* --resource-dir=$RESOURCE_DIR -I $TEST_DIR/polybench/utilities --raise-scf-to-affine $cuda_flags -fPIC -S -g -c -o \"$output_path\""
        
        echo -e "${NC}Running: $cgeist_cmd${NC}"
        echo "Running: $cgeist_cmd" >> "$LOG_FILE"
        
        if eval "$cgeist_cmd" 2>> "$LOG_FILE"; then
            echo -e "${GREEN}  ✓ Success: $output_name${NC}"
            success_count=$((success_count + 1))
            echo "SUCCESS: $rel_path -> $output_name" >> "$LOG_FILE"
        else
            echo -e "${RED}  ✗ Error processing: $rel_path${NC}"
            error_count=$((error_count + 1))
            echo "ERROR: Failed to process $rel_path" >> "$LOG_FILE"
        fi
        
        echo "" >> "$LOG_FILE"  # Add blank line for readability
    done

    # Phase 1 Summary
    echo -e "\n${PURPLE}=== PHASE 1 SUMMARY ===${NC}"
    echo -e "${GREEN}Total files processed: $total_files${NC}"
    echo -e "${GREEN}Successful: $success_count${NC}"
    echo -e "${RED}Errors: $error_count${NC}"

    echo "Phase 1 completed at $(date)" >> "$LOG_FILE"
fi

# ====== PHASE 2: POLYGEIST-OPT RAISE-AFFINE-TO-LINALG PROCESSING ======
if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 2 ]; then
    echo -e "\n${PURPLE}=== PHASE 2: POLYGEIST-OPT RAISE-AFFINE-TO-LINALG PROCESSING ===${NC}"

    # Find all successfully generated .mlir files
    mapfile -t mlir_files < <(find "$OUTPUT_DIR" -name "*.mlir" -type f)
    total_mlir_files=${#mlir_files[@]}

    if [ $total_mlir_files -eq 0 ]; then
        echo -e "${RED}No .mlir files found in $OUTPUT_DIR${NC}"
        if [ "$MODE" -eq 2 ]; then
            echo -e "${RED}Cannot run phase 2 without existing intermediate files${NC}"
            echo -e "${YELLOW}Hint: Run mode 1 first to generate intermediate files${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Found $total_mlir_files .mlir files to process with polygeist-opt${NC}"

        # Reset counters for phase 2
        counter=0

        echo "Starting Phase 2: polygeist-opt --raise-affine-to-linalg processing" >> "$LOG_FILE"

        # Process each .mlir file
        for mlir_file in "${mlir_files[@]}"; do
            counter=$((counter + 1))
            
            # Get filename without path and replace -polygeist-intermediate.mlir with -polygeist-linalg.mlir
            base_name=$(basename "$mlir_file")
            linalg_output_name=$(echo "$base_name" | sed 's|-polygeist-intermediate\.mlir$|-polygeist-linalg.mlir|')
            linalg_output_path="$LINALG_OUTPUT_DIR/$linalg_output_name"
            
            echo -e "${YELLOW}[$counter/$total_mlir_files] Processing: $base_name${NC}"
            
            # Run polygeist-opt command
            polygeist_opt_cmd="polygeist-opt --raise-affine-to-linalg \"$mlir_file\" -o \"$linalg_output_path\""
            
            echo -e "${NC}Running: $polygeist_opt_cmd${NC}"
            echo "Running: $polygeist_opt_cmd" >> "$LOG_FILE"
            
            if eval "$polygeist_opt_cmd" 2>> "$LOG_FILE"; then
                echo -e "${GREEN}  ✓ Success: $linalg_output_name${NC}"
                linalg_success_count=$((linalg_success_count + 1))
                echo "SUCCESS: $base_name -> $linalg_output_name" >> "$LOG_FILE"
            else
                echo -e "${RED}  ✗ Error processing: $base_name${NC}"
                linalg_error_count=$((linalg_error_count + 1))
                echo "ERROR: Failed to process $base_name with polygeist-opt --raise-affine-to-linalg" >> "$LOG_FILE"
            fi
            
            echo "" >> "$LOG_FILE"  # Add blank line for readability
        done
    fi

    # Phase 2 Summary (only if files were processed)
    if [ $total_mlir_files -gt 0 ]; then
        echo -e "\n${PURPLE}=== PHASE 2 SUMMARY ===${NC}"
        echo -e "${GREEN}Total MLIR files processed: $total_mlir_files${NC}"
        echo -e "${GREEN}Successful: $linalg_success_count${NC}"
        echo -e "${RED}Errors: $linalg_error_count${NC}"
    fi

    echo "Phase 2 completed at $(date)" >> "$LOG_FILE"
fi

# ====== PHASE 3: POLYGEIST-OPT CANONICALIZE PROCESSING ======
if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 3 ]; then
    echo -e "\n${PURPLE}=== PHASE 3: POLYGEIST-OPT CANONICALIZE PROCESSING ===${NC}"

    # Find all successfully generated linalg .mlir files
    mapfile -t linalg_files < <(find "$LINALG_OUTPUT_DIR" -name "*.mlir" -type f)
    total_canonicalize_files=${#linalg_files[@]}

    if [ $total_canonicalize_files -eq 0 ]; then
        echo -e "${RED}No .mlir files found in $LINALG_OUTPUT_DIR${NC}"
        if [ "$MODE" -eq 3 ]; then
            echo -e "${RED}Cannot run phase 3 without existing linalg files${NC}"
            echo -e "${YELLOW}Hint: Run mode 2 first to generate linalg files${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Found $total_canonicalize_files .mlir files to process with canonicalize${NC}"

        # Reset counters for phase 3
        counter=0

        echo "Starting Phase 3: polygeist-opt --canonicalize processing" >> "$LOG_FILE"

        # Process each .mlir file
        for linalg_file in "${linalg_files[@]}"; do
            counter=$((counter + 1))
            
            # Get filename without path and replace -polygeist-linalg.mlir with -polygeist-canonicalized.mlir
            base_name=$(basename "$linalg_file")
            canonicalize_output_name=$(echo "$base_name" | sed 's|-polygeist-linalg\.mlir$|-polygeist-canonicalized.mlir|')
            canonicalize_output_path="$CANONICALIZE_OUTPUT_DIR/$canonicalize_output_name"
            
            echo -e "${YELLOW}[$counter/$total_canonicalize_files] Processing: $base_name${NC}"
            
            # Run polygeist-opt command with canonicalize
            canonicalize_cmd="polygeist-opt --canonicalize \"$linalg_file\" -o \"$canonicalize_output_path\""
            
            echo -e "${NC}Running: $canonicalize_cmd${NC}"
            echo "Running: $canonicalize_cmd" >> "$LOG_FILE"
            
            if eval "$canonicalize_cmd" 2>> "$LOG_FILE"; then
                echo -e "${GREEN}  ✓ Success: $canonicalize_output_name${NC}"
                canonicalize_success_count=$((canonicalize_success_count + 1))
                echo "SUCCESS: $base_name -> $canonicalize_output_name" >> "$LOG_FILE"
            else
                echo -e "${RED}  ✗ Error processing: $base_name${NC}"
                canonicalize_error_count=$((canonicalize_error_count + 1))
                echo "ERROR: Failed to process $base_name with polygeist-opt --canonicalize" >> "$LOG_FILE"
            fi
            
            echo "" >> "$LOG_FILE"  # Add blank line for readability
        done
    fi

    # Phase 3 Summary (only if files were processed)
    if [ $total_canonicalize_files -gt 0 ]; then
        echo -e "\n${PURPLE}=== PHASE 3 SUMMARY ===${NC}"
        echo -e "${GREEN}Total linalg files processed: $total_canonicalize_files${NC}"
        echo -e "${GREEN}Successful: $canonicalize_success_count${NC}"
        echo -e "${RED}Errors: $canonicalize_error_count${NC}"
    fi

    echo "Phase 3 completed at $(date)" >> "$LOG_FILE"
fi

# ====== PHASE 4: POLYGEIST-OPT LINALG-debufferize PROCESSING ======
if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 4 ]; then
    echo -e "\n${PURPLE}=== PHASE 4: POLYGEIST-OPT LINALG-debufferize PROCESSING ===${NC}"

    # Find all successfully generated canonicalized .mlir files
    mapfile -t canonicalized_files < <(find "$CANONICALIZE_OUTPUT_DIR" -name "*.mlir" -type f)
    total_debufferized_files=${#canonicalized_files[@]}

    if [ $total_debufferized_files -eq 0 ]; then
        echo -e "${RED}No .mlir files found in $CANONICALIZE_OUTPUT_DIR${NC}"
        if [ "$MODE" -eq 4 ]; then
            echo -e "${RED}Cannot run phase 4 without existing canonicalized files${NC}"
            echo -e "${YELLOW}Hint: Run mode 3 first to generate canonicalized files${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Found $total_debufferized_files .mlir files to process with linalg-debufferize${NC}"

        # Reset counters for phase 4
        counter=0

        echo "Starting Phase 4: polygeist-opt --linalg-debufferize processing" >> "$LOG_FILE"

        # Process each .mlir file
        for canonicalized_file in "${canonicalized_files[@]}"; do
            counter=$((counter + 1))
            
            # Get filename without path and replace -polygeist-canonicalized.mlir with -polygeist-debufferized.mlir
            base_name=$(basename "$canonicalized_file")
            debufferized_output_name=$(echo "$base_name" | sed 's|-polygeist-canonicalized\.mlir$|-polygeist-debufferized.mlir|')
            debufferized_output_path="$debufferizeD_OUTPUT_DIR/$debufferized_output_name"
            
            echo -e "${YELLOW}[$counter/$total_debufferized_files] Processing: $base_name${NC}"
            
            # Run polygeist-opt command with linalg-debufferize
            debufferize_cmd="polygeist-opt --linalg-debufferize \"$canonicalized_file\" -o \"$debufferized_output_path\""
            
            echo -e "${NC}Running: $debufferize_cmd${NC}"
            echo "Running: $debufferize_cmd" >> "$LOG_FILE"
            
            if eval "$debufferize_cmd" 2>> "$LOG_FILE"; then
                echo -e "${GREEN}  ✓ Success: $debufferized_output_name${NC}"
                debufferized_success_count=$((debufferized_success_count + 1))
                echo "SUCCESS: $base_name -> $debufferized_output_name" >> "$LOG_FILE"
            else
                echo -e "${RED}  ✗ Error processing: $base_name${NC}"
                debufferized_error_count=$((debufferized_error_count + 1))
                echo "ERROR: Failed to process $base_name with polygeist-opt --linalg-debufferize" >> "$LOG_FILE"
            fi
            
            echo "" >> "$LOG_FILE"  # Add blank line for readability
        done
    fi

    # Phase 4 Summary (only if files were processed)
    if [ $total_debufferized_files -gt 0 ]; then
        echo -e "\n${PURPLE}=== PHASE 4 SUMMARY ===${NC}"
        echo -e "${GREEN}Total canonicalized files processed: $total_debufferized_files${NC}"
        echo -e "${GREEN}Successful: $debufferized_success_count${NC}"
        echo -e "${RED}Errors: $debufferized_error_count${NC}"
    fi

    echo "Phase 4 completed at $(date)" >> "$LOG_FILE"
fi

# ====== PHASE 5: POLYGEIST-OPT RAISE-AFFINE-TO-LINALG-PIPELINE PROCESSING ======
if [ "$MODE" -eq 5 ]; then
    echo -e "\n${PURPLE}=== PHASE 5: POLYGEIST-OPT AFFINE-PARALLELIZE + RAISE-AFFINE-TO-LINALG-PIPELINE PROCESSING ===${NC}"

    # Find all successfully generated .mlir files from cgeist (same as phase 2)
    mapfile -t mlir_files < <(find "$OUTPUT_DIR" -name "*.mlir" -type f)
    total_pipeline_files=${#mlir_files[@]}

    if [ $total_pipeline_files -eq 0 ]; then
        echo -e "${RED}No .mlir files found in $OUTPUT_DIR${NC}"
        echo -e "${RED}Cannot run phase 5 without existing intermediate files${NC}"
        echo -e "${YELLOW}Hint: Run mode 1 first to generate intermediate files${NC}"
        exit 1
    else
        echo -e "${YELLOW}Found $total_pipeline_files .mlir files to process with pipeline${NC}"

        # Reset counters for phase 5
        counter=0

        echo "Starting Phase 5: polygeist-opt --affine-parallelize --raise-affine-to-linalg-pipeline processing" >> "$LOG_FILE"

        # Process each .mlir file
        for mlir_file in "${mlir_files[@]}"; do
            counter=$((counter + 1))
            
            # Get filename without path and replace -polygeist-intermediate.mlir with -polygeist-pipeline.mlir
            base_name=$(basename "$mlir_file")
            pipeline_output_name=$(echo "$base_name" | sed 's|-polygeist-intermediate\.mlir$|-polygeist-pipeline.mlir|')
            pipeline_output_path="$PIPELINE_OUTPUT_DIR/$pipeline_output_name"
            
            echo -e "${YELLOW}[$counter/$total_pipeline_files] Processing: $base_name${NC}"
            
            # Run polygeist-opt command with affine-parallelize then the pipeline
            pipeline_cmd="polygeist-opt --affine-parallelize --raise-affine-to-linalg-pipeline \"$mlir_file\" -o \"$pipeline_output_path\""
            
            echo -e "${NC}Running: $pipeline_cmd${NC}"
            echo "Running: $pipeline_cmd" >> "$LOG_FILE"
            
            if eval "$pipeline_cmd" 2>> "$LOG_FILE"; then
                echo -e "${GREEN}  ✓ Success: $pipeline_output_name${NC}"
                pipeline_success_count=$((pipeline_success_count + 1))
                echo "SUCCESS: $base_name -> $pipeline_output_name" >> "$LOG_FILE"
            else
                echo -e "${RED}  ✗ Error processing: $base_name${NC}"
                pipeline_error_count=$((pipeline_error_count + 1))
                echo "ERROR: Failed to process $base_name with polygeist-opt --affine-parallelize --raise-affine-to-linalg-pipeline" >> "$LOG_FILE"
            fi
            
            echo "" >> "$LOG_FILE"  # Add blank line for readability
        done
    fi

    # Phase 5 Summary
    echo -e "\n${PURPLE}=== PHASE 5 SUMMARY ===${NC}"
    echo -e "${GREEN}Total MLIR files processed: $total_pipeline_files${NC}"
    echo -e "${GREEN}Successful: $pipeline_success_count${NC}"
    echo -e "${RED}Errors: $pipeline_error_count${NC}"

    echo "Phase 5 completed at $(date)" >> "$LOG_FILE"
fi

# ====== FINAL SUMMARY ======
echo -e "\n${PURPLE}=== FINAL SUMMARY ===${NC}"

if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 1 ]; then
    echo -e "${BLUE}Phase 1 (cgeist):${NC}"
    echo -e "${GREEN}  Total C files processed: $total_files${NC}"
    echo -e "${GREEN}  Successful: $success_count${NC}"
    echo -e "${RED}  Errors: $error_count${NC}"
fi

if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 2 ]; then
    echo -e "${BLUE}Phase 2 (raise-affine-to-linalg):${NC}"
    echo -e "${GREEN}  Total MLIR files processed: $total_mlir_files${NC}"
    echo -e "${GREEN}  Successful: $linalg_success_count${NC}"
    echo -e "${RED}  Errors: $linalg_error_count${NC}"
fi

if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 3 ]; then
    echo -e "${BLUE}Phase 3 (canonicalize):${NC}"
    echo -e "${GREEN}  Total linalg files processed: $total_canonicalize_files${NC}"
    echo -e "${GREEN}  Successful: $canonicalize_success_count${NC}"
    echo -e "${RED}  Errors: $canonicalize_error_count${NC}"
fi

if [ "$MODE" -eq 0 ] || [ "$MODE" -eq 4 ]; then
    echo -e "${BLUE}Phase 4 (linalg-debufferize):${NC}"
    echo -e "${GREEN}  Total canonicalized files processed: $total_debufferized_files${NC}"
    echo -e "${GREEN}  Successful: $debufferized_success_count${NC}"
    echo -e "${RED}  Errors: $debufferized_error_count${NC}"
fi

if [ "$MODE" -eq 5 ]; then
    echo -e "${BLUE}Phase 5 (affine-parallelize + raise-affine-to-linalg-pipeline):${NC}"
    echo -e "${GREEN}  Total MLIR files processed: $total_pipeline_files${NC}"
    echo -e "${GREEN}  Successful: $pipeline_success_count${NC}"
    echo -e "${RED}  Errors: $pipeline_error_count${NC}"
fi

echo -e "${YELLOW}Output directories:${NC}"
echo -e "${YELLOW}  Intermediate files: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}  Linalg files: $LINALG_OUTPUT_DIR${NC}"
echo -e "${YELLOW}  Canonicalized files: $CANONICALIZE_OUTPUT_DIR${NC}"
echo -e "${YELLOW}  debufferized files: $debufferizeD_OUTPUT_DIR${NC}"
echo -e "${YELLOW}  Pipeline files: $PIPELINE_OUTPUT_DIR${NC}"
echo -e "${YELLOW}  Log file: $LOG_FILE${NC}"

echo "Run completed at $(date)" >> "$LOG_FILE"

# Determine exit code
total_errors=$((error_count + linalg_error_count + canonicalize_error_count + debufferized_error_count + pipeline_error_count))
if [ $total_errors -eq 0 ]; then
    echo -e "${GREEN}All files processed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some files failed to process. Check $LOG_FILE for details.${NC}"
    echo -e "${RED}Total errors: $total_errors${NC}"
    exit 1
fi 