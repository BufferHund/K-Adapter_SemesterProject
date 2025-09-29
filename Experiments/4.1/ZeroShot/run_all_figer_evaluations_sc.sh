#!/bin/bash
# ====================================================================
# MASTER SCRIPT FOR FIGER ZERO-SHOT EVALUATIONS
#
# Runs a series of evaluation-only scripts for the FIGER dataset.
# ====================================================================

# Create a directory for log files
LOG_DIR="figer_evaluation_logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating FIGER evaluation log directory: $LOG_DIR"
    mkdir "$LOG_DIR"
fi

# --- Define all evaluation scripts to run ---
EVAL_SCRIPTS=(
    "eval_figer_zeroshot_baseline.sh"
    "eval_figer_zeroshot_fac_only.sh"
    "eval_figer_zeroshot_lin_only.sh"
    "eval_figer_zeroshot_fac_lin_add.sh"
    "eval_figer_zeroshot_fac_lin_concat.sh"
)

# --- Main execution loop ---
TOTAL_SCRIPTS=${#EVAL_SCRIPTS[@]}
CURRENT_SCRIPT=0

for eval_script in "${EVAL_SCRIPTS[@]}"

do
    ((CURRENT_SCRIPT++))
    log_name="$(basename "$eval_script" .sh)"
    log_file="$LOG_DIR/${log_name}.log"
    
    echo "===================================================================="
    echo "[$(date)] STARTING FIGER EVALUATION $CURRENT_SCRIPT/$TOTAL_SCRIPTS: $eval_script"
    echo "Output will be logged to: $log_file"
    echo "===================================================================="

    # Execute the script and redirect all output to its log file
    bash ./${eval_script} > "$log_file" 2>&1

    # Check the exit code of the last command
    if [ $? -eq 0 ]; then
        echo "[$(date)] SUCCESS: FIGER Evaluation $eval_script finished successfully."
    else
        echo "[$(date)] !!!!!!! FAILURE: FIGER Evaluation $eval_script failed. Check logs in $log_file for details."
    fi
    echo ""

done

echo "===================================================================="
echo "[$(date)] All scheduled FIGER evaluations have been attempted."
