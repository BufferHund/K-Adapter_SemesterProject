#!/bin/bash
# ====================================================================
# MASTER SCRIPT FOR TACRED ZERO-SHOT EVALUATIONS
#
# Runs a series of evaluation-only scripts for the TACRED dataset.
# ====================================================================

# Create a directory for log files
LOG_DIR="tacred_evaluation_logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating TACRED evaluation log directory: $LOG_DIR"
    mkdir "$LOG_DIR"
fi

# --- Define all evaluation scripts to run ---
EVAL_SCRIPTS=(
    "eval_tacred_zeroshot_baseline.sh"
    "eval_tacred_zeroshot_fac_only.sh"
    "eval_tacred_zeroshot_lin_only.sh"
    "eval_tacred_zeroshot_fac_lin_add.sh"
    "eval_tacred_zeroshot_fac_lin_concat.sh"
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
    echo "[$(date)] STARTING TACRED EVALUATION $CURRENT_SCRIPT/$TOTAL_SCRIPTS: $eval_script"
    echo "Output will be logged to: $log_file"
    echo "===================================================================="

    # Execute the script and redirect all output to its log file
    bash ./${eval_script} > "$log_file" 2>&1

    # Check the exit code of the last command
    if [ $? -eq 0 ]; then
        echo "[$(date)] SUCCESS: TACRED Evaluation $eval_script finished successfully."
    else
        echo "[$(date)] !!!!!!! FAILURE: TACRED Evaluation $eval_script failed. Check logs in $log_file for details."
    fi
    echo ""

done

echo "===================================================================="
echo "[$(date)] All scheduled TACRED evaluations have been attempted."
