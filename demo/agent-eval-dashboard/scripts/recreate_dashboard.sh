#!/bin/bash
# Recreate the entire dashboard from Downloaded dataset
# This script runs add_results.py in batch mode for each dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${1:-demo/agent-eval-dashboard/example-dataset}"

echo "=============================================="
echo "  Dashboard Recreation Script"
echo "=============================================="
echo ""
echo "Data directory: $DATA_DIR"
echo ""

# Check if data directory exists
if [ ! -d "$REPO_ROOT/$DATA_DIR" ]; then
    echo "Error: Directory $REPO_ROOT/$DATA_DIR does not exist"
    exit 1
fi

cd "$REPO_ROOT"

# Function to count existing datasets
count_datasets() {
    if [ -f "demo/agent-eval-dashboard/leaderboard_data.json" ]; then
        python3 -c "import json; data=json.load(open('demo/agent-eval-dashboard/leaderboard_data.json')); print(len(data.get('datasets', {})))"
    else
        echo "0"
    fi
}

# Function to run batch add with automatic inputs
add_batch() {
    local batch_dir=$1
    local dataset_name=$2

    echo ""
    echo "=============================================="
    echo "  Processing: $dataset_name"
    echo "=============================================="
    echo ""

    # Count how many datasets exist to determine the "create new" option number
    local num_datasets=$(count_datasets)
    local create_new_option=$((num_datasets + 1))

    # Feed inputs: y (confirm add all), create new dataset option, dataset name
    python demo/agent-eval-dashboard/scripts/add_results.py --batch "$batch_dir" <<EOF
y
$create_new_option
$dataset_name
EOF
}

# Add tau2bench airline
add_batch "$DATA_DIR/tau2bench/airline" "Tau2Bench – Airline"

# Add tau2bench retail
add_batch "$DATA_DIR/tau2bench/retail" "Tau2Bench – Retail"

# Add toolsandbox
add_batch "$DATA_DIR/toolsandbox" "ToolSandbox"

echo ""
echo "=============================================="
echo "  Dashboard recreation complete!"
echo "=============================================="
echo ""
echo "View the dashboard:"
echo "  open demo/agent-eval-dashboard/leaderboard/index.html"
echo ""
