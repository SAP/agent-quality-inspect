#!/bin/bash

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments: run all tests
    TEST_PATTERN="tests_acceptance/"
else
    # Arguments provided: use them as test pattern(s)
    TEST_PATTERN="$@"
fi

echo -e "${YELLOW}Starting acceptance test workflow...${NC}\n"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Show what will be tested
echo -e "${BLUE}Will run tests: ${TEST_PATTERN}${NC}\n"

# Step 1: Clean previous builds
echo -e "${YELLOW}[1/6] Cleaning previous builds...${NC}"
rm -rf dist/ *.egg-info

# Step 2: Build the package
echo -e "${YELLOW}[2/6] Building package...${NC}"
python3 -m pip install --quiet --upgrade build
python3 -m build

# Step 3: Create virtual environment
echo -e "${YELLOW}[3/6] Creating virtual environment...${NC}"
VENV_DIR=".venv_acceptance"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"

# Step 4: Activate virtual environment and install package
echo -e "${YELLOW}[4/6] Installing package in virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install the built wheel
pip install --quiet --upgrade pip
pip install --quiet dist/*.whl

# Install pytest (test runner)
pip install --quiet -r requirements_test.txt

# Step 5: Run acceptance tests
echo -e "${YELLOW}[5/6] Running acceptance tests...${NC}"
set +e
pytest $TEST_PATTERN --color=yes -v -n auto 2>&1 | tee tests_acceptance/logs.txt
TEST_RESULT=$?
set -e

# Step 6: Cleanup
echo -e "${YELLOW}[6/6] Cleaning up...${NC}"
deactivate
rm -rf "$VENV_DIR"

# Report results
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✓ Acceptance tests completed successfully!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Acceptance tests failed!${NC}"
    exit 1
fi
