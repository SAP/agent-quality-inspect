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
    TEST_PATTERN="tests/"
else
    # Arguments provided: use them as test pattern(s)
    TEST_PATTERN="$@"
fi

echo -e "${YELLOW}Starting unit test workflow...${NC}\n"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Show what will be tested
echo -e "${BLUE}Will run tests: ${TEST_PATTERN}${NC}\n"

# Step 1: Create virtual environment
echo -e "${YELLOW}[1/4] Creating virtual environment...${NC}"
VENV_DIR=".venv_unit"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"

# Step 2: Activate virtual environment and install dependencies
echo -e "${YELLOW}[2/4] Installing dependencies in virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Install pip and dependencies from pyproject.toml
pip install --quiet --upgrade pip
pip install --quiet -e .  # Editable install using pyproject.toml dependencies
pip install --quiet -r requirements_test.txt

# Step 3: Run unit tests with coverage
echo -e "${YELLOW}[3/4] Running unit tests with coverage...${NC}"
set +e
pytest "$TEST_PATTERN" --color=yes -v -n auto --cov=agent_inspect --cov-report=term-missing --cov-report=xml:coverage.xml
TEST_RESULT=$?
set -e

# Step 4: Cleanup
echo -e "${YELLOW}[4/4] Cleaning up...${NC}"
deactivate
rm -rf "$VENV_DIR"

# Report results
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✓ Unit tests completed successfully!${NC}"
    echo -e "${BLUE}Coverage report saved to coverage.xml${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Unit tests failed!${NC}"
    exit 1
fi
