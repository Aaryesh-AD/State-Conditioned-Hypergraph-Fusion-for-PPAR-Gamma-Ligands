#!/usr/bin/env bash
# env-activate.sh
# Activate the virtual environment for mlg-project-repo-t6
# Usage: source env-activate.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}[mlg-project-repo-t6]${NC} Activating environment..."

# Check for conda environment first
if command -v conda &> /dev/null; then
    CONDA_ENV_NAME="mlg-project-repo-t6"

    # Check if conda env exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo -e "${GREEN}✓${NC} Found conda environment: ${CONDA_ENV_NAME}"

        # Initialize conda for the current shell if needed
        if ! command -v conda &> /dev/null; then
            eval "$(conda shell.bash hook)"
        fi

        # Activate the conda environment
        conda activate "${CONDA_ENV_NAME}"

        if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
            echo -e "${GREEN}✓${NC} Conda environment activated: ${CONDA_ENV_NAME}"
            echo -e "${YELLOW}ℹ${NC}  Python: $(which python)"
            echo -e "${YELLOW}ℹ${NC}  Version: $(python --version)"
            return 0
        else
            echo -e "${RED}✗${NC} Failed to activate conda environment"
        fi
    fi
fi

# Check for standard Python venv locations
VENV_PATHS=(
    "${SCRIPT_DIR}/venv"
    "${SCRIPT_DIR}/.venv"
    "${SCRIPT_DIR}/env"
    "${SCRIPT_DIR}/.env"
)

for VENV_PATH in "${VENV_PATHS[@]}"; do
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        echo -e "${GREEN}✓${NC} Found virtual environment at: ${VENV_PATH}"
        source "${VENV_PATH}/bin/activate"

        if [[ -n "$VIRTUAL_ENV" ]]; then
            echo -e "${GREEN}✓${NC} Virtual environment activated"
            echo -e "${YELLOW}ℹ${NC}  Python: $(which python)"
            echo -e "${YELLOW}ℹ${NC}  Version: $(python --version)"
            return 0
        else
            echo -e "${RED}✗${NC} Failed to activate virtual environment"
        fi
    fi
done

# If we get here, no environment was found
echo -e "${RED}✗${NC} No virtual environment found!"
echo ""
echo "To create a conda environment, run:"
echo "  conda env create -f conda-env.yml"
echo ""
echo "To create a Python virtual environment, run:"
echo "  python -m venv venv"
echo "  source venv/bin/activate"
echo "  pip install -e ."
echo ""
return 1
