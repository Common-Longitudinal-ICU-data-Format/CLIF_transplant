#!/bin/bash

# Setup script for CLIF to Valeos Inpatient project
# Detects uv and uses it if available, otherwise falls back to traditional venv

echo "Setting up environment for CLIF to Valeos Inpatient project..."

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "Found uv, using it for dependency management..."
    uv sync

    if [ $? -eq 0 ]; then
        echo ""
        echo "Setup completed successfully!"
        echo ""
        echo "To run commands, use 'uv run', for example:"
        echo "  uv run marimo edit code/heart_transplant_report.py"
        echo "  uv run python script.py"
    else
        echo "Error: Failed to install dependencies with uv"
        exit 1
    fi
else
    echo "uv not found, using traditional venv..."

    # Check if virtual environment already exists
    if [ -d ".venv" ]; then
        echo "Virtual environment already exists."
    else
        echo "Creating virtual environment..."
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create virtual environment"
            return 1 2>/dev/null || exit 1
        fi
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source .venv/bin/activate

    # Upgrade pip and install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        echo ""
        echo "Setup completed successfully!"
        echo ""
        echo "To activate the environment in future sessions, run:"
        echo "  source .venv/bin/activate"
    else
        echo "Error: Failed to install dependencies"
        return 1 2>/dev/null || exit 1
    fi
fi
