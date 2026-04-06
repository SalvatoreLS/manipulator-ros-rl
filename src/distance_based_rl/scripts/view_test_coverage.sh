#!/bin/bash
# Run tests with coverage reporting
# Usage: ./view_test_coverage.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ROS_WS="$(dirname "$(dirname "$PKG_DIR")")"

echo "======================================"
echo "  Test Coverage Report"
echo "======================================"
echo ""
echo "Workspace: $ROS_WS"
echo "Package: $PKG_DIR"
echo ""

# Source ROS2 setup
if [ -f "$ROS_WS/install/setup.bash" ]; then
    source "$ROS_WS/install/setup.bash"
fi

# Check if pytest-cov is installed
if ! python3 -c "import pytest_cov" 2>/dev/null; then
    echo "⚠ pytest-cov not installed. Installing..."
    pip3 install pytest-cov
fi

cd "$PKG_DIR"

echo ""
echo "======================================"
echo "  Running tests with coverage"
echo "======================================"
python3 -m pytest test/test_sac_agent.py test/test_environment.py \
    --cov=distance_based_rl \
    --cov-report=term-missing \
    --cov-report=html \
    -v

echo ""
echo "======================================"
echo "  Coverage Summary"
echo "======================================"
echo "Terminal report shown above"
echo "HTML report generated: htmlcov/index.html"
echo ""
echo "To view HTML report in browser:"
echo "  open htmlcov/index.html        # macOS"
echo "  xdg-open htmlcov/index.html    # Linux"
echo "  start htmlcov/index.html       # Windows"
