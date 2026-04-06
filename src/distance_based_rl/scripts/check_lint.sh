#!/bin/bash
# Check code quality: linting, style, and formatting
# Usage: ./check_lint.sh

set +e  # Don't exit on errors, just report them

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ROS_WS="$(dirname "$(dirname "$PKG_DIR")")"

echo "======================================"
echo "  Code Quality Check"
echo "======================================"
echo ""
echo "Workspace: $ROS_WS"
echo "Package: $PKG_DIR"
echo ""

# Source ROS2 setup
if [ -f "$ROS_WS/install/setup.bash" ]; then
    source "$ROS_WS/install/setup.bash"
    echo "✓ ROS2 environment sourced"
fi

cd "$PKG_DIR"

# Counter for failures
FAILURES=0

echo ""
echo "======================================"
echo "  Flake8 (Style Guide)"
echo "======================================"
python3 -m flake8 distance_based_rl/ --max-line-length=100 || ((FAILURES++))

echo ""
echo "======================================"
echo "  Copyright Headers"
echo "======================================"
python3 -m pytest test/test_copyright.py -v || ((FAILURES++))

echo ""
echo "======================================"
echo "  PEP257 (Docstring Convention)"
echo "======================================"
python3 -m pytest test/test_pep257.py -v || ((FAILURES++))

echo ""
echo "======================================"
echo "  Summary"
echo "======================================"
if [ $FAILURES -eq 0 ]; then
    echo "✓ All code quality checks passed!"
else
    echo "⚠ $FAILURES check(s) failed"
    exit 1
fi
