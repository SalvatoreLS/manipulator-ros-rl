#!/bin/bash
# Run all tests for distance_based_rl package
# Usage: ./run_all_tests.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ROS_WS="$(dirname "$(dirname "$PKG_DIR")")"

echo "======================================"
echo "  Distance-Based RL - Full Test Suite"
echo "======================================"
echo ""
echo "Workspace: $ROS_WS"
echo "Package: $PKG_DIR"
echo ""

# Source ROS2 setup
if [ -f "$ROS_WS/install/setup.bash" ]; then
    source "$ROS_WS/install/setup.bash"
    echo "✓ ROS2 environment sourced"
else
    echo "⚠ Could not source ROS2 setup.bash"
fi

echo ""
echo "======================================"
echo "  Running Linter Tests (flake8, copyright, pep257)"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_flake8.py -v || echo "⚠ Flake8 tests failed"
python3 -m pytest test/test_copyright.py -v || echo "⚠ Copyright tests failed"
python3 -m pytest test/test_pep257.py -v || echo "⚠ PEP257 tests failed"

echo ""
echo "======================================"
echo "  Running Unit Tests"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_sac_agent.py -v

echo ""
echo "======================================"
echo "  Running Integration Tests"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_environment.py -v

echo ""
echo "======================================"
echo "  Test Summary"
echo "======================================"
echo "✓ Full test suite completed!"
echo ""
echo "To run specific tests:"
echo "  - Unit tests only:        ./run_unit_tests.sh"
echo "  - Integration tests only: ./run_integration_tests.sh"
echo "  - Lint check only:        ./check_lint.sh"
echo "  - View coverage:          ./view_test_coverage.sh"
