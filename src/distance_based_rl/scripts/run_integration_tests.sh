#!/bin/bash
# Run only integration tests for distance_based_rl package
# Usage: ./run_integration_tests.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ROS_WS="$(dirname "$(dirname "$PKG_DIR")")"

echo "======================================"
echo "  Distance-Based RL - Integration Tests"
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

echo ""
echo "======================================"
echo "  Data Handler Mocked Tests"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_environment.py::TestDataHandlerMocked -v

echo ""
echo "======================================"
echo "  Manipulator Environment Tests"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_environment.py::TestManipulatorEnvIntegration -v

echo ""
echo "✓ Integration tests completed!"
