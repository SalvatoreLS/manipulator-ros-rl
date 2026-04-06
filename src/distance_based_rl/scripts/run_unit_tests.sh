#!/bin/bash
# Run only unit tests for distance_based_rl package
# Usage: ./run_unit_tests.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ROS_WS="$(dirname "$(dirname "$PKG_DIR")")"

echo "======================================"
echo "  Distance-Based RL - Unit Tests"
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
echo "  SAC Agent Tests"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_sac_agent.py::TestFCGP -v
python3 -m pytest test/test_sac_agent.py::TestReplayBuffer -v
python3 -m pytest test/test_sac_agent.py::TestSACAgent -v

echo ""
echo "======================================"
echo "  Environment Component Tests"
echo "======================================"
cd "$PKG_DIR"
python3 -m pytest test/test_environment.py::TestDataHandlerMocked -v
python3 -m pytest test/test_environment.py::TestManipulatorEnvIntegration -v

echo ""
echo "✓ Unit tests completed!"
