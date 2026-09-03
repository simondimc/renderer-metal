#!/bin/bash
set -e

# 1. Save the starting project root directory
PROJECT_DIR="$(pwd)"

echo "Cleaning up old build directory..."
rm -rf build

echo "Creating new build directory..."
mkdir build && cd build

echo "Running CMake configuration..."
cmake ..

echo "Compiling the project..."
cmake --build .

echo "Launching Renderer..."
./Renderer

# 2. Move one level back to the project root directory
cd "$PROJECT_DIR"
