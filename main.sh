#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++}
CXXFLAGS="-O3 -Wall -Wextra -pedantic -std=c++17"
EIGEN_INC=${EIGEN_INC:-/usr/include/eigen3}

mkdir -p bin results

$CXX $CXXFLAGS -I"$EIGEN_INC" main.cpp -o bin/fastkpp_demo

DATA=../data/3D_road.csv
K=4096
SEED=0

for A in 0 1 2; do
  echo "Running acc=$A..."
  echo "CMD: bin/fastkpp_demo -i $DATA -k $K -s $SEED -a $A" | tee "results/run_a${A}.txt"
  bin/fastkpp_demo -i "$DATA" -k "$K" -s "$SEED" -a "$A" | tee -a "results/run_a${A}.txt"
done

echo "Done. See results/run_a*.txt"

