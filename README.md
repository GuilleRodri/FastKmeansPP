# FastKPP — Exact accelerated k-means++ seeding (C++)

FastKPP is a small, header-only C++17 library implementing **exact (distribution-preserving) accelerations of k-means++** seeding using two geometry-aware filters:

- **TIE** (triangle-inequality pruning)
- **TIE+Norm** (triangle-inequality pruning + norm-based filtering)

The accelerations reduce redundant distance computations while preserving the original **D² sampling distribution** of k-means++.

## Features

- Header-only C++ library (C++17)
- Eigen-based implementation
- Minimal command-line interface (CLI) example
- Regression tests (golden outputs) for reproducibility

## Repository structure

- `fastkmeanspp.hpp` : header-only library
- `main.cpp` : minimal CLI example
- `data/`  : datasets
- `tests/`  : regression tests

## Requirements

- C++17 compiler (e.g., g++ >= 11)
- Eigen (>= 3.4 recommended)
- make (for tests)

On Ubuntu/Debian:

sudo apt-get update

sudo apt-get install -y g++ libeigen3-dev make

## Build (CLI Example)

From the repository root directory:

g++ -O3 -Wall -Wextra -std=c++17 -I/usr/include/eigen3 main.cpp -o main

## Run

Examples:

./main -i ../data/3D_road.csv -k 4096 -s 0 -a 0
./main -i ../data/3D_road.csv -k 4096 -s 0 -a 1
./main -i ../data/3D_road.csv -k 4096 -s 0 -a 2

## Arguments

-i, --input Path to input file

-k, --clusters Number of centers (k)

-s, --seed RNG seed

-a, --acceleration 0 = Standard, 1 = TIE, 2 = TIE + Norm

## Regression Tests
The tests/ folder contains golden-output regression tests to ensure stable behavior across code changes.

From the tests/ directory:

make update
This regenerates golden outputs. Use this only when you intentionally change the expected results and want to update the reference outputs.

make test
This runs the regression test executable and checks outputs against the committed golden references.

## Reproducibility

GitHub repository: https://github.com/GuilleRodri/FastKmeansPP

Code Ocean capsule: https://codeocean.com/capsule/4860715/tree/v1

## Citations

If you use this software, please cite the associated publications:

Rodríguez-Corominas, Blesa, Blum (2025). Accelerating the k-Means++ Algorithm by Using Geometric Information. IEEE Access.

Guillem Rodríguez-Corominas, Maria J. Blesa, Christian Blum. Accelerating the k-Means++ Algorithm by Using Geometric Information. IEEE Access, volume 13, pages 67693-67717. April 2025. doi.org/10.1109/ACCESS.2025.3561293.

Guillem Rodríguez Corominas, Maria J.~Blesa, Christian Blum. Tight bounds for an accelerated  k-means++ algorithm. Proceedings of the International Conference on Artificial Intelligence Applications and Innovations (AIAI 2025), Cyprus. https://doi.org/10.1007/978-3-031-96239-4_8. 

## License

GPL-3.0 — see LICENSE.

## Contact

Guillem Rodríguez-Corominas — grodriguez@iiia.csic.es
