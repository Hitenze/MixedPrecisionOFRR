# Mixed Precision OFRR

Research code for our manuscript "Mixed Precision Orthogonalization-Free Projection Methods for Computing Eigenvalues and Singular Values"

## Disclaimer

This software is provided "AS IS", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. It was developed primarily for research purposes and to accompany the manuscript mentioned above.

While efforts have been made to ensure correctness, the code may contain bugs or change in future versions. The authors assume no responsibility or liability for any potential damage or loss resulting from the use of this software. Use at your own risk. Support and maintenance for this code may be limited or nonexistent.

## C++

0. Development

   The C++ code was developed by Tianshi Xu under the advisory of Yuanzhe Xi and Yousef Saad. Other co-authors have **NO CONTRIBUTION** to the source code and the experiments.

1. Building

   Basic build:
   ```bash
   cd CPP
   mkdir build
   cd build
   cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
   make
   ```

2. Validation

   ```bash
   ctest --verbose
   ```

3. Reproduce the results in the manuscript

   ```bash
   ./examples/ex00
   ```

## MATLAB

1. Reproduce the results in the manuscript

   Simply go to folder MATLAB and run test scripts

## License

This project is licensed under the MIT License - see the LICENSE file for details.
