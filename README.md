# Mixed Precision OFRR

Research code for our manuscript "Mixed Precision Orthogonalization-Free Projection Methods for Computing Eigenvalues and Singular Values"

## Disclaimer

This software is provided "AS IS", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. It was developed primarily for research purposes and to accompany the manuscript mentioned above.

While efforts have been made to ensure correctness, the code may contain bugs or change in future versions. The authors assume no responsibility or liability for any potential damage or loss resulting from the use of this software. Use at your own risk. Support and maintenance for this code may be limited or nonexistent.

## C++

1. Building

Basic build:
```bash
mkdir build
cd build
cmake ..
make
```

Build with tests enabled:
```bash
mkdir build
cd build
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
make
```

Build with debug information:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Running Tests

After building with tests enabled:

```bash
# Run all tests
cd build
ctest

# Or run specific tests
./tests/vector_test
./tests/matrix_test
./tests/mvops_test
```

Run tests with more detailed output:
```bash
cd build
ctest --verbose
```

## Code Coverage

To build with code coverage enabled:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make
```

Run tests to generate coverage data:
```bash
cd build
# Run all tests
make test

# Or run specific tests
./tests/vector_test
./tests/matrix_test
./tests/mvops_test
```

Generate coverage report:
```bash
cd build
# Generate coverage report (assuming you have coverage target configured)
make coverage

# Or manually generate with lcov
lcov --directory . --capture --output-file coverage.info
# Filter out system headers and test files
lcov --remove coverage.info '/usr/*' '/home/*/googletest/*' '*/tests/*' --output-file coverage.info
# Generate HTML report
genhtml coverage.info --output-directory coverage_report
```

View the coverage report:
```bash
# Open index.html in the coverage_report directory
firefox coverage_report/index.html
```

## Usage

```cpp
#include "containers/vector.hpp"

// Create a vector on the device
msvd::Vector<float> device_vec(1000, msvd::Location::kDEVICE);

// Fill it with random values
device_vec.fill_random();

// Transfer to host
device_vec.to_host();

// Convert to double precision
msvd::Vector<double> double_vec = device_vec.cast<double>();

// Access elements (on host)
for (size_t i = 0; i < double_vec.length(); i++) {
   double value = double_vec.data()[i];
   // Do something with the value
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
