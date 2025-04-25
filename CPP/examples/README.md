# MSVD Examples

This directory contains example programs demonstrating how to use the MSVD library for various linear algebra operations.

## Building Examples

Examples are built automatically when you compile the MSVD library with the `BUILD_EXAMPLES` option enabled (on by default):

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ..
make
```

To build only a specific example:

```bash
cd build
make ex00  # Build only the QR factorization example
```

## Available Examples

### QR Factorization Performance Comparison (ex00)

This example compares the performance of different QR factorization methods (MGS, CGS2, MGS_V2) using matrices of various sizes and precision types (double, float, half).

```bash
./examples/ex00
```

### Matrix Multiplication Performance (ex01)

This example benchmarks matrix multiplication (GEMM) performance with different matrix sizes and precision types.

```bash
./examples/ex01
```

## Adding New Examples

To add a new example, simply create a new file named `exXX.cu` in this directory (where XX is a number). The build system will automatically detect and build it.

All examples should:

1. Include necessary headers from the MSVD library
2. Initialize CUDA resources using `CUDAHandler::init()` and clean up with `CUDAHandler::finalize()`
3. Properly handle different precision types (double, float, half) when applicable
4. Include performance measurements when demonstrating algorithms

## Example Template

```cpp
#include "../core/utils/cuda_handler.hpp"
// Include other necessary headers

using namespace msvd;

int main() {
    // Initialize CUDA
    CUDAHandler::init();
    
    // Your example code here
    
    // Cleanup CUDA resources
    CUDAHandler::finalize();
    
    return 0;
}
``` 