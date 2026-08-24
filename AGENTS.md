# Repository Guidelines

## Project Structure & Module Organization

This research repository has parallel CUDA/C++ and MATLAB implementations. `CPP/core/` holds memory and CUDA utilities; `CPP/containers/` provides matrix and vector types; and `CPP/linalg/` contains BLAS operations, factorizations, and eigensolver/SVD solvers. Supporting code lives in `CPP/io/` and `CPP/testproblems/`, with runnable programs in `CPP/examples/` and GoogleTest sources in `CPP/tests/`. MATLAB packages are under `MATLAB/+src/`; manuscript-reproduction scripts are grouped in `MATLAB/+tests/+test0` through `+test4`, and input matrices are in `MATLAB/data/`. Treat `CPP/build/`, plots, and logs as generated artifacts.

## Build, Test, and Development Commands

The C++ implementation requires CMake 3.18+ and a CUDA toolkit. It uses C++17/CUDA 17 and defaults to CUDA architecture 86; override this with `-DCUDA_ARCHITECTURES=<value>` when needed.

```bash
cmake -S CPP -B CPP/build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build CPP/build -j
ctest --test-dir CPP/build --output-on-failure
./CPP/build/examples/ex00
```

The first two commands configure and build libraries, tests, and examples; the third runs all C++ tests; the last reproduces the primary C++ experiment. From MATLAB, run a reproduction script with `cd MATLAB` followed by `run('+tests/+test0/test.m')`.

## Coding Style & Naming Conventions

Match surrounding code; no formatter or linter is configured. Use three-space indentation and same-line braces in C++/CUDA. Name types in PascalCase, functions and locals in `snake_case`, private fields with a leading underscore, and enum values like `kDEVICE`. Keep declarations in `.hpp` and CUDA implementations in `.cu`. MATLAB files and functions use lowercase `snake_case`, semicolons, and leading help comments for public routines.

## Testing Guidelines

Add C++ tests as `CPP/tests/<component>_test.cpp` using GoogleTest `TEST` or `TEST_F`; use PascalCase suite and case names and precision-appropriate tolerances. Run the full CTest suite before submitting. MATLAB scripts are experimental validation rather than `matlab.unittest` tests; report generated numerical comparisons. Coverage is optional (`-DENABLE_COVERAGE=ON`); no minimum threshold is specified.

## Commit & Pull Request Guidelines

History favors short, plain imperative subjects without prefixes, for example `add MATLAB benchmark`. Add a body when rationale or compatibility effects are not obvious. Pull requests should summarize the change, identify affected C++/MATLAB paths, list validation commands and results, link relevant issues, and include plots or numerical comparisons when outputs change.
