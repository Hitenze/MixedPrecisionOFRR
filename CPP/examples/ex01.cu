#include "../linalg/blas/mvops.hpp"
#include "../containers/vector.hpp"
#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/type_utils.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_fp16.h>

using namespace msvd;
using namespace std;
using namespace std::chrono;

// Timer utility function
template<typename T>
double measure_time(T&& func) {
   auto start = high_resolution_clock::now();
   func();
   auto end = high_resolution_clock::now();
   duration<double, milli> elapsed = end - start;
   return elapsed.count();
}

// Test matrix multiplication performance
template<typename T, typename T_COMPUTE>
void test_matrix_multiplication(size_t m, size_t n, size_t k) {
   cout << "========== Testing " << (std::is_same<T, double>::value ? "double" : 
                                   (std::is_same<T, float>::value ? "float" : "half")) 
        << " precision matrix multiplication ==========" << endl;
   cout << "Matrix dimensions: A(" << m << "x" << k << ") * B(" << k << "x" << n << ") = C(" << m << "x" << n << ")" << endl;
   
   // Create matrices
   Matrix<T> A(m, k, Location::kDEVICE);
   Matrix<T> B(k, n, Location::kDEVICE);
   Matrix<T> C(m, n, Location::kDEVICE);
   
   // Fill matrices with random values
   A.fill_random();
   B.fill_random();
   
   // GEMM parameters
   T_COMPUTE alpha = get_one<T_COMPUTE>();
   T_COMPUTE beta = get_zero<T_COMPUTE>();
   
   // Warmup
   cout << "Warming up..." << endl;
   for (int i = 0; i < 5; i++) {
      gemm<T, T, T_COMPUTE>(false, false, alpha, A, B, beta, C);
   }
   
   // Perform multiple measurements for more accurate results
   const int num_iterations = 10;
   vector<double> times;
   
   cout << "Performing measurements..." << endl;
   for (int i = 0; i < num_iterations; i++) {
      double elapsed = measure_time([&]() {
         gemm<T, T, T_COMPUTE>(false, false, alpha, A, B, beta, C);
      });
      times.push_back(elapsed);
   }
   
   // Calculate statistics
   double total_time = 0.0;
   double min_time = times[0];
   double max_time = times[0];
   
   for (double time : times) {
      total_time += time;
      min_time = min(min_time, time);
      max_time = max(max_time, time);
   }
   
   double avg_time = total_time / num_iterations;
   
   // Calculate FLOPS (floating point operations)
   // Each matrix element requires k multiplications and k-1 additions, approximately 2*m*n*k operations
   double flops = 2.0 * m * n * k;
   double gflops = (flops / 1e9) / (avg_time / 1000.0); // Convert to GFLOPS
   
   // Output results
   cout << "Performance results:" << endl;
   cout << "  Average time: " << fixed << setprecision(3) << avg_time << " ms" << endl;
   cout << "  Minimum time: " << fixed << setprecision(3) << min_time << " ms" << endl;
   cout << "  Maximum time: " << fixed << setprecision(3) << max_time << " ms" << endl;
   cout << "  Performance: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
   cout << endl;
}

int main() {
   // Initialize CUDA
   CUDAHandler::init();
   
   cout << "===============================================" << endl;
   cout << "Matrix Multiplication (GEMM) Performance Test" << endl;
   cout << "===============================================" << endl;
   
   // Test different matrix sizes
   vector<tuple<size_t, size_t, size_t>> matrix_sizes = {
      {1024, 1024, 1024},  // Square matrix multiplication
      {2048, 2048, 2048},  // Larger square matrix
      {4096, 4096, 4096},  // Large square matrix
      {4096, 1024, 2048}   // Non-square matrix
   };
   
   for (const auto& [m, n, k] : matrix_sizes) {
      // Double precision test
      test_matrix_multiplication<double, double>(m, n, k);
      
      // Single precision test
      test_matrix_multiplication<float, float>(m, n, k);
      
      // Half precision test (using float for computation)
      test_matrix_multiplication<__half, float>(m, n, k);
      
      cout << "-----------------------------------------------" << endl;
   }
   
   // Cleanup CUDA resources
   CUDAHandler::finalize();
   
   return 0;
} 