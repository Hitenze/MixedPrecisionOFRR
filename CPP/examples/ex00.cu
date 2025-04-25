#include "../linalg/factorization/qr.hpp"
#include "../linalg/factorization/hessenberg.hpp"
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

// Helper function to get double value from different types
template<typename T>
double to_double(const T& value) {
   if constexpr (std::is_same_v<T, double>) {
      return value;
   } else if constexpr (std::is_same_v<T, float>) {
      return static_cast<double>(value);
   } else if constexpr (std::is_same_v<T, __half>) {
      return static_cast<double>(__half2float(value));
   } else {
      return static_cast<double>(value);
   }
}

// dummy function to run before all tests, run some simple warmup to fill the cache and GPU
void warmup() {
   Matrix<double> A(1000, 1000, Location::kDEVICE);
   A.fill_random();
   Matrix<double> B(1000, 1000, Location::kDEVICE);

   for (int i = 0; i < 10; i++) {
      gemm<double, double, double>(true, false, get_one<double>(), A, A, get_zero<double>(), B);
   }
}

// Test function to compare different QR factorization methods
template<typename T>
void compare_qr_methods(size_t m, size_t n, int ntests) {
   cout << "========== Testing " << (std::is_same<T, double>::value ? "double" : 
                                   (std::is_same<T, float>::value ? "float" : "half")) 
        << " precision (" << m << "x" << n << ") ==========" << endl;
   
   // Create matrices
   Matrix<T> A(m, n, Location::kDEVICE);
   Matrix<T> Q(m, n, Location::kDEVICE);
   Matrix<T> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   cout << "Method comparison (execution time, ms):" << endl;
   
   // Save a copy of A for different methods
   Matrix<T> A_copy(A);
   
   // Test MGS method without re-orthogonalization
   double mgs_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = mgs<T, T, T>(A, Q, R, get_eps<T>(), get_eps<T>(), T(-1.0));
      }
   });
   cout << setw(15) << "MGS: " << fixed << setprecision(3) << mgs_time / static_cast<double>(ntests) << endl;
   
   A = Matrix<T>(A_copy);
   
   // Test MGS method with re-orthogonalization
   double mgsr_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = mgs<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "MGS R: " << fixed << setprecision(3) << mgsr_time / static_cast<double>(ntests) << endl;
   
   A = Matrix<T>(A_copy);
   
   // Test MGS_V2 method (right-looking)
   double mgs_v2_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = mgs_v2<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "MGS V2: " << fixed << setprecision(3) << mgs_v2_time / static_cast<double>(ntests) << endl;

   A = Matrix<T>(A_copy);
   
   // Test CGS method
   double cgs_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = cgs<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "CGS: " << fixed << setprecision(3) << cgs_time / static_cast<double>(ntests) << endl;

   A = Matrix<T>(A_copy);
   
   // Test CGS2 method (with re-orthogonalization)
   double cgs2_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = cgs2<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "CGS2: " << fixed << setprecision(3) << cgs2_time / static_cast<double>(ntests) << endl;

   A = Matrix<T>(A_copy);

   // Test Hessenberg method
   double hess_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = hessenberg<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "Hessenberg: " << fixed << setprecision(3) << hess_time / static_cast<double>(ntests) << endl;

   A = Matrix<T>(A_copy);
   
   // Test Hessenberg V2 method (customized kernel version)
   double hess_v2_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = hessenberg_v2<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "Hessenberg V2: " << fixed << setprecision(3) << hess_v2_time / static_cast<double>(ntests) << endl;

   A = Matrix<T>(A_copy);
   
   // Test Hessenberg V3 method (GEMM-based version)
   double hess_v3_time = measure_time([&]() {
      for (int i = 0; i < ntests; i++) {
         std::vector<int> skip = hessenberg_v3<T, T, T>(A, Q, R);
      }
   });
   cout << setw(15) << "Hessenberg V3: " << fixed << setprecision(3) << hess_v3_time / static_cast<double>(ntests) << endl;
   cout << endl;
}

int main() {

   int ntests = 5;

   // Initialize CUDA
   CUDAHandler::init();
   
   // Set different matrix sizes
   vector<pair<size_t, size_t>> matrix_sizes = {
      {25000, 200},   // Small size (same as hessenberg_test)
      {50000, 200},   // Medium size
      {50000, 400}   // Large size
   };
   
   cout << "===============================================" << endl;
   cout << "QR Factorization Methods Performance Comparison" << endl;
   cout << "===============================================" << endl;

   warmup();
   
   // Test different sizes and types
   for (const auto& [m, n] : matrix_sizes) {
      compare_qr_methods<double>(m, n, ntests);
      compare_qr_methods<float>(m, n, ntests);
      
      // For half precision, use float as compute type
      cout << "========== Testing half precision (" << m << "x" << n << ") ==========" << endl;
      
      Matrix<__half> A(m, n, Location::kDEVICE);
      Matrix<__half> Q(m, n, Location::kDEVICE);
      Matrix<__half> R(n, n, Location::kHOST);
      
      A.fill_random();
      cout << "Method comparison (execution time, ms):" << endl;

      Matrix<__half> A_copy(A);
      
      double mgs_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = mgs<__half, __half, float>(A, Q, R, get_eps<__half>(), get_eps<__half>(), __float2half(-1.0f));
         }
      });
      cout << setw(15) << "MGS: " << fixed << setprecision(3) << mgs_time / static_cast<double>(ntests) << endl;
      
      A = Matrix<__half>(A_copy);
      
      double mgsr_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = mgs<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "MGS R: " << fixed << setprecision(3) << mgsr_time / static_cast<double>(ntests) << endl;

      A = Matrix<__half>(A_copy);
      
      double mgs_v2_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = mgs_v2<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "MGS V2: " << fixed << setprecision(3) << mgs_v2_time / static_cast<double>(ntests) << endl;

      A = Matrix<__half>(A_copy);
      
      double cgs_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = cgs<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "CGS: " << fixed << setprecision(3) << cgs_time / static_cast<double>(ntests) << endl;
      
      A = Matrix<__half>(A_copy);
      
      double cgs2_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = cgs2<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "CGS2: " << fixed << setprecision(3) << cgs2_time / static_cast<double>(ntests) << endl;

      A = Matrix<__half>(A_copy);
      
      double hess_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = hessenberg<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "Hessenberg: " << fixed << setprecision(3) << hess_time / static_cast<double>(ntests) << endl;

      A = Matrix<__half>(A_copy);
      
      double hess_v2_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = hessenberg_v2<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "Hessenberg V2: " << fixed << setprecision(3) << hess_v2_time / static_cast<double>(ntests) << endl;

      A = Matrix<__half>(A_copy);
      
      double hess_v3_time = measure_time([&]() {
         for (int i = 0; i < ntests; i++) {
            std::vector<int> skip = hessenberg_v3<__half, __half, float>(A, Q, R);
         }
      });
      cout << setw(15) << "Hessenberg V3: " << fixed << setprecision(3) << hess_v3_time / static_cast<double>(ntests) << endl;
      cout << endl;
      
   }
   
   // Cleanup CUDA resources
   CUDAHandler::finalize();
   
   return 0;
}