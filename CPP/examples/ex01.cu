#include "cuda_event_timer.hpp"

#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/type_utils.hpp"
#include "../linalg/blas/mvops.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using msvd::CUDAHandler;
using msvd::Location;
using msvd::Matrix;
using msvd::examples::CudaEventTimer;
using msvd::examples::throw_if_error;

template<typename T, typename T_COMPUTE>
std::string precision_name() {
   if constexpr (std::is_same_v<T, double>) {
      return "double";
   } else if constexpr (std::is_same_v<T, float>) {
      return "float";
   } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<T_COMPUTE, float>) {
      return "half (FP32 compute)";
   } else {
      return "unknown";
   }
}

template<typename T, typename T_COMPUTE>
void test_matrix_multiplication(
   std::size_t m,
   std::size_t n,
   std::size_t k,
   CudaEventTimer& timer
) {
   constexpr int warmup_iterations = 5;
   constexpr int measured_iterations = 10;

   std::cout << "========== Testing " << precision_name<T, T_COMPUTE>()
             << " matrix multiplication ==========\n"
             << "Matrix dimensions: A(" << m << 'x' << k << ") * B("
             << k << 'x' << n << ") = C(" << m << 'x' << n << ")\n";

   Matrix<T> A(m, k, Location::kDEVICE);
   Matrix<T> B(k, n, Location::kDEVICE);
   Matrix<T> C(m, n, Location::kDEVICE);
   A.fill_random();
   B.fill_random();

   const T_COMPUTE alpha = msvd::get_one<T_COMPUTE>();
   const T_COMPUTE beta = msvd::get_zero<T_COMPUTE>();
   auto run_gemm = [&]() {
      throw_if_error(
         msvd::gemm<T, T, T_COMPUTE>(false, false, alpha, A, B, beta, C),
         "GEMM"
      );
   };

   std::cout << "Warming up...\n";
   for (int i = 0; i < warmup_iterations; ++i) {
      run_gemm();
   }
   // Matrix initialization and all warmup GEMMs must finish before timing starts.
   CudaEventTimer::synchronize_device();

   std::cout << "Performing measurements...\n";
   std::vector<double> times_ms;
   times_ms.reserve(measured_iterations);
   for (int i = 0; i < measured_iterations; ++i) {
      times_ms.push_back(timer.measure_ms(run_gemm));
   }

   const double total_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
   const double average_ms = total_ms / static_cast<double>(times_ms.size());
   const auto [minimum, maximum] = std::minmax_element(times_ms.begin(), times_ms.end());

   // A GEMM performs approximately 2*m*n*k floating-point operations.
   const double operations = 2.0 * static_cast<double>(m)
                           * static_cast<double>(n)
                           * static_cast<double>(k);
   const double gflops = operations / (average_ms * 1.0e6);

   std::cout << "Performance results (CUDA event time):\n"
             << "  Iterations:   " << measured_iterations << '\n'
             << "  Average time: " << std::fixed << std::setprecision(3)
             << average_ms << " ms\n"
             << "  Minimum time: " << *minimum << " ms\n"
             << "  Maximum time: " << *maximum << " ms\n"
             << "  Performance:  " << std::setprecision(2) << gflops
             << " GFLOPS\n\n";
}

void run_benchmarks() {
   const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> matrix_sizes = {
      {1024, 1024, 1024},
      {2048, 2048, 2048},
      {4096, 4096, 4096},
      {4096, 1024, 2048}
   };

   CudaEventTimer timer;

   std::cout << "===============================================\n"
             << "Matrix Multiplication (GEMM) Performance Test\n"
             << "===============================================\n";

   for (const auto& [m, n, k] : matrix_sizes) {
      test_matrix_multiplication<double, double>(m, n, k, timer);
      test_matrix_multiplication<float, float>(m, n, k, timer);
      test_matrix_multiplication<__half, float>(m, n, k, timer);
      std::cout << "-----------------------------------------------\n";
   }
}

} // namespace

int main() {
   bool cuda_initialized = false;

   try {
      CUDAHandler::init();
      cuda_initialized = true;
      run_benchmarks();

      cuda_initialized = false;
      CUDAHandler::finalize();
      return EXIT_SUCCESS;
   } catch (const std::exception& error) {
      if (cuda_initialized) {
         try {
            cuda_initialized = false;
            CUDAHandler::finalize();
         } catch (const std::exception& cleanup_error) {
            std::cerr << "CUDA cleanup failed: " << cleanup_error.what() << '\n';
         }
      }

      std::cerr << "ex01 failed: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
}
