#include "cuda_event_timer.hpp"

#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/type_utils.hpp"
#include "../linalg/blas/mvops.hpp"
#include "../linalg/factorization/hessenberg.hpp"
#include "../linalg/factorization/qr.hpp"

#include <cuda_fp16.h>

#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

void warmup() {
   Matrix<double> A(1000, 1000, Location::kDEVICE);
   Matrix<double> B(1000, 1000, Location::kDEVICE);
   A.fill_random();

   for (int i = 0; i < 10; ++i) {
      throw_if_error(
         msvd::gemm<double, double, double>(
            true,
            false,
            msvd::get_one<double>(),
            A,
            A,
            msvd::get_zero<double>(),
            B
         ),
         "warmup GEMM"
      );
   }

   // Do not let initialization or warmup work leak into the first measurement.
   CudaEventTimer::synchronize_device();
}

template<typename Function>
void benchmark_method(
   const char* name,
   int repetitions,
   CudaEventTimer& timer,
   Function&& function
) {
   if (repetitions <= 0) {
      throw std::invalid_argument("benchmark repetitions must be positive");
   }

   const double total_ms = timer.measure_ms([&]() {
      for (int i = 0; i < repetitions; ++i) {
         std::invoke(function);
      }
   });

   std::cout << std::setw(18) << (std::string(name) + ": ")
             << std::fixed << std::setprecision(3)
             << total_ms / static_cast<double>(repetitions) << '\n';
}

template<typename T, typename T_COMPUTE>
void compare_qr_methods(
   std::size_t m,
   std::size_t n,
   int repetitions,
   CudaEventTimer& timer
) {
   std::cout << "========== Testing " << precision_name<T, T_COMPUTE>()
             << " precision (" << m << 'x' << n << ") ==========\n";

   Matrix<T> A(m, n, Location::kDEVICE);
   Matrix<T> Q(m, n, Location::kDEVICE);
   Matrix<T> R(n, n, Location::kHOST);
   A.fill_random();

   std::cout << "Method comparison (average CUDA-event elapsed time per run, ms):\n";

   benchmark_method("MGS", repetitions, timer, [&]() {
      (void)msvd::mgs<T, T, T_COMPUTE>(
         A,
         Q,
         R,
         msvd::get_eps<T>(),
         msvd::get_eps<T>(),
         msvd::get_negone<T>()
      );
   });

   benchmark_method("MGS R", repetitions, timer, [&]() {
      (void)msvd::mgs<T, T, T_COMPUTE>(A, Q, R);
   });

   benchmark_method("MGS V2", repetitions, timer, [&]() {
      (void)msvd::mgs_v2<T, T, T_COMPUTE>(A, Q, R);
   });

   benchmark_method("CGS", repetitions, timer, [&]() {
      (void)msvd::cgs<T, T, T_COMPUTE>(A, Q, R);
   });

   benchmark_method("CGS2", repetitions, timer, [&]() {
      (void)msvd::cgs2<T, T, T_COMPUTE>(A, Q, R);
   });

   benchmark_method("Hessenberg", repetitions, timer, [&]() {
      (void)msvd::hessenberg<T, T, T_COMPUTE>(A, Q, R);
   });

   benchmark_method("Hessenberg V2", repetitions, timer, [&]() {
      (void)msvd::hessenberg_v2<T, T, T_COMPUTE>(A, Q, R);
   });

   benchmark_method("Hessenberg V3", repetitions, timer, [&]() {
      (void)msvd::hessenberg_v3<T, T, T_COMPUTE>(A, Q, R);
   });

   std::cout << '\n';
}

void run_benchmarks() {
   constexpr int repetitions = 5;
   const std::vector<std::pair<std::size_t, std::size_t>> matrix_sizes = {
      {25000, 200},
      {50000, 200},
      {50000, 400}
   };

   CudaEventTimer timer;

   std::cout << "===============================================\n"
             << "QR Factorization Methods Performance Comparison\n"
             << "===============================================\n";

   warmup();

   for (const auto& [m, n] : matrix_sizes) {
      compare_qr_methods<double, double>(m, n, repetitions, timer);
      compare_qr_methods<float, float>(m, n, repetitions, timer);
      compare_qr_methods<__half, float>(m, n, repetitions, timer);
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

      std::cerr << "ex00 failed: " << error.what() << '\n';
      return EXIT_FAILURE;
   }
}
