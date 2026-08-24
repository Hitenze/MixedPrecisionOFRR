#include <gtest/gtest.h>
#include "../linalg/factorization/hessenberg.hpp"
#include "../linalg/blas/mvops.hpp"
#include "../containers/vector.hpp"
#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/type_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include <cuda_fp16.h>

namespace msvd {

// Test environment for Hessenberg tests
class HessenbergTestEnvironment : public ::testing::Environment {
public:
   void SetUp() override {
      // Initialize CUDA handlers
      CUDAHandler::init();
   }
   
   void TearDown() override {
      // Clean up CUDA handlers
      CUDAHandler::finalize();
   }
};

namespace {

enum class HessenbergVariant {
   kCublas,
   kCustomKernel,
   kMgsLike
};

struct MatrixShape {
   size_t rows;
   size_t cols;
   const char* name;
};

const char* variant_name(HessenbergVariant variant) {
   switch (variant) {
      case HessenbergVariant::kCublas:
         return "hessenberg";
      case HessenbergVariant::kCustomKernel:
         return "hessenberg_v2";
      case HessenbergVariant::kMgsLike:
         return "hessenberg_v3";
   }
   return "unknown";
}

template<typename T>
const char* scalar_name() {
   if constexpr (std::is_same_v<T, double>) {
      return "double";
   } else if constexpr (std::is_same_v<T, float>) {
      return "float";
   } else {
      return "half";
   }
}

template<typename T>
T scalar_from_double(double value) {
   if constexpr (std::is_same_v<T, __half>) {
      return __float2half(static_cast<float>(value));
   } else {
      return static_cast<T>(value);
   }
}

template<typename T>
double scalar_to_double(T value) {
   if constexpr (std::is_same_v<T, __half>) {
      return static_cast<double>(__half2float(value));
   } else {
      return static_cast<double>(value);
   }
}

template<typename T, typename TCompute>
std::vector<int> factorize(
   HessenbergVariant variant,
   const Matrix<T>& A,
   Matrix<T>& Q,
   Matrix<T>& R
) {
   switch (variant) {
      case HessenbergVariant::kCublas:
         return hessenberg<T, T, TCompute>(A, Q, R);
      case HessenbergVariant::kCustomKernel:
         return hessenberg_v2<T, T, TCompute>(A, Q, R);
      case HessenbergVariant::kMgsLike:
         return hessenberg_v3<T, T, TCompute>(A, Q, R);
   }
   return {};
}

template<typename T, typename TCompute>
void check_shape(
   HessenbergVariant variant,
   const MatrixShape& shape,
   double relative_tolerance
) {
   const std::string trace_name =
      std::string(scalar_name<T>()) + "/" + variant_name(variant) + "/"
      + shape.name;
   const ::testing::ScopedTrace trace(__FILE__, __LINE__, trace_name);

   Matrix<T> A(shape.rows, shape.cols, Location::kHOST);
   for (size_t j = 0; j < shape.cols; ++j) {
      for (size_t i = 0; i < shape.rows; ++i) {
         const double row = static_cast<double>(i + 1);
         const double col = static_cast<double>(j + 1);
         const double value = std::sin(0.173 * row * (col + 1.0))
                              + std::cos(0.117 * (row + 2.0) * col)
                              + (i == j ? 3.0 : 0.0);
         A(i, j) = scalar_from_double<T>(value);
      }
   }
   A.to_device();

   Matrix<T> Q(shape.rows, shape.cols, Location::kDEVICE);
   Matrix<T> R(shape.cols, shape.cols, Location::kHOST);
   const std::vector<int> skip = factorize<T, TCompute>(variant, A, Q, R);

   ASSERT_EQ(cudaGetLastError(), cudaSuccess);
   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
   ASSERT_EQ(skip.size(), shape.cols);
   for (size_t j = 0; j < shape.cols; ++j) {
      EXPECT_EQ(skip[j], 0) << "unexpected skipped column " << j;
   }
   EXPECT_EQ(check_special_values(Q), MSVDStatus::kSuccess);

   Matrix<T> QR(shape.rows, shape.cols, Location::kDEVICE);
   R.to_device();
   EXPECT_EQ(check_special_values(R), MSVDStatus::kSuccess);
   const TCompute alpha = static_cast<TCompute>(1.0);
   const TCompute beta = static_cast<TCompute>(0.0);
   ASSERT_EQ(
      (gemm<T, T, TCompute>(false, false, alpha, Q, R, beta, QR)),
      MSVDStatus::kSuccess
   );
   ASSERT_EQ(cudaGetLastError(), cudaSuccess);
   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

   A.to_host();
   QR.to_host();
   double max_abs_a = 0.0;
   double max_abs_error = 0.0;
   for (size_t j = 0; j < shape.cols; ++j) {
      for (size_t i = 0; i < shape.rows; ++i) {
         const double a_value = scalar_to_double(A(i, j));
         const double qr_value = scalar_to_double(QR(i, j));
         max_abs_a = std::max(max_abs_a, std::abs(a_value));
         max_abs_error = std::max(
            max_abs_error,
            std::abs(a_value - qr_value)
         );
      }
   }
   EXPECT_LE(
      max_abs_error,
      relative_tolerance * std::max(1.0, max_abs_a)
   );
}

} // namespace

// Sweep the three implementations over full-rank, boundary-relevant inputs.
// The double cases bracket 256- and 1024-thread boundaries; float and half
// retain a small odd shape plus a 1024-boundary case.
TEST(HessenbergTest, ShapeSweep) {
   constexpr std::array<HessenbergVariant, 3> variants = {
      HessenbergVariant::kCublas,
      HessenbergVariant::kCustomKernel,
      HessenbergVariant::kMgsLike
   };
   constexpr std::array<MatrixShape, 10> double_shapes = {{
      {1, 1, "minimum_1x1"},
      {7, 7, "square_7x7"},
      {65, 4, "tall_skinny_65x4"},
      {255, 3, "block_255x3"},
      {256, 3, "block_256x3"},
      {257, 3, "block_257x3"},
      {1023, 3, "block_1023x3"},
      {1024, 3, "block_1024x3"},
      {1025, 3, "block_1025x3"},
      {1025, 5, "non_aligned_1025x5"}
   }};
   constexpr std::array<MatrixShape, 2> reduced_precision_shapes = {{
      {37, 5, "odd_37x5"},
      {1025, 3, "block_1025x3"}
   }};

   for (const HessenbergVariant variant : variants) {
      for (const MatrixShape& shape : double_shapes) {
         check_shape<double, double>(variant, shape, 1e-10);
      }
      for (const MatrixShape& shape : reduced_precision_shapes) {
         check_shape<float, float>(variant, shape, 2e-5);
         check_shape<__half, float>(variant, shape, 2e-2);
      }
   }
}

// Exercise the last remaining-column update, where both GEMM views end at the
// allocation boundary. This small case is also suitable for compute-sanitizer.
TEST(HessenbergTest, TwoColumnBoundary) {
   const size_t m = 5;
   const size_t n = 2;

   Matrix<double> A(m, n, Location::kHOST);
   A(0, 0) = 4.0;
   A(1, 0) = 1.0;
   A(2, 0) = -2.0;
   A(3, 0) = 0.5;
   A(4, 0) = 3.0;
   A(0, 1) = 1.0;
   A(1, 1) = -3.0;
   A(2, 1) = 2.0;
   A(3, 1) = 5.0;
   A(4, 1) = -1.0;
   A.to_device();

   Matrix<double> Q(m, n, Location::kDEVICE);
   Matrix<double> R(n, n, Location::kHOST);
   const std::vector<int> skip = hessenberg<double, double, double>(A, Q, R);
   CUDA_CHECK(cudaDeviceSynchronize());

   ASSERT_EQ(skip.size(), n);
   EXPECT_EQ(skip[0], 0);
   EXPECT_EQ(skip[1], 0);
   EXPECT_DOUBLE_EQ(R(0, 0), 4.0);
   EXPECT_DOUBLE_EQ(R(0, 1), 1.0);
   EXPECT_DOUBLE_EQ(R(1, 1), 4.875);

   Matrix<double> QR(m, n, Location::kDEVICE);
   R.to_device();
   double alpha = 1.0;
   double beta = 0.0;
   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   CUDA_CHECK(cudaDeviceSynchronize());

   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         EXPECT_NEAR(QR(i, j), A(i, j), 1e-12);
      }
   }

}

// Test double precision Hessenberg QR
TEST(HessenbergTest, DoublePrecision) {
   const size_t m = 2000;  // Number of rows
   const size_t n = 200;   // Number of columns
   
   // Create matrices
   Matrix<double> A(m, n, Location::kDEVICE);
   Matrix<double> Q(m, n, Location::kDEVICE);
   Matrix<double> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform Hessenberg QR factorization
   std::vector<int> skip = hessenberg<double, double, double>(A, Q, R);

   MSVDStatus has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   // Verify A = Q * R
   Matrix<double> QR(m, n, Location::kDEVICE);
   R.to_device();
   
   // Compute QR using CUBLAS
   double alpha = 1.0;
   double beta = 0.0;
   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   const double tol = 1e-10;
   double max_diff = 0.0;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         double diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
   
   // Next test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(double) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = hessenberg<double, double, double>(A, Q, R, 1e-12);

   has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
}

// Test single precision Hessenberg QR
TEST(HessenbergTest, SinglePrecision) {
   const size_t m = 2000;
   const size_t n = 200;
   
   // Create matrices
   Matrix<float> A(m, n, Location::kDEVICE);
   Matrix<float> Q(m, n, Location::kDEVICE);
   Matrix<float> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform Hessenberg QR factorization
   std::vector<int> skip = hessenberg<float, float, float>(A, Q, R);
   
   MSVDStatus has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   // Verify A = Q * R
   Matrix<float> QR(m, n, Location::kDEVICE);
   R.to_device();
   
   // Compute QR using CUBLAS
   float alpha = 1.0f;
   float beta = 0.0f;
   gemm<float, float, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   const float tol = 1e-5f;
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Next test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(float) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = hessenberg<float, float, float>(A, Q, R, 1e-06f);

   has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   
}

// Test half precision Hessenberg QR
TEST(HessenbergTest, HalfPrecision) {
   const size_t m = 2000;
   const size_t n = 100;
   
   // Create matrices
   Matrix<__half> A(m, n, Location::kDEVICE);
   Matrix<__half> Q(m, n, Location::kDEVICE);
   Matrix<__half> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform Hessenberg QR factorization
   std::vector<int> skip = hessenberg<__half, __half, float>(A, Q, R);
   
   MSVDStatus has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   // Verify A = Q * R
   Matrix<__half> QR(m, n, Location::kDEVICE);
   R.to_device();
   
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);

   // Compute QR using CUBLAS
   float alpha = 1.0f;
   float beta = 0.0f;
   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   const float tol = 5e-2f;  // Larger tolerance for half precision
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Next test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(__half) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = hessenberg<__half, __half, float>(A, Q, R, 1e-02f);

   has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   R.to_device();
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
}

// Test half precision Hessenberg QR
TEST(HessenbergTest, HalfPrecisionV2) {
   const size_t m = 2000;
   const size_t n = 100;
   
   // Create matrices
   Matrix<__half> A(m, n, Location::kDEVICE);
   Matrix<__half> Q(m, n, Location::kDEVICE);
   Matrix<__half> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform Hessenberg QR factorization
   std::vector<int> skip = hessenberg_v2<__half, __half, float>(A, Q, R);
   
   MSVDStatus has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   // Verify A = Q * R
   Matrix<__half> QR(m, n, Location::kDEVICE);
   R.to_device();
   
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);

   // Compute QR using CUBLAS
   float alpha = 1.0f;
   float beta = 0.0f;
   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   const float tol = 5e-2f;  // Larger tolerance for half precision
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Next test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(__half) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = hessenberg_v2<__half, __half, float>(A, Q, R, 1e-02f);

   has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   R.to_device();
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
}

// Test half precision Hessenberg QR
TEST(HessenbergTest, HalfPrecisionV3) {
   const size_t m = 2000;
   const size_t n = 100;
   
   // Create matrices
   Matrix<__half> A(m, n, Location::kDEVICE);
   Matrix<__half> Q(m, n, Location::kDEVICE);
   Matrix<__half> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform Hessenberg QR factorization
   std::vector<int> skip = hessenberg_v3<__half, __half, float>(A, Q, R);
   
   MSVDStatus has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   // Verify A = Q * R
   Matrix<__half> QR(m, n, Location::kDEVICE);
   R.to_device();
   
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);

   // Compute QR using CUBLAS
   float alpha = 1.0f;
   float beta = 0.0f;
   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   const float tol = 5e-2f;  // Larger tolerance for half precision
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Next test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(__half) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = hessenberg_v3<__half, __half, float>(A, Q, R, 1e-02f);

   has_nan_inf = check_special_values(Q);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   R.to_device();
   has_nan_inf = check_special_values(R);
   EXPECT_EQ(has_nan_inf, MSVDStatus::kSuccess);
   
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
}

} // namespace msvd

// Main function
int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   ::testing::AddGlobalTestEnvironment(new msvd::HessenbergTestEnvironment);
   return RUN_ALL_TESTS();
}
