#include <gtest/gtest.h>
#include "../linalg/factorization/qr.hpp"
#include "../linalg/blas/mvops.hpp"
#include "../containers/vector.hpp"
#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/type_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_fp16.h>

namespace msvd {

// Test environment for QR tests
class QRTestEnvironment : public ::testing::Environment {
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

enum class QRAlgorithm {
   kMGS,
   kMGSV2,
   kCGS,
   kCGS2
};

struct QRAlgorithmCase {
   const char* name;
   QRAlgorithm algorithm;
};

struct MatrixShape {
   size_t rows;
   size_t cols;
};

template<typename T>
T qr_test_value(double value) {
   return static_cast<T>(value);
}

template<>
__half qr_test_value<__half>(double value) {
   return __float2half(static_cast<float>(value));
}

template<typename T>
double qr_test_double(T value) {
   return static_cast<double>(value);
}

template<>
double qr_test_double<__half>(__half value) {
   return static_cast<double>(__half2float(value));
}

template<typename T, typename T_COMPUTE>
void check_qr_shape(
   const QRAlgorithmCase& algorithm_case,
   const MatrixShape& shape,
   const char* precision,
   double orthogonality_tolerance,
   double reconstruction_tolerance
) {
   SCOPED_TRACE(::testing::Message()
      << "algorithm=" << algorithm_case.name
      << ", precision=" << precision
      << ", shape=" << shape.rows << "x" << shape.cols);

   Matrix<T> A(shape.rows, shape.cols, Location::kHOST);
   std::vector<double> original(shape.rows * shape.cols);
   for (size_t j = 0; j < shape.cols; ++j) {
      for (size_t i = 0; i < shape.rows; ++i) {
         // The leading identity keeps every tested matrix well conditioned,
         // while the dense perturbation exercises all rows of every column.
         const double value = (i == j ? 2.0 : 0.0)
            + 0.02 * std::sin(0.37 * static_cast<double>((i + 1) * (j + 1)));
         A(i, j) = qr_test_value<T>(value);
         original[i + j * shape.rows] = qr_test_double(A(i, j));
      }
   }
   A.to_device();

   Matrix<T> Q(shape.rows, shape.cols, Location::kDEVICE);
   Matrix<T> R(shape.cols, shape.cols, Location::kHOST);
   std::vector<int> skip;
   switch (algorithm_case.algorithm) {
      case QRAlgorithm::kMGS:
         skip = mgs<T, T, T_COMPUTE>(A, Q, R);
         break;
      case QRAlgorithm::kMGSV2:
         skip = mgs_v2<T, T, T_COMPUTE>(A, Q, R);
         break;
      case QRAlgorithm::kCGS:
         skip = cgs<T, T, T_COMPUTE>(A, Q, R);
         break;
      case QRAlgorithm::kCGS2:
         skip = cgs2<T, T, T_COMPUTE>(A, Q, R);
         break;
   }

   const cudaError_t sync_status = cudaDeviceSynchronize();
   ASSERT_EQ(sync_status, cudaSuccess) << cudaGetErrorString(sync_status);
   ASSERT_EQ(skip.size(), shape.cols);
   for (size_t j = 0; j < shape.cols; ++j) {
      ASSERT_EQ(skip[j], 0) << "unexpectedly skipped column " << j;
   }

   Q.to_host();
   for (size_t j = 0; j < shape.cols; ++j) {
      for (size_t i = 0; i < shape.rows; ++i) {
         ASSERT_TRUE(std::isfinite(qr_test_double(Q(i, j))))
            << "non-finite Q(" << i << ", " << j << ")";
      }
      for (size_t i = 0; i < shape.cols; ++i) {
         ASSERT_TRUE(std::isfinite(qr_test_double(R(i, j))))
            << "non-finite R(" << i << ", " << j << ")";
      }
   }

   double max_orthogonality_error = 0.0;
   for (size_t j = 0; j < shape.cols; ++j) {
      for (size_t k = 0; k < shape.cols; ++k) {
         double dot_product = 0.0;
         for (size_t i = 0; i < shape.rows; ++i) {
            dot_product += qr_test_double(Q(i, j)) * qr_test_double(Q(i, k));
         }
         const double expected = j == k ? 1.0 : 0.0;
         max_orthogonality_error = std::max(
            max_orthogonality_error, std::abs(dot_product - expected));
      }
   }
   EXPECT_LT(max_orthogonality_error, orthogonality_tolerance);

   double max_reconstruction_error = 0.0;
   for (size_t j = 0; j < shape.cols; ++j) {
      for (size_t i = 0; i < shape.rows; ++i) {
         double reconstructed = 0.0;
         for (size_t k = 0; k < shape.cols; ++k) {
            reconstructed += qr_test_double(Q(i, k)) * qr_test_double(R(k, j));
         }
         max_reconstruction_error = std::max(
            max_reconstruction_error,
            std::abs(reconstructed - original[i + j * shape.rows]));
      }
   }
   EXPECT_LT(max_reconstruction_error, reconstruction_tolerance);
}

} // namespace

// Cover degenerate, square, tall, and CUDA boundary-adjacent dimensions for
// every QR implementation without making the regular test suite expensive.
TEST(QRTest, MultipleShapesCudaRegression) {
   const std::array<QRAlgorithmCase, 4> algorithms{{
      {"MGS", QRAlgorithm::kMGS},
      {"MGS_V2", QRAlgorithm::kMGSV2},
      {"CGS", QRAlgorithm::kCGS},
      {"CGS2", QRAlgorithm::kCGS2}
   }};
   const std::array<MatrixShape, 6> shapes{{
      {1, 1},
      {7, 3},
      {31, 31},
      {33, 7},
      {255, 9},
      {257, 9}
   }};
   const std::array<MatrixShape, 2> mixed_precision_shapes{{
      {37, 5},
      {1025, 3}
   }};

   for (const QRAlgorithmCase& algorithm_case : algorithms) {
      for (const MatrixShape& shape : shapes) {
         ASSERT_NO_FATAL_FAILURE((check_qr_shape<double, double>(
            algorithm_case, shape, "double", 1e-10, 1e-10)));
      }
      for (const MatrixShape& shape : mixed_precision_shapes) {
         ASSERT_NO_FATAL_FAILURE((check_qr_shape<float, float>(
            algorithm_case, shape, "float", 2e-5, 2e-5)));
         ASSERT_NO_FATAL_FAILURE((check_qr_shape<__half, float>(
            algorithm_case, shape, "half/float-compute", 2e-2, 2e-2)));
      }
   }
}

// Test double precision QR
TEST(QRTest, MGS_DoublePrecision) {
   const size_t m = 2000;  // Number of rows
   const size_t n = 200;  // Number of columns
   
   // Create matrices
   Matrix<double> A(m, n, Location::kDEVICE);
   Matrix<double> Q(m, n, Location::kDEVICE);
   Matrix<double> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization
   std::vector<int> skip = mgs<double, double, double>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<double> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   double alpha = 1.0;
   double beta = 0.0;

   gemm<double, double, double>(true, false, alpha, Q, Q, beta, QTQ);
   
   QTQ.to_host();
   
   const double tol = 1e-10;
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            // Skip columns that were marked as skipped
            if (skip[i]) continue;
            EXPECT_NEAR(QTQ(i, j), 1.0, tol);
         } else {
            // Skip if either column was skipped
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(QTQ(i, j), 0.0, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<double> QR(m, n, Location::kDEVICE);
   
   R.to_device();

   // Compute QR using CUBLAS
   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
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
   std::vector<int> skip2 = mgs<double, double, double>(A, Q, R, 1e-12, 1e-12, 1.0/sqrt(2.0));
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         double diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

// Test single precision QR
TEST(QRTest, MGS_SinglePrecision) {
   const size_t m = 2000;
   const size_t n = 200;
   
   // Create matrices
   Matrix<float> A(m, n, Location::kDEVICE);
   Matrix<float> Q(m, n, Location::kDEVICE);
   Matrix<float> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization
   std::vector<int> skip = mgs<float, float, float>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<float> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   float alpha = 1.0f;
   float beta = 0.0f;
   
   gemm<float, float, float>(true, false, alpha, Q, Q, beta, QTQ);
   
   // Copy to host for verification
   QTQ.to_host();
   
   const float tol = 2e-6f;  // Reduced tolerance for single precision
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            if (skip[i]) continue;
            EXPECT_NEAR(QTQ(i, j), 1.0f, tol);
         } else {
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(QTQ(i, j), 0.0f, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<float> QR(m, n, Location::kDEVICE);
   R.to_device();
   
   // Compute QR using CUBLAS
   gemm<float, float, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
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
   std::vector<int> skip2 = mgs<float, float, float>(A, Q, R, 1e-06f, 1e-06f, 1.0f/sqrt(2.0f));
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<float, float, float>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

// Test half precision QR with float compute
TEST(QRTest, MGS_HalfPrecision) {
   const size_t m = 2000;
   const size_t n = 200;
   
   // Create matrices
   Matrix<__half> A(m, n, Location::kDEVICE);
   Matrix<__half> Q(m, n, Location::kDEVICE);
   Matrix<__half> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization
   std::vector<int> skip = mgs<__half, __half, float>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<__half> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   float alpha = 1.0f;
   float beta = 0.0f;
   
   gemm<__half, __half, float>(true, false, alpha, Q, Q, beta, QTQ);
   
   // Copy to host for verification
   QTQ.to_host();
   
   const float tol = 2e-2f;  // Much larger tolerance for half precision
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            if (skip[i]) continue;
            EXPECT_NEAR(__half2float(QTQ(i, j)), 1.0f, tol);
         } else {
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(__half2float(QTQ(i, j)), 0.0f, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<__half> QR(m, n, Location::kDEVICE);
   R.to_device();

   // Compute QR using CUBLAS
   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(__half) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = mgs<__half, __half, float>(A, Q, R, __float2half(1e-02f), __float2half(1e-02f), __float2half(1.0f/sqrt(2.0f)));
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

// Test double precision CGS2
TEST(QRTest, CGS2_DoublePrecision) {
   const size_t m = 2000;  // Number of rows
   const size_t n = 200;  // Number of columns
   
   // Create matrices
   Matrix<double> A(m, n, Location::kDEVICE);
   Matrix<double> Q(m, n, Location::kDEVICE);
   Matrix<double> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization using CGS2
   std::vector<int> skip = cgs2<double, double, double>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<double> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   double alpha = 1.0;
   double beta = 0.0;

   gemm<double, double, double>(true, false, alpha, Q, Q, beta, QTQ);
   
   // Copy to host for verification
   QTQ.to_host();
   
   const double tol = 1e-10;
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            // Skip columns that were marked as skipped
            if (skip[i]) continue;
            EXPECT_NEAR(QTQ(i, j), 1.0, tol);
         } else {
            // Skip if either column was skipped
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(QTQ(i, j), 0.0, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<double> QR(m, n, Location::kDEVICE);

   R.to_device();

   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   double max_diff = 0.0;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         double diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(double) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = cgs2<double, double, double>(A, Q, R, 1e-12);
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         double diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

// CGS2 must apply the second projection to Q, not only accumulate it in R.
TEST(QRTest, CGS2_ReorthogonalizesNearlyDependentColumns) {
   const size_t m = 3;
   const size_t n = 2;
   const double delta = 1e-8;

   Matrix<double> A(m, n, Location::kHOST);
   A.fill(0.0);
   A(0, 0) = 1.0;
   A(1, 0) = delta;
   A(0, 1) = 1.0;
   A(2, 1) = delta;
   A.to_device();

   Matrix<double> Q(m, n, Location::kDEVICE);
   Matrix<double> R(n, n, Location::kHOST);
   const std::vector<int> skip = cgs2<double, double, double>(A, Q, R);

   ASSERT_EQ(skip.size(), n);
   EXPECT_EQ(skip[0], 0);
   EXPECT_EQ(skip[1], 0);

   Matrix<double> QTQ(n, n, Location::kDEVICE);
   gemm<double, double, double>(true, false, 1.0, Q, Q, 0.0, QTQ);
   QTQ.to_host();
   EXPECT_NEAR(QTQ(0, 0), 1.0, 1e-14);
   EXPECT_NEAR(QTQ(1, 1), 1.0, 1e-14);
   EXPECT_NEAR(QTQ(0, 1), 0.0, 1e-12);
   EXPECT_NEAR(QTQ(1, 0), 0.0, 1e-12);

   Matrix<double> QR(m, n, Location::kDEVICE);
   R.to_device();
   gemm<double, double, double>(false, false, 1.0, Q, R, 0.0, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         EXPECT_NEAR(QR(i, j), A(i, j), 1e-14);
      }
   }
}

// Test single precision CGS2
TEST(QRTest, CGS2_SinglePrecision) {
   const size_t m = 2000;
   const size_t n = 200;
   
   // Create matrices
   Matrix<float> A(m, n, Location::kDEVICE);
   Matrix<float> Q(m, n, Location::kDEVICE);
   Matrix<float> R(n, n, Location::kHOST); // R on host for CGS2
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization using CGS2
   std::vector<int> skip = cgs2<float, float, float>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<float> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   float alpha = 1.0f;
   float beta = 0.0f;
   
   gemm<float, float, float>(true, false, alpha, Q, Q, beta, QTQ);
   
   // Copy to host for verification
   QTQ.to_host();
   
   const float tol = 2e-6f;  // Reduced tolerance for single precision
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            if (skip[i]) continue;
            EXPECT_NEAR(QTQ(i, j), 1.0f, tol);
         } else {
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(QTQ(i, j), 0.0f, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<float> QR(m, n, Location::kDEVICE);

   R.to_device();

   gemm<float, float, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(float) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = cgs2<float, float, float>(A, Q, R, 5e-06f);
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<float, float, float>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

// Test half precision CGS2 with float compute
TEST(QRTest, CGS2_HalfPrecision) {
   const size_t m = 2000;
   const size_t n = 200;
   
   // Create matrices
   Matrix<__half> A(m, n, Location::kDEVICE);
   Matrix<__half> Q(m, n, Location::kDEVICE);
   Matrix<__half> R(n, n, Location::kHOST); // R on host for CGS2
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization using CGS2
   std::vector<int> skip = cgs2<__half, __half, float>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<__half> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   float alpha = 1.0f;
   float beta = 0.0f;
   
   gemm<__half, __half, float>(true, false, alpha, Q, Q, beta, QTQ);
   
   // Copy to host for verification
   QTQ.to_host();
   
   const float tol = 2e-2f;  // Much larger tolerance for half precision
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            if (skip[i]) continue;
            EXPECT_NEAR(__half2float(QTQ(i, j)), 1.0f, tol);
         } else {
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(__half2float(QTQ(i, j)), 0.0f, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<__half> QR(m, n, Location::kDEVICE);

   R.to_device();

   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   float max_diff = 0.0f;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(__half) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = cgs2<__half, __half, float>(A, Q, R, __float2half(5e-02f));
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<__half, __half, float>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         float diff = std::abs(__half2float(A(i, j)) - __half2float(QR(i, j)));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

// Test double precision MGS_V2
TEST(QRTest, MGS_V2_DoublePrecision) {
   const size_t m = 2000;  // Number of rows
   const size_t n = 200;  // Number of columns
   
   // Create matrices
   Matrix<double> A(m, n, Location::kDEVICE);
   Matrix<double> Q(m, n, Location::kDEVICE);
   Matrix<double> R(n, n, Location::kHOST);
   
   // Fill A with random values
   A.fill_random();
   
   // Perform QR factorization using CGS2
   std::vector<int> skip = mgs_v2<double, double, double>(A, Q, R);
   
   // Verify Q^T * Q = I
   Matrix<double> QTQ(n, n, Location::kDEVICE);
   
   // Compute Q^T * Q
   double alpha = 1.0;
   double beta = 0.0;

   gemm<double, double, double>(true, false, alpha, Q, Q, beta, QTQ);
   
   // Copy to host for verification
   QTQ.to_host();
   
   const double tol = 1e-10;
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
         if (i == j) {
            // Skip columns that were marked as skipped
            if (skip[i]) continue;
            EXPECT_NEAR(QTQ(i, j), 1.0, tol);
         } else {
            // Skip if either column was skipped
            if (skip[i] || skip[j]) continue;
            EXPECT_NEAR(QTQ(i, j), 0.0, tol);
         }
      }
   }
   
   // Verify A = Q * R
   Matrix<double> QR(m, n, Location::kDEVICE);

   R.to_device();

   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   
   // Copy results to host for verification
   A.to_host();
   QR.to_host();
   
   double max_diff = 0.0;
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         double diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);

   // Test with matrix with linearly dependent columns
   A.fill_random();
   A.to_device();
   R.to_host();
   CUDA_CHECK(cudaMemcpy(&A(0, 3), &A(0, 0), 3 * sizeof(double) * m, cudaMemcpyDeviceToDevice));
   std::vector<int> skip2 = mgs_v2<double, double, double>(A, Q, R, 1e-12);
   for (size_t i = 0; i < n; i++) {
      if (i != 3 && i != 4 && i != 5) {
         EXPECT_EQ(skip2[i], 0);
      } else {
         EXPECT_EQ(skip2[i], 1);
      }
   }
   R.to_device();
   QR.to_device();
   gemm<double, double, double>(false, false, alpha, Q, R, beta, QR);
   A.to_host();
   QR.to_host();
   for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++) {
         double diff = std::abs(A(i, j) - QR(i, j));
         max_diff = std::max(max_diff, diff);
      }
   }
   EXPECT_LT(max_diff, tol);
}

} // namespace msvd

// Set up the environment
int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   ::testing::AddGlobalTestEnvironment(new msvd::QRTestEnvironment);
   return RUN_ALL_TESTS();
}
