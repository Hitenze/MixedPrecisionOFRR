#include <gtest/gtest.h>
#include "../linalg/factorization/qr.hpp"
#include "../linalg/blas/mvops.hpp"
#include "../containers/vector.hpp"
#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/type_utils.hpp"
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