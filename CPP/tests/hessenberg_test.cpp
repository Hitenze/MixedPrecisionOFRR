#include <gtest/gtest.h>
#include "../linalg/factorization/hessenberg.hpp"
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