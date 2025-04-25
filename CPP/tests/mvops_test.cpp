#include <gtest/gtest.h>
#include "../linalg/blas/mvops.hpp"
#include "../containers/vector.hpp"
#include "../containers/matrix.hpp"
#include "../core/utils/cuda_handler.hpp"
#include "../core/utils/error_handling.hpp"
#include <iostream>
#include <cuda_fp16.h>
#include <limits>

namespace msvd {

// Global setup for tests
class MvopsTestEnvironment : public ::testing::Environment {
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

// Test the unified dot function
TEST(MvopsTest, UnifiedDot) {
   const int n = 5;
   
   // Test double precision on GPU
   {
      Vector<double> x(n, Location::kHOST);
      Vector<double> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1);
         y[i] = static_cast<double>(i + 1);
      }
      
      // Copy vectors to device
      x.to_device();
      y.to_device();
      
      // Calculate dot product
      double result;
      MSVDStatus status = dot<double, double, double>(x, y, result);
      
      // Expected result: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(result, 55.0);
   }
   
   // Test double precision on CPU
   {
      // Create host vectors
      Vector<double> x(n, Location::kHOST);
      Vector<double> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1);
         y[i] = static_cast<double>(i + 1);
      }
      
      // Calculate dot product
      double result;
      MSVDStatus status = dot<double, double, double>(x, y, result);
      
      // Expected result: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(result, 55.0);
   }
   
   // Test single precision on GPU
   {
      // Create device vectors
      Vector<float> x(n, Location::kHOST);
      Vector<float> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i + 1);
         y[i] = static_cast<float>(i + 1);
      }
      
      // Copy vectors to device
      x.to_device();
      y.to_device();
      
      // Calculate dot product
      float result;
      MSVDStatus status = dot<float, float, float>(x, y, result);
      
      // Expected result: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(result, 55.0f);
   }
   
   // Test half precision on GPU with mixed types
   {
      // Create device vectors
      Vector<__half> x(n, Location::kHOST);
      Vector<__half> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = __float2half(static_cast<float>(i + 1));
         y[i] = __float2half(static_cast<float>(i + 1));
      }
      
      // Copy data to device vectors
      x.to_device();
      y.to_device();
      
      // Calculate dot product with output in FP16
      __half result16;
      MSVDStatus status2 = dot<__half, __half, float>(x, y, result16);
      
      // Expected result: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
      EXPECT_EQ(status2, MSVDStatus::kSuccess);
      EXPECT_NEAR(__half2float(result16), 55.0f, 0.1f);
   }
   
   // Test error cases: vectors with different locations
   {
      Vector<float> x(n, Location::kHOST);
      Vector<float> y(n, Location::kDEVICE);
      
      // Calculate dot product - this should throw an exception
      float result;
      
      try {
         dot<float, float, float>(x, y, result);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Both vectors must be in the same location (CPU or GPU)");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }
   
   // Test vectors of different sizes
   {
      // Create host vectors of different sizes
      Vector<float> x(n, Location::kHOST);
      Vector<float> y(n + 2, Location::kHOST);
      
      
      // Calculate dot product - should throw exception
      float result;
      try {
         MSVDStatus status = dot<float, float, float>(x, y, result);
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Both vectors must be of the same size");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }
   
   // Test empty vectors on CPU
   {
      // Create empty vectors
      Vector<float> x(0, Location::kHOST);
      Vector<float> y(0, Location::kHOST);
      
      // Calculate dot product with empty vectors - should return 0
      float result = 1.0f;  // Initialize with non-zero value
      MSVDStatus status = dot<float, float, float>(x, y, result);
      
      // Should return success and set result to 0
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(result, 0.0f);
   }
   
   // Test empty vectors on GPU
   {
      // Create empty vectors on GPU
      Vector<double> x(0, Location::kDEVICE);
      Vector<double> y(0, Location::kDEVICE);
      
      // Calculate dot product with empty vectors - should return 0
      double result = 1.0;  // Initialize with non-zero value
      MSVDStatus status = dot<double, double, double>(x, y, result);
      
      // Should return success and set result to 0
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(result, 0.0);
   }
   
   // Test half precision empty vectors on GPU
   {
      // Create empty vectors on GPU
      Vector<__half> x(0, Location::kDEVICE);
      Vector<__half> y(0, Location::kDEVICE);
      
      // Calculate dot product with empty vectors - should return 0
      __half result = __float2half(1.0f);  // Initialize with non-zero value
      MSVDStatus status = dot<__half, __half, float>(x, y, result);
      
      // Should return success and set result to 0
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(__half2float(result), 0.0f);
   }
}

// Test the gemv function
TEST(MvopsTest, Gemv) {
   const int m = 3; // Number of rows
   const int n = 2; // Number of columns
   
   // Test double precision on GPU
   {
      // Create a 3x2 matrix on host
      Matrix<double> A(m, n, Location::kHOST);
      // Initialize matrix in column-major format
      A(0, 0) = 1.0; A(1, 0) = 2.0; A(2, 0) = 3.0; // First column
      A(0, 1) = 4.0; A(1, 1) = 5.0; A(2, 1) = 6.0; // Second column
      
      // Copy to device
      A.to_device();
      
      // Create vectors
      Vector<double> x(n, Location::kHOST);   // Vector with n elements
      Vector<double> y(m, Location::kHOST);   // Vector with m elements
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1);  // [1, 2]
      }
      
      for (int i = 0; i < m; i++) {
         y[i] = 0.0;  // Initialize y to zeros
      }
      
      // Copy vectors to device
      x.to_device();
      y.to_device();
      
      // Compute y = alpha * A * x + beta * y with alpha=1, beta=0
      double alpha = 1.0;
      double beta = 0.0;
      MSVDStatus status = gemv<double, double, double>(false, alpha, A, x, beta, y);
      
      // Check status
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Copy result back to host
      y.to_host();
      
      // Expected result: y = A * x = [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2] = [9, 12, 15]
      EXPECT_DOUBLE_EQ(y[0], 9.0);
      EXPECT_DOUBLE_EQ(y[1], 12.0);
      EXPECT_DOUBLE_EQ(y[2], 15.0);
   }
   
   // Test single precision on GPU
   {
      // Create a 3x2 matrix on host
      Matrix<float> A(m, n, Location::kHOST);
      
      // Initialize matrix in column-major format
      A(0, 0) = 1.0f; A(1, 0) = 2.0f; A(2, 0) = 3.0f; // First column
      A(0, 1) = 4.0f; A(1, 1) = 5.0f; A(2, 1) = 6.0f; // Second column
      
      // Copy to device
      A.to_device();
      
      // Create vectors
      Vector<float> x(n, Location::kHOST);   // Vector with n elements
      Vector<float> y(m, Location::kHOST);   // Vector with m elements
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i + 1);  // [1, 2]
      }
      
      for (int i = 0; i < m; i++) {
         y[i] = 0.0f;  // Initialize y to zeros
      }
      
      // Copy vectors to device
      x.to_device();
      y.to_device();
      
      // Compute y = alpha * A * x + beta * y with alpha=1, beta=0
      float alpha = 1.0f;
      float beta = 0.0f;
      MSVDStatus status = gemv<float, float, float>(false, alpha, A, x, beta, y);
      
      // Check status
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Copy result back to host
      y.to_host();
      
      // Expected result: y = A * x = [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2] = [9, 12, 15]
      EXPECT_FLOAT_EQ(y[0], 9.0f);
      EXPECT_FLOAT_EQ(y[1], 12.0f);
      EXPECT_FLOAT_EQ(y[2], 15.0f);
   }
   
   // Test half precision with float compute on GPU
   {
      // Create a 3x2 matrix on host
      Matrix<__half> A(m, n, Location::kHOST);
      
      // Initialize matrix in column-major format
      A(0, 0) = __float2half(1.0f); A(1, 0) = __float2half(2.0f); A(2, 0) = __float2half(3.0f); // First column
      A(0, 1) = __float2half(4.0f); A(1, 1) = __float2half(5.0f); A(2, 1) = __float2half(6.0f); // Second column
      
      // Copy to device
      A.to_device();
      
      // Create vectors
      Vector<__half> x(n, Location::kHOST);   // Vector with n elements
      Vector<__half> y(m, Location::kHOST);   // Vector with m elements
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = __float2half(static_cast<float>(i + 1));  // [1, 2]
      }
      
      for (int i = 0; i < m; i++) {
         y[i] = __float2half(0.0f);  // Initialize y to zeros
      }
      
      // Copy vectors to device
      x.to_device();
      y.to_device();
      
      // Compute y = alpha * A * x + beta * y with alpha=1, beta=0
      float alpha = 1.0f;
      float beta = 0.0f;
      MSVDStatus status = gemv<__half, __half, float>(false, alpha, A, x, beta, y);
      
      // Check status
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Copy result back to host
      y.to_host();
      
      // Expected result: y = A * x = [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2] = [9, 12, 15]
      EXPECT_NEAR(__half2float(y[0]), 9.0f, 0.01);
      EXPECT_NEAR(__half2float(y[1]), 12.0f, 0.01);
      EXPECT_NEAR(__half2float(y[2]), 15.0f, 0.01);
   }
   
   // Test half precision with half compute on GPU
   {
      // Create a 3x2 matrix on host
      Matrix<__half> A(m, n, Location::kHOST);
      
      // Initialize matrix in column-major format
      A(0, 0) = __float2half(1.0f); A(1, 0) = __float2half(2.0f); A(2, 0) = __float2half(3.0f); // First column
      A(0, 1) = __float2half(4.0f); A(1, 1) = __float2half(5.0f); A(2, 1) = __float2half(6.0f); // Second column
      
      // Copy to device
      A.to_device();
      
      // Create vectors
      Vector<__half> x(n, Location::kHOST);   // Vector with n elements
      Vector<__half> y(m, Location::kHOST);   // Vector with m elements
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = __float2half(static_cast<float>(i + 1));  // [1, 2]
      }
      
      for (int i = 0; i < m; i++) {
         y[i] = __float2half(0.0f);  // Initialize y to zeros
      }
      
      // Copy vectors to device
      x.to_device();
      y.to_device();
      
      // Compute y = alpha * A * x + beta * y with alpha=1, beta=0
      __half alpha = __float2half(1.0f);
      __half beta = __float2half(0.0f);
      MSVDStatus status = gemv<__half, __half, __half>(false, alpha, A, x, beta, y);
      
      // Check status
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Copy result back to host
      y.to_host();
      
      // Expected result: y = A * x = [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2] = [9, 12, 15]
      EXPECT_NEAR(__half2float(y[0]), 9.0f, 0.01);
      EXPECT_NEAR(__half2float(y[1]), 12.0f, 0.01);
      EXPECT_NEAR(__half2float(y[2]), 15.0f, 0.01);
   }
   
   // Test double precision on CPU with transpose
   {
      // Create a 3x2 matrix on host
      Matrix<double> A(m, n, Location::kHOST);
      
      // Initialize matrix in column-major format
      A(0, 0) = 1.0; A(1, 0) = 2.0; A(2, 0) = 3.0; // First column
      A(0, 1) = 4.0; A(1, 1) = 5.0; A(2, 1) = 6.0; // Second column
      
      // Create vectors
      Vector<double> x(m, Location::kHOST);   // Vector with m elements (for A^T)
      Vector<double> y(n, Location::kHOST);   // Vector with n elements (for A^T)
      
      // Initialize vectors
      for (int i = 0; i < m; i++) {
         x[i] = static_cast<double>(i + 1);  // [1, 2, 3]
      }
      
      for (int i = 0; i < n; i++) {
         y[i] = 0.0;  // Initialize y to zeros
      }
      
      // Compute y = alpha * A^T * x + beta * y with alpha=1, beta=0
      double alpha = 1.0;
      double beta = 0.0;
      MSVDStatus status = gemv<double, double, double>(true, alpha, A, x, beta, y);
      
      // Check status
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Expected result for transpose: y = A^T * x = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
      EXPECT_DOUBLE_EQ(y[0], 14.0);
      EXPECT_DOUBLE_EQ(y[1], 32.0);
   }
   
   // Test mixed device/host (float version)
   {
      // Create a 3x2 matrix on host
      Matrix<float> A(m, n, Location::kHOST);
      
      // Create vectors - input on device, output on host (should fail)
      Vector<float> x(n, Location::kDEVICE);   // Input vector on device
      Vector<float> y(m, Location::kHOST);     // Output vector on host
      
      // Compute y = alpha * A * x + beta * y with alpha=1, beta=0
      // This should fail because vectors are on different devices
      float alpha = 1.0f;
      float beta = 0.0f;
      try {
         gemv<float, float, float>(false, alpha, A, x, beta, y);
      } catch (std::runtime_error const & err) {
         EXPECT_EQ(err.what(), std::string("Vectors and matrix must all be in the same location (CPU or GPU)"));
      } catch (...) {
         FAIL() << "Expected runtime_error";
      }
   }
}

// Test the gemm function
TEST(MvopsTest, Gemm) {
   // Test double precision on CPU
   {
      // Create matrices on host
      Matrix<double> A(3, 2, Location::kHOST);
      Matrix<double> B(2, 4, Location::kHOST);
      Matrix<double> C(3, 4, Location::kHOST);
      
      // Initialize matrices
      // Matrix A (3x2):
      // [1.0, 3.0]
      // [2.0, 4.0]
      // [5.0, 6.0]
      A(0, 0) = 1.0; A(0, 1) = 3.0;
      A(1, 0) = 2.0; A(1, 1) = 4.0;
      A(2, 0) = 5.0; A(2, 1) = 6.0;
      
      // Matrix B (2x4):
      // [1.0, 2.0, 3.0, 4.0]
      // [5.0, 6.0, 7.0, 8.0]
      B(0, 0) = 1.0; B(0, 1) = 2.0; B(0, 2) = 3.0; B(0, 3) = 4.0;
      B(1, 0) = 5.0; B(1, 1) = 6.0; B(1, 2) = 7.0; B(1, 3) = 8.0;
      
      // Fill C with zeros
      C.fill(0.0);
      
      // Compute C = A * B
      double alpha = 1.0;
      double beta = 0.0;
      MSVDStatus status = gemm<double, double, double>(false, false, alpha, A, B, beta, C);
      
      // Expected result:
      // [16.0, 20.0, 24.0, 28.0]
      // [22.0, 28.0, 34.0, 40.0]
      // [35.0, 46.0, 57.0, 68.0]
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(C(0, 0), 16.0);
      EXPECT_DOUBLE_EQ(C(0, 1), 20.0);
      EXPECT_DOUBLE_EQ(C(0, 2), 24.0);
      EXPECT_DOUBLE_EQ(C(0, 3), 28.0);
      EXPECT_DOUBLE_EQ(C(1, 0), 22.0);
      EXPECT_DOUBLE_EQ(C(1, 1), 28.0);
      EXPECT_DOUBLE_EQ(C(1, 2), 34.0);
      EXPECT_DOUBLE_EQ(C(1, 3), 40.0);
      EXPECT_DOUBLE_EQ(C(2, 0), 35.0);
      EXPECT_DOUBLE_EQ(C(2, 1), 46.0);
      EXPECT_DOUBLE_EQ(C(2, 2), 57.0);
      EXPECT_DOUBLE_EQ(C(2, 3), 68.0);
   }
   
   // Test single precision on GPU
   {
      // Create matrices
      Matrix<float> A(3, 2, Location::kHOST);
      Matrix<float> B(2, 4, Location::kHOST);
      Matrix<float> C(3, 4, Location::kDEVICE);
      
      // Initialize matrices
      // Matrix A (3x2):
      // [1.0, 3.0]
      // [2.0, 4.0]
      // [5.0, 6.0]
      A(0, 0) = 1.0f; A(0, 1) = 3.0f;
      A(1, 0) = 2.0f; A(1, 1) = 4.0f;
      A(2, 0) = 5.0f; A(2, 1) = 6.0f;
      
      // Matrix B (2x4):
      // [1.0, 2.0, 3.0, 4.0]
      // [5.0, 6.0, 7.0, 8.0]
      B(0, 0) = 1.0f; B(0, 1) = 2.0f; B(0, 2) = 3.0f; B(0, 3) = 4.0f;
      B(1, 0) = 5.0f; B(1, 1) = 6.0f; B(1, 2) = 7.0f; B(1, 3) = 8.0f;
      
      // Copy to device
      A.to_device();
      B.to_device();
      
      // Fill C with zeros on device
      C.fill(0.0f);
      
      // Compute C = A * B
      float alpha = 1.0f;
      float beta = 0.0f;
      MSVDStatus status = gemm<float, float, float>(false, false, alpha, A, B, beta, C);
      
      // Copy result back to host
      C.to_host();
      
      // Expected result:
      // [16.0, 20.0, 24.0, 28.0]
      // [22.0, 28.0, 34.0, 40.0]
      // [35.0, 46.0, 57.0, 68.0]
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(C(0, 0), 16.0f);
      EXPECT_FLOAT_EQ(C(0, 1), 20.0f);
      EXPECT_FLOAT_EQ(C(0, 2), 24.0f);
      EXPECT_FLOAT_EQ(C(0, 3), 28.0f);
      EXPECT_FLOAT_EQ(C(1, 0), 22.0f);
      EXPECT_FLOAT_EQ(C(1, 1), 28.0f);
      EXPECT_FLOAT_EQ(C(1, 2), 34.0f);
      EXPECT_FLOAT_EQ(C(1, 3), 40.0f);
      EXPECT_FLOAT_EQ(C(2, 0), 35.0f);
      EXPECT_FLOAT_EQ(C(2, 1), 46.0f);
      EXPECT_FLOAT_EQ(C(2, 2), 57.0f);
      EXPECT_FLOAT_EQ(C(2, 3), 68.0f);
   }
   
   // Test half precision on GPU with float compute
   {
      // Create matrices
      Matrix<__half> A(3, 2, Location::kHOST);
      Matrix<__half> B(2, 4, Location::kHOST);
      Matrix<__half> C(3, 4, Location::kDEVICE);
      
      // Matrix A (3x2):
      // [1.0, 3.0]
      // [2.0, 4.0]
      // [5.0, 6.0]
      A(0, 0) = __float2half(1.0f); A(0, 1) = __float2half(3.0f);
      A(1, 0) = __float2half(2.0f); A(1, 1) = __float2half(4.0f);
      A(2, 0) = __float2half(5.0f); A(2, 1) = __float2half(6.0f);
      
      // Matrix B (2x4):
      // [1.0, 2.0, 3.0, 4.0]
      // [5.0, 6.0, 7.0, 8.0]
      B(0, 0) = __float2half(1.0f); B(0, 1) = __float2half(2.0f); B(0, 2) = __float2half(3.0f); B(0, 3) = __float2half(4.0f);
      B(1, 0) = __float2half(5.0f); B(1, 1) = __float2half(6.0f); B(1, 2) = __float2half(7.0f); B(1, 3) = __float2half(8.0f);
      
      // Copy to device
      A.to_device();
      B.to_device();
      
      // Fill C with zeros on device
      C.fill(__float2half(0.0f));
      
      // Compute C = A * B with FP32 compute
      float alpha = 1.0f;
      float beta = 0.0f;
      MSVDStatus status = gemm<__half, __half, float>(false, false, alpha, A, B, beta, C);
      
      // Copy result back to host
      C.to_host();
      
      // Expected result (with some tolerance due to half precision):
      // [16.0, 20.0, 24.0, 28.0]
      // [22.0, 28.0, 34.0, 40.0]
      // [35.0, 46.0, 57.0, 68.0]
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_NEAR(__half2float(C(0, 0)), 16.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(0, 1)), 20.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(0, 2)), 24.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(0, 3)), 28.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(1, 0)), 22.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(1, 1)), 28.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(1, 2)), 34.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(1, 3)), 40.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(2, 0)), 35.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(2, 1)), 46.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(2, 2)), 57.0f, 0.1f);
      EXPECT_NEAR(__half2float(C(2, 3)), 68.0f, 0.1f);
   }
   
   // Test half precision on GPU with half compute
   {
      // Create matrices
      Matrix<__half> A(3, 2, Location::kHOST);
      Matrix<__half> B(2, 4, Location::kHOST);
      Matrix<__half> C(3, 4, Location::kDEVICE);
      
      // Initialize matrices
      // Matrix A (3x2):
      // [1.0, 3.0]
      // [2.0, 4.0]
      // [5.0, 6.0]
      A(0, 0) = __float2half(1.0f); A(0, 1) = __float2half(3.0f);
      A(1, 0) = __float2half(2.0f); A(1, 1) = __float2half(4.0f);
      A(2, 0) = __float2half(5.0f); A(2, 1) = __float2half(6.0f);
      
      // Matrix B (2x4):
      // [1.0, 2.0, 3.0, 4.0]
      // [5.0, 6.0, 7.0, 8.0]
      B(0, 0) = __float2half(1.0f); B(0, 1) = __float2half(2.0f); B(0, 2) = __float2half(3.0f); B(0, 3) = __float2half(4.0f);
      B(1, 0) = __float2half(5.0f); B(1, 1) = __float2half(6.0f); B(1, 2) = __float2half(7.0f); B(1, 3) = __float2half(8.0f);
      
      // Copy to device
      A.to_device();
      B.to_device();
      
      // Fill C with zeros on device
      C.fill(__float2half(0.0f));
      
      // Compute C = A * B with FP16 compute
      __half alpha = __float2half(1.0f);
      __half beta = __float2half(0.0f);
      MSVDStatus status = gemm<__half, __half, __half>(false, false, alpha, A, B, beta, C);
      
      // Copy result back to host
      C.to_host();
      
      // Expected result (with larger tolerance due to half precision computation):
      // [16.0, 20.0, 24.0, 28.0]
      // [22.0, 28.0, 34.0, 40.0]
      // [35.0, 46.0, 57.0, 68.0]
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_NEAR(__half2float(C(0, 0)), 16.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(0, 1)), 20.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(0, 2)), 24.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(0, 3)), 28.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(1, 0)), 22.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(1, 1)), 28.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(1, 2)), 34.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(1, 3)), 40.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(2, 0)), 35.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(2, 1)), 46.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(2, 2)), 57.0f, 0.2f);
      EXPECT_NEAR(__half2float(C(2, 3)), 68.0f, 0.2f);
   }
   
   // Test transposed matrices
   {
      // Create matrices on host for transposed test
      Matrix<double> A(2, 3, Location::kHOST);  // Will be transposed to 3x2
      Matrix<double> B(4, 2, Location::kHOST);  // Will be transposed to 2x4
      Matrix<double> C(3, 4, Location::kHOST);
      
      // Initialize matrices for transposed operation
      // Matrix A (2x3) to be transposed to (3x2):
      // [1.0, 2.0, 5.0]
      // [3.0, 4.0, 6.0]
      A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 5.0;
      A(1, 0) = 3.0; A(1, 1) = 4.0; A(1, 2) = 6.0;
      
      // Matrix B (4x2) to be transposed to (2x4):
      // [1.0, 5.0]
      // [2.0, 6.0]
      // [3.0, 7.0]
      // [4.0, 8.0]
      B(0, 0) = 1.0; B(0, 1) = 5.0;
      B(1, 0) = 2.0; B(1, 1) = 6.0;
      B(2, 0) = 3.0; B(2, 1) = 7.0;
      B(3, 0) = 4.0; B(3, 1) = 8.0;
      
      // Fill C with zeros
      C.fill(0.0);
      
      // Compute C = A^T * B^T
      double alpha = 1.0;
      double beta = 0.0;
      MSVDStatus status = gemm<double, double, double>(true, true, alpha, A, B, beta, C);
      
      // Expected result should be the same as regular test:
      // [16.0, 20.0, 24.0, 28.0]
      // [22.0, 28.0, 34.0, 40.0]
      // [35.0, 46.0, 57.0, 68.0]
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(C(0, 0), 16.0);
      EXPECT_DOUBLE_EQ(C(0, 1), 20.0);
      EXPECT_DOUBLE_EQ(C(0, 2), 24.0);
      EXPECT_DOUBLE_EQ(C(0, 3), 28.0);
      EXPECT_DOUBLE_EQ(C(1, 0), 22.0);
      EXPECT_DOUBLE_EQ(C(1, 1), 28.0);
      EXPECT_DOUBLE_EQ(C(1, 2), 34.0);
      EXPECT_DOUBLE_EQ(C(1, 3), 40.0);
      EXPECT_DOUBLE_EQ(C(2, 0), 35.0);
      EXPECT_DOUBLE_EQ(C(2, 1), 46.0);
      EXPECT_DOUBLE_EQ(C(2, 2), 57.0);
      EXPECT_DOUBLE_EQ(C(2, 3), 68.0);
   }
   
   // Test error cases
   {
      // Test matrices in different locations
      Matrix<float> A(3, 2, Location::kHOST);
      Matrix<float> B(2, 4, Location::kDEVICE);
      Matrix<float> C(3, 4, Location::kHOST);
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      // Using try/catch because we expect an exception
      try {
         gemm<float, float, float>(false, false, alpha, A, B, beta, C);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "All matrices must be in the same location (CPU or GPU)");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
      
      // Test dimension mismatch
      Matrix<float> A2(3, 3, Location::kHOST);  // Wrong dimension for multiplication
      Matrix<float> B2(2, 4, Location::kHOST);
      Matrix<float> C2(3, 4, Location::kHOST);
      
      try {
         gemm<float, float, float>(false, false, alpha, A2, B2, beta, C2);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Matrix dimensions do not match for multiplication");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
      
      // Test output matrix wrong dimensions
      Matrix<float> A3(3, 2, Location::kHOST);
      Matrix<float> B3(2, 4, Location::kHOST);
      Matrix<float> C3(3, 3, Location::kHOST);  // Wrong output dimension
      
      try {
         gemm<float, float, float>(false, false, alpha, A3, B3, beta, C3);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Output matrix C has wrong dimensions");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }
}

// Test the iamax function
TEST(MvopsTest, Iamax) {
   // Test double precision on CPU
   {
      const int n = 131072;
      Vector<double> x(n, Location::kHOST);
      
      // Initialize vector with a known maximum absolute value
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i - 65536) / 131072.0;  // [-0.5, ..., 0.5]
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 72345;
      x[expected_idx] = -8.0;  // This should be the largest absolute value
      
      // Find the index of maximum absolute value
      int result;
      double result_val;
      MSVDStatus status = iamax<double>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_DOUBLE_EQ(result_val, -8.0);
   }
   
   // Test single precision on CPU
   {
      const int n = 131072;
      Vector<float> x(n, Location::kHOST);
      
      // Initialize vector with a known maximum absolute value
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i - 65536) / 131072.0f;  // [-0.5, ..., 0.5]
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 77654;
      x[expected_idx] = 10.0f;  // This should be the largest absolute value
      
      // Find the index of maximum absolute value
      int result;
      float result_val;
      MSVDStatus status = iamax<float>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_FLOAT_EQ(result_val, 10.0f);
   }
   
   // Test double precision on GPU
   {
      const int n = 131072;
      Vector<double> x(n, Location::kHOST);
      
      // Initialize vector with a known maximum absolute value
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i - 65536) / 131072.0;  // [-0.5, ..., 0.5]
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 76666;
      x[expected_idx] = -9.0;  // This should be the largest absolute value
      
      // Copy data to device
      x.to_device();
      
      // Find the index of maximum absolute value
      int result;
      double result_val;
      MSVDStatus status = iamax<double>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_DOUBLE_EQ(result_val, 0.0); // GPU won't return value to host
   }
   
   // Test single precision on GPU
   {
      const int n = 131072;
      Vector<float> x(n, Location::kHOST);
         
      // Initialize vector with a known maximum absolute value
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i - 65536) / 131072.0f;  // [-0.5, ..., 0.5]
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 88888;
      x[expected_idx] = 12.0f;  // This should be the largest absolute value
      
      // Copy data to device
      x.to_device();
      
      // Find the index of maximum absolute value
      int result;
      float result_val;
      MSVDStatus status = iamax<float>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_FLOAT_EQ(result_val, 0.0f); // GPU won't return value to host
   }
   
   // Test half precision on GPU
   {
      const int n = 131072;
      Vector<__half> x(n, Location::kHOST);
      
      // Initialize vector with a known maximum absolute value
      for (int i = 0; i < n; i++) {
         x[i] = __float2half(static_cast<float>(i - 65536) / 131072.0f);  // [-0.5, ..., 0.5]
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 73333;
      x[expected_idx] = __float2half(-21.0f);  // This should be the largest absolute value
      
      // Copy data to device
      x.to_device();
      
      // Find the index of maximum absolute value
      int result;
      __half result_val;
      MSVDStatus status = iamax<__half>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_FLOAT_EQ(__half2float(result_val), __half2float(-21.0f));
   }

   // Test half precision on GPU
   {
      const int n = 131072;
      Vector<__half> x(n, Location::kHOST);
      
      // Initialize vector with a known maximum absolute value
      for (int i = 0; i < n; i++) {
         x[i] = __float2half(static_cast<float>(i - 65536) / 131072.0f);  // [-0.5, ..., 0.5]
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 73333;
      x[expected_idx] = __float2half(33.0f);  // This should be the largest absolute value
      
      // Copy data to device
      x.to_device();
      
      // Find the index of maximum absolute value
      int result;
      float result_val;
      MSVDStatus status = iamax<__half, float>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_FLOAT_EQ(result_val, 33.0f); // GPU won't return value to host
   }

   // Test mixed sign values on GPU
   {
      const int n = 131072;
      Vector<float> x(n, Location::kHOST);
      
      // Initialize vector with alternating signs
      for (int i = 0; i < n; i++) {
         x[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
      }
      
      // Set a specific value to be the maximum absolute value
      const int expected_idx = n - 1;  // Last element should have largest absolute value
      
      // Copy data to device
      x.to_device();
      
      // Find the index of maximum absolute value
      int result;
      float result_val;
      MSVDStatus status = iamax<float>(x, result, result_val);
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(result, expected_idx + 1);  // Added +1 for 1-based indexing
      EXPECT_FLOAT_EQ(result_val, 0.0f); // GPU won't return value to host
   }
   
   // Test error cases
   {
      // Test with empty vector
      Vector<float> empty_vector(0, Location::kHOST);
      int result;
      float result_val;
      try {
         iamax<float>(empty_vector, result, result_val);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Empty vector not allowed for iamax operation");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }

   // Test device output with double precision
   {
      const int n = 131072;
      
      // Initialize vector with random values on host first
      Vector<double> h_x(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<double>(i - 65536) / 131072.0;
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 72345;
      h_x[expected_idx] = -8.0;  // This should be the largest absolute value
      
      // Copy to device
      Vector<double> x(h_x);
      x.to_device();
      
      // Allocate device memory for result
      int* d_result;
      CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
      double* d_result_val;
      CUDA_CHECK(cudaMalloc(&d_result_val, sizeof(double)));
      
      // Find the index of maximum absolute value with result on device
      MSVDStatus status = iamax<double>(x, *d_result, *d_result_val, Location::kDEVICE);
      
      // Copy result back to host for verification
      int h_result;
      CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result));

      double h_result_val;
      CUDA_CHECK(cudaMemcpy(&h_result_val, d_result_val, sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result_val));
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(h_result, expected_idx + 1);
      EXPECT_DOUBLE_EQ(h_result_val, -8.0);
   }
   
   // Test device output with single precision
   {
      const int n = 131072;
      
      // Initialize vector with random values on host first
      Vector<float> h_x(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<float>(i - 65536) / 131072.0f;
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 77654;
      h_x[expected_idx] = 10.0f;  // This should be the largest absolute value
      
      // Copy to device
      Vector<float> x(h_x);
      x.to_device();
      
      // Allocate device memory for result
      int* d_result;
      CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
      float* d_result_val;
      CUDA_CHECK(cudaMalloc(&d_result_val, sizeof(float)));
      
      // Find the index of maximum absolute value with result on device
      MSVDStatus status = iamax<float>(x, *d_result, *d_result_val, Location::kDEVICE);
      
      // Copy result back to host for verification
      int h_result;
      CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result));

      float h_result_val;
      CUDA_CHECK(cudaMemcpy(&h_result_val, d_result_val, sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result_val));
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(h_result, expected_idx + 1);
      EXPECT_FLOAT_EQ(h_result_val, 10.0f);
   }
   
   // Test device output with half precision and single precision result return
   {
      const int n = 131072;
      
      // Initialize vector with random values on host first
      Vector<__half> h_x(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         h_x[i] = __float2half(static_cast<float>(i - 65536) / 131072.0f);
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 75678;
      h_x[expected_idx] = __float2half(15.0f);  // This should be the largest absolute value
      
      // Copy to device
      Vector<__half> x(h_x);
      x.to_device();
      
      // Allocate device memory for result
      int* d_result;
      CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
      float* d_result_val;
      CUDA_CHECK(cudaMalloc(&d_result_val, sizeof(float)));
      
      // Find the index of maximum absolute value with result on device
      MSVDStatus status = iamax<__half,  float>(x, *d_result, *d_result_val, Location::kDEVICE);
      
      // Copy result back to host for verification
      int h_result;
      CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result));

      float h_result_val;
      CUDA_CHECK(cudaMemcpy(&h_result_val, d_result_val, sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result_val));
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(h_result, expected_idx + 1);
      EXPECT_FLOAT_EQ(h_result_val, 15.0f);
   }

   
   // Test device output with half precision and half precision result return
   {
      const int n = 131072;
      
      // Initialize vector with random values on host first
      Vector<__half> h_x(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         h_x[i] = __float2half(static_cast<float>(i - 65536) / 131072.0f);
      }
      
      // Set a specific value to be the maximum
      const int expected_idx = 75678;
      h_x[expected_idx] = __float2half(15.0f);  // This should be the largest absolute value
      
      // Copy to device
      Vector<__half> x(h_x);
      x.to_device();
      
      // Allocate device memory for result
      int* d_result;
      CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
      __half* d_result_val;
      CUDA_CHECK(cudaMalloc(&d_result_val, sizeof(__half)));
      
      // Find the index of maximum absolute value with result on device
      MSVDStatus status = iamax<__half>(x, *d_result, *d_result_val, Location::kDEVICE);
      
      // Copy result back to host for verification
      int h_result;
      CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result));

      __half h_result_val;
      CUDA_CHECK(cudaMemcpy(&h_result_val, d_result_val, sizeof(__half), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_result_val));
      
      // Check results (note: using 1-based indexing now)
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_EQ(h_result, expected_idx + 1);
      EXPECT_FLOAT_EQ(__half2float(h_result_val), __half2float(15.0f));
   }
}

// Test the matrix_iamax function
TEST(MvopsTest, MatrixIamax) {
   // Test double precision on CPU
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<double> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = static_cast<double>(row - 5000) / 10000.0;
         }
      }

      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Allocate memory for results
      int results[cols] = {0};
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<double>(A, results);
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col]) 
            << "Mismatch at column " << col << ": expected " 
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test single precision on CPU
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<float> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = static_cast<float>(row - 5000) / 10000.0f;
         }
      }
      
      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Allocate memory for results
      int results[cols] = {0};
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<float>(A, results);
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col]) 
            << "Mismatch at column " << col << ": expected " 
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test double precision on GPU
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<double> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = static_cast<double>(row - 5000) / 10000.0;
         }
      }
      
      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Copy data to device
      A.to_device();
      
      // Allocate memory for results
      int results[cols] = {0};
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<double>(A, results, Location::kHOST);
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col])
            << "Mismatch at column " << col << ": expected "
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test double precision on GPU with results on device
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<double> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = static_cast<double>(row - 5000) / 10000.0;
         }
      }
      
      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Copy data to device
      A.to_device();
      
      // Allocate memory for results
      int results[cols] = {0};
      int* d_results;
      CUDA_CHECK(cudaMalloc(&d_results, cols * sizeof(int)));
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<double>(A, d_results, Location::kDEVICE);
      
      // Copy results back to host
      CUDA_CHECK(cudaMemcpy(results, d_results, cols * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_results));
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col])
            << "Mismatch at column " << col << ": expected "
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test single precision on GPU
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<float> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = static_cast<float>(row - 5000) / 10000.0f;
         }
      }
      
      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Copy data to device
      A.to_device();
      
      // Allocate memory for results
      int results[cols] = {0};
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<float>(A, results, Location::kHOST);
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col])
            << "Mismatch at column " << col << ": expected "
            << expected_results[col] << ", got " << results[col];
      }
   }

   // Test single precision on GPU with results on device
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<float> A(rows, cols, Location::kHOST);

      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = static_cast<float>(row - 5000) / 10000.0f;
         }
      }

      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }

      // Copy data to device
      A.to_device();

      // Allocate memory for results
      int results[cols] = {0};
      int* d_results;
      CUDA_CHECK(cudaMalloc(&d_results, cols * sizeof(int)));
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<float>(A, d_results, Location::kDEVICE);

      // Copy results back to host
      CUDA_CHECK(cudaMemcpy(results, d_results, cols * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_results));
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col])
            << "Mismatch at column " << col << ": expected "
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test half precision on GPU
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<__half> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = __float2half(static_cast<float>(row - 5000) / 10000.0f);
         }
      }

      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Copy to device
      A.to_device();
      
      // Allocate memory for results
      int results[cols] = {0};
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<__half>(A, results, Location::kHOST);
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col])
            << "Mismatch at column " << col << ": expected "
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test half precision on GPU with results on device
   {
      const int rows = 10000;
      const int cols = 1000;
      srand(42);
      Matrix<__half> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with known maximum absolute values in each column
      for (int col = 0; col < cols; ++col) {
         for (int row = 0; row < rows; ++row) {
            A(row, col) = __float2half(static_cast<float>(row - 5000) / 10000.0f);
         }
      }

      int expected_results[cols];
      for (int col = 0; col < cols; ++col) {
         expected_results[col] = rand() % rows;
         A(expected_results[col], col) = col % 2 == 0 ? - col - 1 : col + 1;
      }
      
      // Copy to device
      A.to_device();
      
      // Allocate memory for results
      int results[cols] = {0};
      int* d_results;
      CUDA_CHECK(cudaMalloc(&d_results, cols * sizeof(int)));
      
      // Call matrix_iamax
      MSVDStatus status = matrix_iamax<__half>(A, d_results, Location::kDEVICE);
      
      // Copy results back to host
      CUDA_CHECK(cudaMemcpy(results, d_results, cols * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_results));
      
      // Check status and results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int col = 0; col < cols; ++col) {
         EXPECT_EQ(results[col], expected_results[col])
            << "Mismatch at column " << col << ": expected "
            << expected_results[col] << ", got " << results[col];
      }
   }
   
   // Test error cases
   {
      // Test with empty matrix
      Matrix<float> empty_matrix(0, 3, Location::kHOST);
      int results[3] = {0};
      
      try {
         matrix_iamax<float>(empty_matrix, results);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Empty matrix not allowed for matrix_iamax operation");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
      
      // Test with null results
      Matrix<float> A(3, 2, Location::kHOST);
      
      try {
         matrix_iamax<float>(A, nullptr);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Results array cannot be null");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }
}

// Test the axpy function
TEST(MvopsTest, Axpy) {
   // Test double precision on CPU
   {
      const int n = 1234;
      Vector<double> x(n, Location::kHOST);
      Vector<double> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1); 
         y[i] = static_cast<double>(i * 2); 
      }
      
      // Scalar alpha
      double alpha = 0.33;
      
      // Check results
      Vector<double> expected(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected[i] = alpha * x[i] + y[i];
      }
      
      // Call axpy
      MSVDStatus status = axpy<double, double, double>(alpha, x, y);
      
      // Check results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_DOUBLE_EQ(y[i], expected[i]);
      }
   }
   
   // Test single precision on CPU
   {
      const int n = 1234;
      Vector<float> x(n, Location::kHOST);
      Vector<float> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i + 1); 
         y[i] = static_cast<float>(i * 2); 
      }
      
      // Scalar alpha
      float alpha = 0.33f;
      
      // Check results
      Vector<float> expected(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected[i] = alpha * x[i] + y[i];
      }
      
      // Call axpy
      MSVDStatus status = axpy<float, float, float>(alpha, x, y);
      
      // Check results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_FLOAT_EQ(y[i], expected[i]);
      }
   }
   
   // Test double precision on GPU
   {
      const int n = 1234;
      
      // Temporary host vectors for initialization
      Vector<double> h_x(n, Location::kHOST);
      Vector<double> h_y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<double>(i + 1); 
         h_y[i] = static_cast<double>(i * 2); 
      }
      
      // Copy data to device vectors
      Vector<double> x = h_x;
      Vector<double> y = h_y;
      x.to_device();
      y.to_device();
      
      // Scalar alpha
      double alpha = 0.33;
      
      // Check results
      Vector<double> expected_double(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected_double[i] = alpha * h_x[i] + h_y[i];
      }
      
      // Call axpy
      MSVDStatus status_double = axpy<double, double, double>(alpha, x, y);
      
      // Copy result back to host
      Vector<double> result_double(n, Location::kHOST);
      CUDA_CHECK(cudaMemcpy(result_double.data(), y.data(), n * sizeof(double), cudaMemcpyDeviceToHost));
      
      // Check results
      EXPECT_EQ(status_double, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_DOUBLE_EQ(result_double[i], expected_double[i]);
      }
   }
   
   // Test single precision on GPU
   {
      const int n = 1234;
      // Temporary host vectors for initialization
      Vector<float> h_x(n, Location::kHOST);
      Vector<float> h_y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<float>(i + 1); 
         h_y[i] = static_cast<float>(i * 2); 
      }
      
      // Copy data to device vectors
      Vector<float> x = h_x;
      Vector<float> y = h_y;
      x.to_device();
      y.to_device();
      
      // Scalar alpha
      float alpha = 0.33f;
      
      // Check results
      Vector<float> expected_float(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected_float[i] = alpha * h_x[i] + h_y[i];
      }
      
      // Call axpy
      MSVDStatus status_float = axpy<float, float, float>(alpha, x, y);
      
      // Copy result back to host
      Vector<float> result_float(n, Location::kHOST);
      CUDA_CHECK(cudaMemcpy(result_float.data(), y.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
      
      // Check results
      EXPECT_EQ(status_float, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_FLOAT_EQ(result_float[i], expected_float[i]);
      }
   }
   
   // Test half precision on GPU
   {
      const int n = 1234;
      // Temporary host vectors for initialization
      Vector<__half> h_x(n, Location::kHOST);
      Vector<__half> h_y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         h_x[i] = __float2half(static_cast<float>((i + 1)/123.4f)); 
         h_y[i] = __float2half(static_cast<float>((i * 2)/123.4f)); 
      }
      
      // Copy data to device vectors
      Vector<__half> x = h_x;
      Vector<__half> y = h_y;
      x.to_device();
      y.to_device();
      
      // Scalar alpha
      float alpha = 0.33f; 
      
      // Check results
      Vector<float> expected_float(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected_float[i] = alpha * __half2float(h_x[i]) + __half2float(h_y[i]);
      }
      
      // Call axpy
      MSVDStatus status_half = axpy<__half, __half, float>(alpha, x, y);
      
      // Copy result back to host
      Vector<__half> result_half(n, Location::kHOST);
      CUDA_CHECK(cudaMemcpy(result_half.data(), y.data(), n * sizeof(__half), cudaMemcpyDeviceToHost));
      
      // Check results - allow small tolerance for half precision
      EXPECT_EQ(status_half, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_NEAR(__half2float(result_half[i]), expected_float[i], 0.01f);
      }
   }
   
   // Test error cases
   {
      // Test with vectors in different locations
      Vector<float> x(5, Location::kHOST);
      Vector<float> y(5, Location::kDEVICE);
      
      // Test axpy with vectors in different locations - should throw exception
      float alpha = 2.0f;
      
      try {
         axpy<float, float, float>(alpha, x, y);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Both vectors must be in the same location (CPU or GPU)");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }
}

// Test the scale function
TEST(MvopsTest, Scale) {
   // Test double precision on CPU
   {
      const int n = 1234;
      Vector<double> x(n, Location::kHOST);
      Vector<double> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1); 
      }
      
      // Scalar alpha
      double alpha = 0.33;
      
      // Check results
      Vector<double> expected(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected[i] = alpha * x[i];
      }
      
      // Call scale
      MSVDStatus status = scale<double, double, double>(alpha, x, y);
      
      // Check results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_DOUBLE_EQ(y[i], expected[i]);
      }
   }
   
   // Test single precision on CPU
   {
      const int n = 1234;
      Vector<float> x(n, Location::kHOST);
      Vector<float> y(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i + 1); 
      }
      
      // Scalar alpha
      float alpha = 0.33f;
      
      // Check results
      Vector<float> expected(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected[i] = alpha * x[i];
      }
      
      // Call scale
      MSVDStatus status = scale<float, float, float>(alpha, x, y);
      
      // Check results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_FLOAT_EQ(y[i], expected[i]);
      }
   }
   
   // Test double precision on GPU
   {
      const int n = 1234;
      Vector<double> y(n, Location::kDEVICE);
      
      // Temporary host vectors for initialization
      Vector<double> h_x(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<double>(i + 1); 
      }
      
      // Copy data to device vectors
      Vector<double> x = h_x;
      x.to_device();
      
      // Scalar alpha
      double alpha = 0.33;
      
      // Check results
      Vector<double> expected(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected[i] = alpha * h_x[i];
      }
      
      // Call scale
      MSVDStatus status = scale<double, double, double>(alpha, x, y);
      
      // Copy result back to host
      Vector<double> result(n, Location::kHOST);
      CUDA_CHECK(cudaMemcpy(result.data(), y.data(), n * sizeof(double), cudaMemcpyDeviceToHost));
      
      // Check results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_DOUBLE_EQ(result[i], expected[i]);
      }
   }
   
   // Test single precision on GPU
   {
      const int n = 1234;
      Vector<float> y(n, Location::kDEVICE);
      
      // Temporary host vectors for initialization
      Vector<float> h_x(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<float>(i + 1); 
      }
      
      // Copy data to device vectors
      Vector<float> x = h_x;
      x.to_device();
      
      // Scalar alpha
      float alpha = 3.0f;
      
      // Expected result: y = alpha * x = 3 * [1, 2, 3, 4, 5] = [3, 6, 9, 12, 15]
      Vector<float> expected(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected[i] = alpha * h_x[i];
      }
      
      // Call scale
      MSVDStatus status = scale<float, float, float>(alpha, x, y);
      
      // Copy result back to host
      Vector<float> result(n, Location::kHOST);
      CUDA_CHECK(cudaMemcpy(result.data(), y.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
      
      // Check results
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_FLOAT_EQ(result[i], expected[i]);
      }
   }
   
   // Test half precision on GPU
   {
      const int n = 1234;
      Vector<__half> y(n, Location::kDEVICE);
      
      // Temporary host vectors for initialization
      Vector<__half> h_x(n, Location::kHOST);
      
      // Initialize vectors
      for (int i = 0; i < n; i++) {
         h_x[i] = __float2half(static_cast<float>((i + 1)/123.4f)); 
      }
      
      // Copy data to device vectors
      Vector<__half> x = h_x;
      x.to_device();
      
      // Scalar alpha
      float alpha = 0.33f;  // Using float for compute with half precision
      
      // Check results
      Vector<float> expected_float(n, Location::kHOST);
      for (int i = 0; i < n; i++) {
         expected_float[i] = alpha * __half2float(h_x[i]);
      }
      
      // Call scale
      MSVDStatus status = scale<__half, __half, float>(alpha, x, y);
      
      // Copy result back to host
      Vector<__half> result(n, Location::kHOST);
      CUDA_CHECK(cudaMemcpy(result.data(), y.data(), n * sizeof(__half), cudaMemcpyDeviceToHost));
      
      // Check results - allow small tolerance for half precision
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      for (int i = 0; i < n; i++) {
         EXPECT_NEAR(__half2float(result[i]), expected_float[i], 0.01f);
      }
   }
   
   // Test error cases
   {
      // Test with vectors in different locations
      Vector<float> x(5, Location::kHOST);
      Vector<float> y(5, Location::kDEVICE);
      
      // Test scale with vectors in different locations - should throw exception
      float alpha = 2.0f;
      
      try {
         scale<float, float, float>(alpha, x, y);
         FAIL() << "Expected std::runtime_error";
      } catch(std::runtime_error const & err) {
         EXPECT_EQ(std::string(err.what()), "Both vectors must be in the same location (CPU or GPU)");
      } catch(...) {
         FAIL() << "Expected std::runtime_error";
      }
   }
}

// Test the check_special_values function for vectors
TEST(MvopsTest, CheckSpecialValuesVector) {
   // Test with double precision on CPU
   {
      const int n = 1234;
      Vector<double> x(n, Location::kHOST);
      
      // Initialize vector with normal values
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1);
      }
      
      // All values are normal
      MSVDStatus status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      x[678] = std::numeric_limits<double>::quiet_NaN();
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      x[678] = std::numeric_limits<double>::infinity();
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test with single precision on CPU
   {
      const int n = 1234;
      Vector<float> x(n, Location::kHOST);
      
      // Initialize vector with normal values
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i + 1);
      }
      
      // All values are normal
      MSVDStatus status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      x[876] = std::numeric_limits<float>::quiet_NaN();
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      x[876] = std::numeric_limits<float>::infinity();
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test with double precision on GPU
   {
      const int n = 1234;
      Vector<double> x(n, Location::kDEVICE);
      
      // Temporary host vector for initialization
      Vector<double> h_x(n, Location::kHOST);
      
      // Initialize vector with normal values
      for (int i = 0; i < n; i++) {
         h_x[i] = static_cast<double>(i + 1);
      }
      
      // Copy data to device vector
      CUDA_CHECK(cudaMemcpy(x.data(), h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
      
      // All values are normal
      MSVDStatus status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      h_x[777] = std::numeric_limits<double>::quiet_NaN();
      CUDA_CHECK(cudaMemcpy(x.data(), h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      h_x[777] = std::numeric_limits<double>::infinity();
      CUDA_CHECK(cudaMemcpy(x.data(), h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test with half precision on GPU
   {
      const int n = 1234;
      Vector<__half> x(n, Location::kDEVICE);
      
      // Temporary host vector for initialization
      Vector<__half> h_x(n, Location::kHOST);
      
      // Initialize vector with normal values
      for (int i = 0; i < n; i++) {
         h_x[i] = __float2half(static_cast<float>(i + 1));
      }
      
      // Copy data to device vector
      CUDA_CHECK(cudaMemcpy(x.data(), h_x.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
      
      // All values are normal
      MSVDStatus status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      h_x[777] = __float2half(NAN);
      CUDA_CHECK(cudaMemcpy(x.data(), h_x.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      h_x[777] = __float2half(INFINITY);
      CUDA_CHECK(cudaMemcpy(x.data(), h_x.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
      status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test empty vector
   {
      Vector<float> x(0, Location::kHOST);
      MSVDStatus status = check_special_values(x);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
   }
}

// Test the check_special_values function for matrices
TEST(MvopsTest, CheckSpecialValuesMatrix) {
   // Test with double precision on CPU
   {
      const int rows = 4321;
      const int cols = 1234;
      Matrix<double> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with normal values
      for (int j = 0; j < cols; j++) {
         for (int i = 0; i < rows; i++) {
            A(i, j) = static_cast<double>(i + j * rows + 1);
         }
      }
      
      // All values are normal
      MSVDStatus status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      A(678, 876) = std::numeric_limits<double>::quiet_NaN();
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      A(678, 876) = std::numeric_limits<double>::infinity();
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test with single precision on CPU
   {
      const int rows = 1234;
      const int cols = 4321;
      Matrix<float> A(rows, cols, Location::kHOST);
      
      // Initialize matrix with normal values
      for (int j = 0; j < cols; j++) {
         for (int i = 0; i < rows; i++) {
            A(i, j) = static_cast<float>(i + j * rows + 1);
         }
      }
      
      // All values are normal
      MSVDStatus status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      A(876, 678) = std::numeric_limits<float>::quiet_NaN();
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      A(876, 678) = std::numeric_limits<float>::infinity();
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test with double precision on GPU
   {
      const int rows = 1234;
      const int cols = 4321;
      
      // Temporary host matrix for initialization
      Matrix<double> h_A(rows, cols, Location::kHOST);
      
      // Initialize matrix with normal values
      for (int j = 0; j < cols; j++) {
         for (int i = 0; i < rows; i++) {
            h_A(i, j) = static_cast<double>(i + j * rows + 1);
         }
      }
      
      // Copy data to device matrix
      Matrix<double> A = h_A;
      A.to_device();
      
      // All values are normal
      MSVDStatus status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      h_A(678, 876) = std::numeric_limits<double>::quiet_NaN();
      CUDA_CHECK(cudaMemcpy(A.data(), h_A.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice));
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      h_A(678, 876) = std::numeric_limits<double>::infinity();
      CUDA_CHECK(cudaMemcpy(A.data(), h_A.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice));
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test with half precision on GPU
   {
      const int rows = 1234;
      const int cols = 4321;
      
      // Temporary host matrix for initialization
      Matrix<__half> h_A(rows, cols, Location::kHOST);
      
      // Initialize matrix with normal values
      for (int j = 0; j < cols; j++) {
         for (int i = 0; i < rows; i++) {
            h_A(i, j) = __float2half(static_cast<float>((i + j * rows + 1) / 54321.0f));
         }
      }
      
      // Copy data to device matrix
      Matrix<__half> A = h_A;
      A.to_device();
      
      // All values are normal
      MSVDStatus status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      
      // Add a NaN
      h_A(777, 777) = __float2half(NAN);
      CUDA_CHECK(cudaMemcpy(A.data(), h_A.data(), rows * cols * sizeof(__half), cudaMemcpyHostToDevice));
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorNaN);
      
      // Add an Inf
      h_A(777, 777) = __float2half(INFINITY);
      CUDA_CHECK(cudaMemcpy(A.data(), h_A.data(), rows * cols * sizeof(__half), cudaMemcpyHostToDevice));
      status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kErrorInf);
   }
   
   // Test empty matrix
   {
      Matrix<float> A(0, 0, Location::kHOST);
      MSVDStatus status = check_special_values(A);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
   }
}

// Test the unified norm2 function
TEST(MvopsTest, UnifiedNorm2) {
   const int n = 5;
   
   // Test double precision on GPU
   {
      Vector<double> x(n, Location::kHOST);
      
      // Initialize vector [1.0, 2.0, 3.0, 4.0, 5.0]
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1);
      }
      
      // Copy vector to device
      x.to_device();
      
      // Calculate 2-norm
      double result;
      MSVDStatus status = norm2<double, double, double>(x, result);
      
      // Expected result: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2) = sqrt(55) ≈ 7.416198
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(result, std::sqrt(55.0));
   }
   
   // Test double precision on CPU
   {
      // Create host vector
      Vector<double> x(n, Location::kHOST);
      
      // Initialize vector
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<double>(i + 1);
      }
      
      // Calculate 2-norm
      double result;
      MSVDStatus status = norm2<double, double, double>(x, result);
      
      // Expected result: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2) = sqrt(55) ≈ 7.416198
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_DOUBLE_EQ(result, std::sqrt(55.0));
   }
   
   // Test single precision on GPU
   {
      // Create device vector
      Vector<float> x(n, Location::kHOST);
      
      // Initialize vector
      for (int i = 0; i < n; i++) {
         x[i] = static_cast<float>(i + 1);
      }
      
      // Copy vector to device
      x.to_device();
      
      // Calculate 2-norm
      float result;
      MSVDStatus status = norm2<float, float, float>(x, result);
      
      // Expected result: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2) = sqrt(55) ≈ 7.416198
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(result, std::sqrt(55.0f));
   }
   
   // Test half precision on GPU
   {
      // Create device vector
      Vector<__half> x(n, Location::kHOST);
      
      // Initialize vector
      for (int i = 0; i < n; i++) {
         x[i] = __float2half(static_cast<float>(i + 1));
      }
      
      // Copy data to device vector
      x.to_device();
      
      // Calculate 2-norm with output in FP16
      __half result16;
      MSVDStatus status = norm2<__half, __half, float>(x, result16);
      
      // Expected result: sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2) = sqrt(55) ≈ 7.416198
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_NEAR(__half2float(result16), std::sqrt(55.0f), 0.1f);
   }
   
   // Test empty vector on CPU
   {
      // Create empty vector
      Vector<float> x(0, Location::kHOST);
      
      // Calculate 2-norm with empty vector - should return 0
      float result = 1.0f;  // Initialize with non-zero value
      MSVDStatus status = norm2<float, float, float>(x, result);
      
      // Should return success and set result to 0
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(result, 0.0f);
   }
   
   // Test half precision empty vector on GPU
   {
      // Create empty vector on GPU
      Vector<__half> x(0, Location::kDEVICE);
      
      // Calculate 2-norm with empty vector - should return 0
      __half result = __float2half(1.0f);  // Initialize with non-zero value
      MSVDStatus status = norm2<__half, __half, float>(x, result);
      
      // Should return success and set result to 0
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      EXPECT_FLOAT_EQ(__half2float(result), 0.0f);
   }
}

// Test of symmetric eigenvalue problem
TEST(MvopsTest, SYEV) {
   const int n = 10;
   
   // Test double precision on CPU
   {
      Matrix<double> A(n, n, Location::kHOST);
      A.fill(0.0);
      
      // Create a tridiagonal matrix with 2 on diagonal and -1 on super- and sub-diagonal
      for(int i = 0; i < n; i++) {
         A(i, i) = 2.0;
         if(i < n - 1) {
            A(i, i + 1) = -1.0;
            A(i + 1, i) = -1.0;
         }
      }
      Matrix<double> V(A);
      
      // The true eigenvalues are 
      // 0.0810140527710052
      // 0.317492934337638
      // 0.69027853210943
      // 1.16916997399623
      // 1.71537032345343
      // 2.28462967654657
      // 2.83083002600377
      // 3.30972146789057
      // 3.68250706566236
      // 3.91898594722899
      double true_eigenvalues[n] = {
         0.0810140527710052,
         0.317492934337638,
         0.69027853210943,
         1.16916997399623,
         1.71537032345343,
         2.28462967654657,
         2.83083002600377,
         3.30972146789057,
         3.68250706566236,
         3.91898594722899
      };

      Vector<double> W(n, Location::kHOST);
      
      double work_query;
      int lwork = -1;
      MSVDStatus status = syev(V, W, &work_query, lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      lwork = static_cast<int>(work_query);
      Vector<double> work(lwork, Location::kHOST);
      status = syev(V, W, work.data(), lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      double max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         max_diff = std::max(max_diff, std::abs(W[i] - true_eigenvalues[i]));
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-10);

      // Test the eigenvectors, note that we should have A*V close to V*D where D is the diagonal matrix of eigenvalues
      Matrix<double> AV(n, n, Location::kHOST);
      double alpha = 1.0;
      double beta = 0.0;
      gemm<double, double, double>(false, false, alpha, A, V, beta, AV);
      Matrix<double> D(n, n, Location::kHOST);
      D.fill(0.0);
      for(int i = 0; i < n; i++) {
         D(i, i) = true_eigenvalues[i];
      }
      Matrix<double> VD(n, n, Location::kHOST);
      gemm<double, double, double>(false, false, alpha, V, D, beta, VD);
      
      max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         for(int j = 0; j < n; j++) {
            max_diff = std::max(max_diff, std::abs(AV(i, j) - VD(i, j)));
         }
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-10);
      
   }

   // Test single precision on CPU
   {
      Matrix<float> A(n, n, Location::kHOST);
      A.fill(0.0);
      
      // Create a tridiagonal matrix with 2 on diagonal and -1 on super- and sub-diagonal
      for(int i = 0; i < n; i++) {
         A(i, i) = 2.0;
         if(i < n - 1) {
            A(i, i + 1) = -1.0;
            A(i + 1, i) = -1.0;
         }
      }
      Matrix<float> V(A);
      
      // The true eigenvalues are 
      // 0.0810140527710052
      // 0.317492934337638
      // 0.69027853210943
      // 1.16916997399623
      // 1.71537032345343
      // 2.28462967654657
      // 2.83083002600377
      // 3.30972146789057
      float true_eigenvalues[n] = {
         0.0810140527710052,
         0.317492934337638,
         0.69027853210943,
         1.16916997399623,
         1.71537032345343,
         2.28462967654657,
         2.83083002600377,
         3.30972146789057,
         3.68250706566236,
         3.91898594722899
      };

      Vector<float> W(n, Location::kHOST);
      
      float work_query;
      int lwork = -1;
      MSVDStatus status = syev(V, W, &work_query, lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      lwork = static_cast<int>(work_query);
      Vector<float> work(lwork, Location::kHOST);
      status = syev(V, W, work.data(), lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      float max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         max_diff = std::max(max_diff, std::abs(W[i] - true_eigenvalues[i]));
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-06);
      
      // Test the eigenvectors, note that we should have A*V close to V*D where D is the diagonal matrix of eigenvalues
      Matrix<float> AV(n, n, Location::kHOST);
      float alpha = 1.0;
      float beta = 0.0;
      gemm<float, float, float>(false, false, alpha, A, V, beta, AV);
      Matrix<float> D(n, n, Location::kHOST);
      D.fill(0.0);
      for(int i = 0; i < n; i++) {
         D(i, i) = true_eigenvalues[i];
      }
      Matrix<float> VD(n, n, Location::kHOST);
      gemm<float, float, float>(false, false, alpha, V, D, beta, VD);
      
      max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         for(int j = 0; j < n; j++) {
            max_diff = std::max(max_diff, std::abs(AV(i, j) - VD(i, j)));
         }
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-06);
   }
}

// Test of symmetric eigenvalue problem
TEST(MvopsTest, SYGV) {
   const int n = 10;
   
   // Test double precision on CPU
   {
      Matrix<double> A(n, n, Location::kHOST);
      Matrix<double> B(n, n, Location::kHOST);
      A.fill(0.0);
      B.fill(0.0);
      
      // Create a tridiagonal matrix with 2 on diagonal and -1 on super- and sub-diagonal
      for(int i = 0; i < n; i++) {
         A(i, i) = 2.0;
         if(i < n - 1) {
            A(i, i + 1) = -1.0;
            A(i + 1, i) = -1.0;
         }
      }
      Matrix<double> V(A);

      // Create a B matrix with diagonal 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
      for(int i = 0; i < n; i++) {
         B(i, i) = i + 1;
      }
      Matrix<double> B_chol(B);
      
      // The true eigenvalues are 
      // 0.0141309446511971
      // 0.0594567850513288
      // 0.131202907506177
      // 0.220848003169762
      // 0.316795717268112
      // 0.410267468002017
      // 0.52158439013008
      // 0.705660103820319
      // 1.08997474266915
      // 2.38801544566837
      double true_eigenvalues[n] = {
         0.0141309446511971,
         0.0594567850513288,
         0.131202907506177,
         0.220848003169762,
         0.316795717268112,
         0.410267468002017,
         0.52158439013008,
         0.705660103820319,
         1.08997474266915,
         2.38801544566837
      };

      Vector<double> W(n, Location::kHOST);
      
      double work_query;
      int lwork = -1;
      MSVDStatus status = sygv(V, B_chol, W, &work_query, lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      lwork = static_cast<int>(work_query);
      Vector<double> work(lwork, Location::kHOST);
      status = sygv(V, B_chol, W, work.data(), lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      double max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         max_diff = std::max(max_diff, std::abs(W[i] - true_eigenvalues[i]));
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-10);

      // Test the eigenvectors, note that we should have A*V close to V*D where D is the diagonal matrix of eigenvalues
      Matrix<double> AV(n, n, Location::kHOST);
      double alpha = 1.0;
      double beta = 0.0;
      gemm<double, double, double>(false, false, alpha, A, V, beta, AV);
      Matrix<double> D(n, n, Location::kHOST);
      D.fill(0.0);
      for(int i = 0; i < n; i++) {
         D(i, i) = true_eigenvalues[i];
      }
      Matrix<double> BV(n, n, Location::kHOST);
      gemm<double, double, double>(false, false, alpha, B, V, beta, BV);
      Matrix<double> BVD(n, n, Location::kHOST);
      gemm<double, double, double>(false, false, alpha, BV, D, beta, BVD);
      
      max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         for(int j = 0; j < n; j++) {
            max_diff = std::max(max_diff, std::abs(AV(i, j) - BVD(i, j)));
         }
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-10);
   }

   // Test single precision on CPU
   {
      Matrix<float> A(n, n, Location::kHOST);
      Matrix<float> B(n, n, Location::kHOST);
      A.fill(0.0);
      B.fill(0.0);
      
      // Create a tridiagonal matrix with 2 on diagonal and -1 on super- and sub-diagonal
      for(int i = 0; i < n; i++) {
         A(i, i) = 2.0;
         if(i < n - 1) {
            A(i, i + 1) = -1.0;
            A(i + 1, i) = -1.0;
         }
      }
      Matrix<float> V(A);

      // Create a B matrix with diagonal 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
      for(int i = 0; i < n; i++) {
         B(i, i) = i + 1;
      }
      Matrix<float> B_chol(B);
      
      // The true eigenvalues are 
      // 0.0141309446511971
      // 0.0594567850513288
      // 0.131202907506177
      // 0.220848003169762
      // 0.316795717268112
      // 0.410267468002017
      // 0.52158439013008
      // 0.705660103820319
      // 1.08997474266915
      // 2.38801544566837
      float true_eigenvalues[n] = {
         0.0141309446511971,
         0.0594567850513288,
         0.131202907506177,
         0.220848003169762,
         0.316795717268112,
         0.410267468002017,
         0.52158439013008,
         0.705660103820319,
         1.08997474266915,
         2.38801544566837
      };

      Vector<float> W(n, Location::kHOST);
      
      float work_query;
      int lwork = -1;
      MSVDStatus status = sygv(V, B_chol, W, &work_query, lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      lwork = static_cast<int>(work_query);
      Vector<float> work(lwork, Location::kHOST);
      status = sygv(V, B_chol, W, work.data(), lwork);
      EXPECT_EQ(status, MSVDStatus::kSuccess);
      float max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         max_diff = std::max(max_diff, std::abs(W[i] - true_eigenvalues[i]));
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-06);

      // Test the eigenvectors, note that we should have A*V close to V*D where D is the diagonal matrix of eigenvalues
      Matrix<float> AV(n, n, Location::kHOST);
      float alpha = 1.0;
      float beta = 0.0;
      gemm<float, float, float>(false, false, alpha, A, V, beta, AV);
      Matrix<float> D(n, n, Location::kHOST);
      D.fill(0.0);
      for(int i = 0; i < n; i++) {
         D(i, i) = true_eigenvalues[i];
      }
      Matrix<float> BV(n, n, Location::kHOST);
      gemm<float, float, float>(false, false, alpha, B, V, beta, BV);
      Matrix<float> BVD(n, n, Location::kHOST);
      gemm<float, float, float>(false, false, alpha, BV, D, beta, BVD);
      
      max_diff = 0.0;
      for(int i = 0; i < n; i++) {
         for(int j = 0; j < n; j++) {
            max_diff = std::max(max_diff, std::abs(AV(i, j) - BVD(i, j)));
         }
      }
      EXPECT_NEAR(max_diff, 0.0, 1e-06);
   }
}

} // namespace msvd 

// Register the global test environment
int main(int argc, char** argv) {
   ::testing::InitGoogleTest(&argc, argv);
   ::testing::AddGlobalTestEnvironment(new msvd::MvopsTestEnvironment());
   return RUN_ALL_TESTS();
} 