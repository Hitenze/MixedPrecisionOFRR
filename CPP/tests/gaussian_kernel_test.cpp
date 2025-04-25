#include <gtest/gtest.h>
#include "../containers/matrix.hpp"
#include "../testproblems/gaussian_kernel.hpp"
#include <cmath>
#include <random>

namespace msvd {

/**
 * @brief Test fixture for Gaussian Kernel tests
 */
class GaussianKernelTest : public ::testing::Test {
protected:
   void SetUp() override {
      // Common setup for all tests
   }

   void TearDown() override {
      // Common cleanup for all tests
   }
};

/**
 * @brief Test kernel generation and properties
 */
TEST_F(GaussianKernelTest, GenerateAndValidate) {
   // Parameters for the test
   const double f = 1.5;      // Scale factor
   const double l = 2.0;      // Length scale
   const double s = 0.1;      // Noise scale
   const unsigned long long seed = 42;  // Random seed
   const int n = 10;          // Number of points
   const int d = 3;           // Dimension

   // Generate the kernel
   Matrix<double> K = generate_gaussian_kernel<double>(f, l, s, seed, n, d);
   
   // Verify dimensions
   EXPECT_EQ(K.rows(), n);
   EXPECT_EQ(K.cols(), n);
   
   // Verify it's on host
   EXPECT_EQ(K.location(), Location::kHOST);
   
   // Verify symmetry (K[i,j] = K[j,i])
   for (int i = 0; i < n; i++) {
      for (int j = i+1; j < n; j++) {  // Only check upper triangle
         EXPECT_DOUBLE_EQ(K(i, j), K(j, i));
      }
   }
   
   // Verify diagonal elements (should be f^2 * (1 + s^2))
   double expected_diag = f * f * (1.0 + s * s);
   for (int i = 0; i < n; i++) {
      EXPECT_DOUBLE_EQ(K(i, i), expected_diag);
   }
   
   // Generate the same random data points as in the kernel generation function
   Matrix<double> data(n, d, Location::kHOST);
   data.fill_random(seed);
   
   // Scale the data points the same way as in the kernel generation
   double scale_factor = std::pow(static_cast<double>(n), 1.0 / d);
   data.scale(scale_factor);
   
   // Verify off-diagonal elements match the formula: f^2 * exp(-d^2 / 2l^2)
   double l_squared_2 = 2.0 * l * l;
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         if (i != j) {
            // Calculate expected value using the formula
            // Compute Euclidean distance ||x_i - x_j||^2
            double dist_squared = 0.0;
            for (int k = 0; k < d; k++) {
               double diff = data(i, k) - data(j, k);
               dist_squared += diff * diff;
            }
            
            // Compute expected kernel value: f^2 * exp(-dist_squared/(2*l^2))
            double exp_arg = -dist_squared / l_squared_2;
            double expected_value;
            if (exp_arg < -700.0) {  // Avoid underflow in exp()
               expected_value = 0.0;
            } else {
               expected_value = f * f * std::exp(exp_arg);
            }
            
            // Check that the kernel value matches the expected value
            EXPECT_NEAR(K(i, j), expected_value, 1e-10);
            
            // Also verify that off-diagonal elements are in [0, f^2] (original test)
            EXPECT_GE(K(i, j), 0.0);
            EXPECT_LE(K(i, j), f * f);
         }
      }
   }
   
   // Test with device memory
   Matrix<double> K_device = generate_gaussian_kernel<double>(f, l, s, seed, n, d);
   K_device.to_device();
   
   // Transfer back to host for verification
   K_device.to_host();
   
   // Compare with the original matrix (should be exactly the same)
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         EXPECT_DOUBLE_EQ(K(i, j), K_device(i, j));
      }
   }
}

/**
 * @brief Test kernel with different precision types
 */
TEST_F(GaussianKernelTest, PrecisionTypes) {
   // Parameters for the test
   const double f = 1.0;
   const double l = 1.0;
   const double s = 0.05;
   const unsigned long long seed = 123;
   const int n = 5;
   const int d = 2;
   
   // Generate random data points for verification
   Matrix<double> data(n, d, Location::kHOST);
   data.fill_random(seed);
   
   // Scale the data points the same way as in the kernel generation
   double scale_factor = std::pow(static_cast<double>(n), 1.0 / d);
   data.scale(scale_factor);
   
   // Calculate the expected kernel matrix using the formula
   Matrix<double> expected_kernel(n, n, Location::kHOST);
   double l_squared_2 = 2.0 * l * l;
   
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         if (i == j) {
            expected_kernel(i, j) = f * f * (1.0 + s * s);
         } else {
            // Compute squared Euclidean distance
            double dist_squared = 0.0;
            for (int k = 0; k < d; k++) {
               double diff = data(i, k) - data(j, k);
               dist_squared += diff * diff;
            }
            
            // Compute kernel value: f^2 * exp(-dist_squared/(2*l^2))
            double exp_arg = -dist_squared / l_squared_2;
            if (exp_arg < -700.0) {
               expected_kernel(i, j) = 0.0;
            } else {
               expected_kernel(i, j) = f * f * std::exp(exp_arg);
            }
         }
      }
   }
   
   // Generate kernels with different precision
   Matrix<double> K_double = generate_gaussian_kernel<double>(f, l, s, seed, n, d);
   
   // Test double precision against the expected kernel
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         EXPECT_NEAR(K_double(i, j), expected_kernel(i, j), 1e-10);
      }
   }
   
   // Now test with float precision
   Matrix<float> K_float = generate_gaussian_kernel<float>(f, l, s, seed, n, d);
   
   // Test dimensions
   EXPECT_EQ(K_float.rows(), n);
   EXPECT_EQ(K_float.cols(), n);
   
   // Test float precision values against the expected kernel
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         // Use a larger tolerance for float values
         EXPECT_NEAR(K_float(i, j), static_cast<float>(expected_kernel(i, j)), 1e-5);
      }
   }
   
   // Test with half precision
   Matrix<__half> K_half = generate_gaussian_kernel<__half>(f, l, s, seed, n, d);
   
   // Test dimensions
   EXPECT_EQ(K_half.rows(), n);
   EXPECT_EQ(K_half.cols(), n);
   
   // Test half precision values against the expected kernel
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         // Use a much larger tolerance for half precision
         EXPECT_NEAR(__half2float(K_half(i, j)), static_cast<float>(expected_kernel(i, j)), 1e-2);
      }
   }
}

/**
 * @brief Test kernel with invalid parameters
 */
TEST_F(GaussianKernelTest, InvalidParameters) {
   // Test invalid length scale
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, 0.0, 0.1, 42, 10, 3), std::invalid_argument);
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, -1.0, 0.1, 42, 10, 3), std::invalid_argument);
   
   // Test invalid noise scale
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, 1.0, -0.1, 42, 10, 3), std::invalid_argument);
   
   // Test invalid dimensions
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, 1.0, 0.1, 42, 0, 3), std::invalid_argument);
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, 1.0, 0.1, 42, 10, 0), std::invalid_argument);
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, 1.0, 0.1, 42, -1, 3), std::invalid_argument);
   EXPECT_THROW(generate_gaussian_kernel<double>(1.0, 1.0, 0.1, 42, 10, -1), std::invalid_argument);
}

} // namespace msvd

int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
} 