#pragma once
#include "../containers/vector.hpp"
#include "../containers/matrix.hpp"
#include "../core/utils/error_handling.hpp"
#include <cmath>
#include <random>
#include <limits>

namespace msvd {

/**
 * @brief Generate a Gaussian Kernel matrix
 * @param [in] f Scale factor for the entire kernel
 * @param [in] l Length scale parameter
 * @param [in] s Noise scale parameter
 * @param [in] seed Random seed for data generation
 * @param [in] n Number of data points
 * @param [in] d Dimension of each data point
 * @details Generates a Gaussian kernel matrix based on random data points.
 *          The kernel is computed as f^2 * (K + s^2*I), where 
 *          K[i,j] = exp(-||x_i - x_j||^2 / (2*l^2))
 * @return A matrix containing the Gaussian kernel
 */
template<typename T>
Matrix<T> generate_gaussian_kernel(double f, double l, double s, unsigned long long seed, int n, int d);

/**
 * @brief Generate a Gaussian Kernel matrix (double precision specialization)
 * @param [in] f Scale factor for the entire kernel
 * @param [in] l Length scale parameter
 * @param [in] s Noise scale parameter
 * @param [in] seed Random seed for data generation
 * @param [in] n Number of data points
 * @param [in] d Dimension of each data point
 * @details Generates a Gaussian kernel matrix based on random data points.
 *          The kernel is computed as f^2 * (K + s^2*I), where 
 *          K[i,j] = exp(-||x_i - x_j||^2 / (2*l^2))
 * @return A double precision matrix containing the Gaussian kernel
 */
template<>
Matrix<double> generate_gaussian_kernel<double>(double f, double l, double s, unsigned long long seed, int n, int d) {
   // Parameter validation to avoid numerical issues
   if (l <= 0.0) {
      throw std::invalid_argument("Length scale l must be positive");
   }
   if (s < 0.0) {
      throw std::invalid_argument("Noise scale s must be non-negative");
   }
   if (n <= 0 || d <= 0) {
      throw std::invalid_argument("Data dimensions must be positive");
   }
   
   // Generate random data points (on host)
   Matrix<double> data(n, d, Location::kHOST);
   data.fill_random(seed);
   
   // Scale directly by n^(1/d) without changing range
   double scale_factor = std::pow(static_cast<double>(n), 1.0 / d);
   data.scale(scale_factor);
   
   // Create kernel matrix
   Matrix<double> kernel(n, n, Location::kHOST);
   
   // Compute kernel: exp(-||x-y||^2/(2*l^2))
   double l_squared_2 = 2.0 * l * l;
   for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {  // Only compute upper triangular part due to symmetry
         if (i == j) {
            // Add noise on diagonal: K(x,x) + s^2
            kernel(i, j) = 1.0 + s * s;
         } else {
            // Compute squared Euclidean distance ||x_i - x_j||^2
            double dist_squared = 0.0;
            for (int k = 0; k < d; k++) {
               double diff = data(i, k) - data(j, k);
               dist_squared += diff * diff;
            }
            
            // Compute kernel value: exp(-dist_squared/(2*l^2))
            // Guard against potential overflow
            double exp_arg = -dist_squared / l_squared_2;
            if (exp_arg < -700.0) {  // Avoid underflow in exp()
               kernel(i, j) = 0.0;
            } else {
               kernel(i, j) = std::exp(exp_arg);
            }
            
            // Exploit symmetry: K(i,j) = K(j,i)
            kernel(j, i) = kernel(i, j);
         }
      }
   }
   
   // Scale entire kernel by f^2
   kernel.scale(f * f);
   
   return kernel;
}

/**
 * @brief Generate a Gaussian Kernel matrix (non-double precision implementation)
 * @param [in] f Scale factor for the entire kernel
 * @param [in] l Length scale parameter
 * @param [in] s Noise scale parameter
 * @param [in] seed Random seed for data generation
 * @param [in] n Number of data points
 * @param [in] d Dimension of each data point
 * @details Generates a Gaussian kernel matrix in double precision and then casts to the target type.
 *          This ensures maximum numerical accuracy during intermediate calculations.
 * @return A matrix of type T containing the Gaussian kernel
 */
template<typename T>
Matrix<T> generate_gaussian_kernel(double f, double l, double s, unsigned long long seed, int n, int d) {
   // Generate double precision kernel
   Matrix<double> kernel_double = generate_gaussian_kernel<double>(f, l, s, seed, n, d);
   
   // Cast to the target precision and return
   Matrix<T> result = kernel_double.cast<T>();
   return result;
}

} // namespace msvd 