#pragma once
#include <cuda_fp16.h>
#include <type_traits>
#include <limits>
#include <cusparse.h>
#include <cublas_v2.h>

namespace msvd {

/**
 * @brief Get the appropriate CUDA data type for template type T
 * @details Maps C++ types to CUDA data types for use with cuSPARSE and other CUDA libraries \n
 *          Supports double, float, and __half precision types
 * 
 * @tparam T The C++ type to map to CUDA data type
 * @return The corresponding CUDA data type (CUDA_R_64F, CUDA_R_32F, or CUDA_R_16F)
 */ 
template<typename T>
cudaDataType_t get_cuda_datatype() {
   if constexpr (std::is_same_v<T, double>) {
      return CUDA_R_64F;
   } 
   else if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
   } 
   else if constexpr (std::is_same_v<T, __half>) {
      return CUDA_R_16F;
   } 
   else {
      throw std::runtime_error("Unsupported data type");
      return CUDA_R_32F; // Default to float
   }
}

/**
 * @brief Get the appropriate cuBLAS data type for template type T
 * @details Maps C++ types to cuBLAS data types for use with cuBLAS and other CUDA libraries \n
 *          Supports double, float, and __half precision types
 * 
 * @tparam T The C++ type to map to cuBLAS data type
 */
template<typename T>
cudaDataType_t get_cublas_dtype() {
   if constexpr (std::is_same_v<T, double>) {
      return CUDA_R_64F;
   }
   else if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
   }
   else if constexpr (std::is_same_v<T, __half>) {
      return CUDA_R_16F;
   }
   else {
      throw std::runtime_error("Unsupported data type");
      return CUDA_R_32F; // Default to float
   }
}

/**
 * @brief Get the appropriate cuBLAS compute type for template type T
 * @details Maps C++ types to cuBLAS compute types for use with cuBLAS and other CUDA libraries \n
 *          Supports double, float, and __half precision types
 * 
 * @tparam T The C++ type to map to cuBLAS compute type
 */
template<typename T>
cublasComputeType_t get_cublas_compute_type() {
   if constexpr (std::is_same_v<T, double>) {
      return CUBLAS_COMPUTE_64F;
   }
   else if constexpr (std::is_same_v<T, float>) {
      return CUBLAS_COMPUTE_32F;
   }
   else if constexpr (std::is_same_v<T, __half>) {
      return CUBLAS_COMPUTE_16F;
   }
   else {
      throw std::runtime_error("Unsupported data type");
      return CUBLAS_COMPUTE_32F; // Default to float
   }
}

/**
 * @brief Get the appropriate cuBLAS operation type for a given boolean value
 * @details Maps a boolean value to the corresponding cuBLAS operation type \n
 *          Supports boolean values (true for transpose, false for no transpose)
 * 
 * @param trans Boolean value indicating whether to use transpose operation
 * @return The corresponding cuBLAS operation type (CUBLAS_OP_T or CUBLAS_OP_N)
 */
cublasOperation_t inline get_cublas_transpose(bool trans) {
   return trans ? CUBLAS_OP_T : CUBLAS_OP_N;
}

/**
 * @brief Get the appropriate CUDA data type for template type T
 * @details Maps C++ types to CUDA data types for use with cuSPARSE and other CUDA libraries \n
 *          Supports double, float, and __half precision types
 * 
 * @tparam T The C++ type to map to CUDA data type
 * @return The corresponding CUDA data type (CUDA_R_64F, CUDA_R_32F, or CUDA_R_16F)
 */
template <typename T>
cudaDataType get_cusparse_dtype() {
   if constexpr (std::is_same_v<T, double>) {
      return CUDA_R_64F;
   } else if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
   } else if constexpr (std::is_same_v<T, __half>) {
      return CUDA_R_16F;
   }
   return CUDA_R_32F; // default case
}

/**
 * @brief Get machine epsilon for the given type
 * @details Returns the smallest number ε such that 1 + ε != 1 for the specified type \n
 *          Uses standard library epsilon for double and float \n
 *          For half precision, uses an approximate value of 2^-10
 * 
 * @tparam T Type for which to get epsilon (double, float, or __half)
 * @return Machine epsilon for the specified type
 */
template<typename T>
T get_eps() 
{
   if constexpr (std::is_same_v<T, double>) 
   {
      return std::numeric_limits<double>::epsilon();
   }
   else if constexpr (std::is_same_v<T, float>) 
   {
      return std::numeric_limits<float>::epsilon();
   }
   else if constexpr (std::is_same_v<T, __half>) 
   {
      // Half precision eps ≈ 2^-10 ≈ 0.00097656
      return __float2half(0.00097656f);
   }
}

/**
 * @brief Get the value 1 in the appropriate type
 * @details Returns the constant 1 represented in the specified type \n
 *          Handles double, float, and __half precision types
 * 
 * @tparam T Type in which to represent 1 (double, float, or __half)
 * @return The value 1 in the specified type
 */
template <typename T>
T get_one() {
   if constexpr (std::is_same_v<T, double>) {
      return 1.0;
   } else if constexpr (std::is_same_v<T, float>) {
      return 1.0f;
   } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(1.0f);
   } else {
      throw std::runtime_error("Unsupported data type");
      return T(1.0f);
   }
}

/**
 * @brief Get the value 0 in the appropriate type
 * @details Returns the constant 0 represented in the specified type \n
 *          Handles double, float, and __half precision types
 * 
 * @tparam T Type in which to represent 0 (double, float, or __half)
 * @return The value 0 in the specified type
 */
template <typename T>
T get_zero() {
   if constexpr (std::is_same_v<T, double>) {
      return 0.0;
   } else if constexpr (std::is_same_v<T, float>) {
      return 0.0f;
   } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(0.0f);
   } else {
      throw std::runtime_error("Unsupported data type");
      return T(0);
   }
}

template <typename T>
T get_negone() {
   if constexpr (std::is_same_v<T, double>) {
      return -1.0;
   } else if constexpr (std::is_same_v<T, float>) {
      return -1.0f;
   } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(-1.0f);
   } else {
      throw std::runtime_error("Unsupported data type");
      return T(-1.0f);
   }
}

/**
 * @brief Get the value 1/sqrt(2) in the appropriate type
 * @details Returns the constant 1/sqrt(2) represented in the specified type \n
 *          Handles double, float, and __half precision types
 * 
 * @tparam T Type in which to represent 1/sqrt(2) (double, float, or __half)
 * @return The value 1/sqrt(2) in the specified type
 */
template <typename T>
T get_one_over_sqrt_two() {
   if constexpr (std::is_same_v<T, double>) {
      return 1.0 / std::sqrt(2.0);
   } else if constexpr (std::is_same_v<T, float>) {
      return 1.0f / std::sqrt(2.0f);
   } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(1.0f / std::sqrt(2.0f));
   } else {
      throw std::runtime_error("Unsupported data type");
      return T(1.0f / std::sqrt(2.0f));
   }
}

/**
 * @brief Device function to compute absolute value for double
 * @details CUDA device implementation of absolute value function for double precision
 * 
 * @param x Input value
 * @return Absolute value of x
 */
__device__ inline double abs_val(double x) {
   return fabs(x);
}

/**
 * @brief Device function to compute absolute value for float
 * @details CUDA device implementation of absolute value function for single precision
 * 
 * @param x Input value
 * @return Absolute value of x
 */
__device__ inline float abs_val(float x) {
   return fabsf(x);
}

/**
 * @brief Device function to compute absolute value for __half
 * @details CUDA device implementation of absolute value function for half precision
 * 
 * @param x Input value
 * @return Absolute value of x
 */
__device__ inline __half abs_val(__half x) {
   return __habs(x);
}

/**
 * @brief Device function to check if double value is NaN
 * @details CUDA device implementation to test if a double precision value is Not-a-Number
 * 
 * @param x Input value to test
 * @return true if x is NaN, false otherwise
 */
__device__ inline bool isnan_val(double x) {
   return isnan(x);
}

/**
 * @brief Device function to check if float value is NaN
 * @details CUDA device implementation to test if a single precision value is Not-a-Number
 * 
 * @param x Input value to test
 * @return true if x is NaN, false otherwise
 */
__device__ inline bool isnan_val(float x) {
   return isnan(x);
}

/**
 * @brief Device function to check if __half value is NaN
 * @details CUDA device implementation to test if a half precision value is Not-a-Number
 * 
 * @param x Input value to test
 * @return true if x is NaN, false otherwise
 */
__device__ inline bool isnan_val(__half x) {
   return __hisnan(x);
}

/**
 * @brief Device function to check if double value is Inf
 * @details CUDA device implementation to test if a double precision value is Infinity
 * 
 * @param x Input value to test
 * @return true if x is Inf, false otherwise
 */
__device__ inline bool isinf_val(double x) {
   return isinf(x);
}

/**
 * @brief Device function to check if float value is Inf
 * @details CUDA device implementation to test if a single precision value is Infinity
 * 
 * @param x Input value to test
 * @return true if x is Inf, false otherwise
 */
__device__ inline bool isinf_val(float x) {
   return isinf(x);
}

/**
 * @brief Device function to check if __half value is Inf
 * @details CUDA device implementation to test if a half precision value is Infinity
 * 
 * @param x Input value to test
 * @return true if x is Inf, false otherwise
 */
__device__ inline bool isinf_val(__half x) {
   return __hisinf(x);
}

} // namespace msvd 