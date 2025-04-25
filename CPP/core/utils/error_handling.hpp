#pragma once
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include <string>

/**
 * @file error_handling.hpp
 * @brief Error handling utilities for MSVD library
 * @details Provides error code definitions, error string conversions, and error checking macros \n
 *          Includes support for internal error codes and CUDA libraries (cuBLAS, cuSPARSE, cuSOLVER, and cuRAND)
 */

namespace msvd {

/**
 * @brief Error codes for MSVD operations
 * @details All error codes used across the MSVD package are defined here
 *          to provide a consistent error handling mechanism
 */
enum class MSVDStatus {
   // Success
   kSuccess = 0,
   
   // General errors
   kErrorInvalidArgument = 1000,
   kErrorInvalidOperation = 1001,
   kErrorOutOfMemory = 1002,
   kErrorNotImplemented = 1003,
   kErrorInternalError = 1004,
   kErrorNaN = 1005,            // NaN value detected
   kErrorInf = 1006,            // Inf value detected
   
   // BLAS related errors
   kErrorBLASInvalidLocation = 2000,   // Vectors in different locations
   kErrorBLASMixedPrecisionCPU = 2001, // CPU doesn't support mixed precision
   kErrorBLASHalfPrecisionCPU = 2002,  // CPU doesn't support half precision
   kErrorBLASUnsupportedType = 2003,   // Unsupported data type combination
   kErrorBLASCuBLASFailure = 2004,     // cuBLAS operation failed
   kErrorBLASInvalidSize = 2005,        // Invalid size for operation
   
   // Factorization related errors
   kErrorFactorizationFailed = 3000,
   kErrorSingularMatrix = 3001,
   
   // Solver related errors
   kErrorSolverDivergence = 4000,
   kErrorSolverMaxIterations = 4001,
   
   // IO related errors
   kErrorFileNotFound = 5000,
   kErrorInvalidFormat = 5001,
   
   // CUDA related errors
   kErrorCUDAGeneral = 6000,
   kErrorCUDAMemory = 6001,
   kErrorCUDALaunch = 6002,
   kErrorCUDASync = 6003
};

/**
 * @brief Convert MSVDStatus to string
 * @param status The status code
 * @return Human-readable string description of the status
 */
inline std::string GetStatusString(MSVDStatus status) {
   switch (status) {
      // Success
      case MSVDStatus::kSuccess:
         return "Success";
      
      // General errors
      case MSVDStatus::kErrorInvalidArgument:
         return "Invalid argument";
      case MSVDStatus::kErrorInvalidOperation:
         return "Invalid operation";
      case MSVDStatus::kErrorOutOfMemory:
         return "Out of memory";
      case MSVDStatus::kErrorNotImplemented:
         return "Not implemented";
      case MSVDStatus::kErrorInternalError:
         return "Internal error";
      case MSVDStatus::kErrorNaN:
         return "NaN value detected";
      case MSVDStatus::kErrorInf:
         return "Inf value detected";
      
      // BLAS related errors
      case MSVDStatus::kErrorBLASInvalidLocation:
         return "BLAS error: Vectors must be in the same location (CPU or GPU)";
      case MSVDStatus::kErrorBLASMixedPrecisionCPU:
         return "BLAS error: Mixed precision operations not supported on CPU";
      case MSVDStatus::kErrorBLASHalfPrecisionCPU:
         return "BLAS error: Half precision not supported on CPU";
      case MSVDStatus::kErrorBLASUnsupportedType:
         return "BLAS error: Unsupported data type combination";
      case MSVDStatus::kErrorBLASCuBLASFailure:
         return "BLAS error: cuBLAS operation failed";
      case MSVDStatus::kErrorBLASInvalidSize:
         return "BLAS error: Invalid size for operation";
      
      // Factorization related errors
      case MSVDStatus::kErrorFactorizationFailed:
         return "Factorization failed";
      case MSVDStatus::kErrorSingularMatrix:
         return "Singular matrix detected";
      
      // Solver related errors
      case MSVDStatus::kErrorSolverDivergence:
         return "Solver diverged";
      case MSVDStatus::kErrorSolverMaxIterations:
         return "Solver reached maximum iterations without converging";
      
      // IO related errors
      case MSVDStatus::kErrorFileNotFound:
         return "File not found";
      case MSVDStatus::kErrorInvalidFormat:
         return "Invalid file format";
      
      // CUDA related errors
      case MSVDStatus::kErrorCUDAGeneral:
         return "CUDA error: General failure";
      case MSVDStatus::kErrorCUDAMemory:
         return "CUDA error: Memory operation failed";
      case MSVDStatus::kErrorCUDALaunch:
         return "CUDA error: Kernel launch failed";
      case MSVDStatus::kErrorCUDASync:
         return "CUDA error: Device synchronization failed";
      
      // Default case
      default:
         return "Unknown error code: " + std::to_string(static_cast<int>(status));
   }
}

/**
 * @brief Convert cuBLAS status code to readable string
 * @details Maps cuBLAS error codes to human-readable error messages \n
 *          Useful for debugging and error reporting in cuBLAS operations
 * 
 * @param status The cuBLAS status code to convert
 * @return String representation of the error code
 */
inline const char* cublasGetErrorString(cublasStatus_t status) {
   switch (status) {
      case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
      default: return "CUBLAS_STATUS_UNKNOWN_ERROR";
   }
}

/**
 * @brief Convert cuSPARSE status code to readable string
 * @details Maps cuSPARSE error codes to human-readable error messages \n
 *          Useful for debugging and error reporting in cuSPARSE operations
 * 
 * @param status The cuSPARSE status code to convert
 * @return String representation of the error code
 */
inline const char* cusparseGetErrorString_new(cusparseStatus_t status) {
   switch (status) {
      case CUSPARSE_STATUS_SUCCESS:                   return "CUSPARSE_STATUS_SUCCESS";
      case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
      case CUSPARSE_STATUS_ALLOC_FAILED:             return "CUSPARSE_STATUS_ALLOC_FAILED";
      case CUSPARSE_STATUS_INVALID_VALUE:            return "CUSPARSE_STATUS_INVALID_VALUE";
      case CUSPARSE_STATUS_ARCH_MISMATCH:            return "CUSPARSE_STATUS_ARCH_MISMATCH";
      case CUSPARSE_STATUS_MAPPING_ERROR:            return "CUSPARSE_STATUS_MAPPING_ERROR";
      case CUSPARSE_STATUS_EXECUTION_FAILED:         return "CUSPARSE_STATUS_EXECUTION_FAILED";
      case CUSPARSE_STATUS_INTERNAL_ERROR:           return "CUSPARSE_STATUS_INTERNAL_ERROR";
      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      case CUSPARSE_STATUS_ZERO_PIVOT:               return "CUSPARSE_STATUS_ZERO_PIVOT";
      default: return "CUSPARSE_STATUS_UNKNOWN_ERROR";
   }
}

/**
 * @brief Convert cuSOLVER status code to readable string
 * @details Maps cuSOLVER error codes to human-readable error messages \n
 *          Useful for debugging and error reporting in cuSOLVER operations
 * 
 * @param status The cuSOLVER status code to convert
 * @return String representation of the error code
 */
inline const char* cusolverGetErrorString(cusolverStatus_t status) {
   switch (status) {
      case CUSOLVER_STATUS_SUCCESS:                 return "CUSOLVER_STATUS_SUCCESS";
      case CUSOLVER_STATUS_NOT_INITIALIZED:         return "CUSOLVER_STATUS_NOT_INITIALIZED";
      case CUSOLVER_STATUS_ALLOC_FAILED:           return "CUSOLVER_STATUS_ALLOC_FAILED";
      case CUSOLVER_STATUS_INVALID_VALUE:          return "CUSOLVER_STATUS_INVALID_VALUE";
      case CUSOLVER_STATUS_ARCH_MISMATCH:          return "CUSOLVER_STATUS_ARCH_MISMATCH";
      case CUSOLVER_STATUS_EXECUTION_FAILED:       return "CUSOLVER_STATUS_EXECUTION_FAILED";
      case CUSOLVER_STATUS_INTERNAL_ERROR:         return "CUSOLVER_STATUS_INTERNAL_ERROR";
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      default: return "CUSOLVER_STATUS_UNKNOWN_ERROR";
   }
}

/**
 * @brief Convert cuRAND status code to readable string
 * @details Maps cuRAND error codes to human-readable error messages \n
 *          Useful for debugging and error reporting in cuRAND operations
 * 
 * @param status The cuRAND status code to convert
 * @return String representation of the error code
 */
inline const char* curandGetErrorString(curandStatus_t status) {
   switch (status) {
      case CURAND_STATUS_SUCCESS:                     return "CURAND_STATUS_SUCCESS";
      case CURAND_STATUS_VERSION_MISMATCH:           return "CURAND_STATUS_VERSION_MISMATCH";
      case CURAND_STATUS_NOT_INITIALIZED:            return "CURAND_STATUS_NOT_INITIALIZED";
      case CURAND_STATUS_ALLOCATION_FAILED:          return "CURAND_STATUS_ALLOCATION_FAILED";
      case CURAND_STATUS_TYPE_ERROR:                 return "CURAND_STATUS_TYPE_ERROR";
      case CURAND_STATUS_OUT_OF_RANGE:               return "CURAND_STATUS_OUT_OF_RANGE";
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:  return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
      case CURAND_STATUS_LAUNCH_FAILURE:             return "CURAND_STATUS_LAUNCH_FAILURE";
      case CURAND_STATUS_PREEXISTING_FAILURE:        return "CURAND_STATUS_PREEXISTING_FAILURE";
      case CURAND_STATUS_INITIALIZATION_FAILED:      return "CURAND_STATUS_INITIALIZATION_FAILED";
      case CURAND_STATUS_ARCH_MISMATCH:              return "CURAND_STATUS_ARCH_MISMATCH";
      case CURAND_STATUS_INTERNAL_ERROR:             return "CURAND_STATUS_INTERNAL_ERROR";
      default: return "CURAND_STATUS_UNKNOWN_ERROR";
   }
}

/**
 * @brief Error checking macro for CUDA runtime functions
 * @details Wraps CUDA runtime calls with error checking \n
 *          Throws a std::runtime_error with detailed error information if the call fails \n
 *          The error message includes file name, line number, and error description
 * 
 * @param call The CUDA runtime function call to check
 */
#define CUDA_CHECK(call) \
do { \
   cudaError_t err = call; \
   if (err != cudaSuccess) { \
      std::stringstream ss; \
      ss << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
         << cudaGetErrorString(err) << " (" << err << ")"; \
      throw std::runtime_error(ss.str()); \
   } \
} while(0)

/**
 * @brief Error checking macro for cuBLAS functions
 * @details Wraps cuBLAS calls with error checking \n
 *          Throws a std::runtime_error with detailed error information if the call fails \n
 *          The error message includes file name, line number, and error description
 * 
 * @param call The cuBLAS function call to check
 */
#define CUBLAS_CHECK(call) \
do { \
   cublasStatus_t status = call; \
   if (status != CUBLAS_STATUS_SUCCESS) { \
      std::stringstream ss; \
      ss << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": " \
         << cublasGetErrorString(status) << " (" << status << ")"; \
      throw std::runtime_error(ss.str()); \
   } \
} while(0)

/**
 * @brief Error checking macro for cuSPARSE functions
 * @details Wraps cuSPARSE calls with error checking \n
 *          Throws a std::runtime_error with detailed error information if the call fails \n
 *          The error message includes file name, line number, and error description
 * 
 * @param call The cuSPARSE function call to check
 */
#define CUSPARSE_CHECK(call) \
do { \
   cusparseStatus_t status = call; \
   if (status != CUSPARSE_STATUS_SUCCESS) { \
      std::stringstream ss; \
      ss << "cuSPARSE error in " << __FILE__ << ":" << __LINE__ << ": " \
         << cusparseGetErrorString_new(status) << " (" << status << ")"; \
      throw std::runtime_error(ss.str()); \
   } \
} while(0)

/**
 * @brief Error checking macro for cuSOLVER functions
 * @details Wraps cuSOLVER calls with error checking \n
 *          Throws a std::runtime_error with detailed error information if the call fails \n
 *          The error message includes file name, line number, and error description
 * 
 * @param call The cuSOLVER function call to check
 */
#define CUSOLVER_CHECK(call) \
do { \
   cusolverStatus_t status = call; \
   if (status != CUSOLVER_STATUS_SUCCESS) { \
      std::stringstream ss; \
      ss << "cuSOLVER error in " << __FILE__ << ":" << __LINE__ << ": " \
         << cusolverGetErrorString(status) << " (" << status << ")"; \
      throw std::runtime_error(ss.str()); \
   } \
} while(0)

/**
 * @brief Error checking macro for cuRAND functions
 * @details Wraps cuRAND calls with error checking \n
 *          Throws a std::runtime_error with detailed error information if the call fails \n
 *          The error message includes file name, line number, and error description
 * 
 * @param call The cuRAND function call to check
 */
#define CURAND_CHECK(call) \
do { \
   curandStatus_t status = call; \
   if (status != CURAND_STATUS_SUCCESS) { \
      std::stringstream ss; \
      ss << "cuRAND error in " << __FILE__ << ":" << __LINE__ << ": " \
         << curandGetErrorString(status) << " (" << status << ")"; \
      throw std::runtime_error(ss.str()); \
   } \
} while(0)

} // namespace msvd 