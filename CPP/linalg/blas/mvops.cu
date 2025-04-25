#include "mvops.hpp"
#include "../../core/utils/cuda_handler.hpp"
#include "../../core/utils/type_utils.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <algorithm>

// Include different BLAS headers based on compilation options
#if defined(USE_OPENBLAS)
   #include <cblas.h>
   #include <lapacke.h>
#elif defined(USE_MKL)
   #include <mkl.h>
#else
   #include <cblas.h>
   #include <lapacke.h>
#endif

namespace msvd {

/**
 * @brief Get the appropriate CBLAS operation type for a given boolean value
 * @details Maps a boolean value to the corresponding CBLAS operation type \n
 *          Supports boolean values (true for transpose, false for no transpose)
 * @note A local helper function, I am lazy to put this input type_utils.hpp since that requires a lot of changes
 * @param trans Boolean value indicating whether to use transpose operation
 * @return The corresponding CBLAS operation type
 */
CBLAS_TRANSPOSE get_cblas_transpose(bool trans) {
   return trans ? CblasTrans : CblasNoTrans;
}

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus dot(const Vector<T_I>& x, const Vector<T_I>& y, T_O& result) {
   // 1. Check that both vectors are in the same location (CPU or GPU)
   if (x.location() != y.location()) {
      throw std::runtime_error("Both vectors must be in the same location (CPU or GPU)");
      return MSVDStatus::kErrorBLASInvalidLocation;
   }
   
   // Check that both vectors are of the same size
   if (x.length() != y.length()) {
      throw std::runtime_error("Both vectors must be of the same size");
      return MSVDStatus::kErrorBLASInvalidSize;
   }

   int n = x.length();
   
   // Check for empty vectors, return 0 if empty
   if (n <= 0) {
      result = get_zero<T_O>();
      return MSVDStatus::kSuccess;
   }

   // 2. Handle based on location
   if (x.location() == Location::kHOST) {
      // CPU implementation using CBLAS
      
      // CPU implementation requires all types to be the same
      if constexpr (!std::is_same_v<T_I, T_O> || !std::is_same_v<T_I, T_COMPUTE>) {
         throw std::runtime_error("For CPU dot product, input, output, and compute types must be the same");
         return MSVDStatus::kErrorBLASMixedPrecisionCPU;
      }
      
      // Use CBLAS for CPU implementation
      if constexpr (std::is_same_v<T_I, double>) {
         // Double precision
         T_O sum = cblas_ddot(n, x.data(), 1, y.data(), 1);
         result = sum;
      } 
      else if constexpr (std::is_same_v<T_I, float>) {
         // Single precision
         T_O sum = cblas_sdot(n, x.data(), 1, y.data(), 1);
         result = sum;
      }
      else {
         // Half precision not supported in CPU
         throw std::runtime_error("Half precision not supported for CPU dot product");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
   } 
   else {
      // GPU implementation using cuBLAS
      try {
         cublasHandle_t handle = CUDAHandler::cublas();
         
         if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
            // Double precision
            CUBLAS_CHECK(cublasDdot(handle, n, x.data(), 1, y.data(), 1, &result));
         } 
         else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
            // Single precision
            CUBLAS_CHECK(cublasSdot(handle, n, x.data(), 1, y.data(), 1, &result));
         } 
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, float>) {
            // Call cuBLAS with float result for all half precision operations
            CUBLAS_CHECK(cublasDotEx(handle, n, 
                                    x.data(), CUDA_R_16F, 1,
                                    y.data(), CUDA_R_16F, 1,
                                    &result, CUDA_R_16F,
                                    CUDA_R_32F));
         }
         else {
            throw std::runtime_error("Unsupported type combination for dot product");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus norm2(const Vector<T_I>& x, T_O& result) {
   // 1. Check vector validity
   int n = x.length();
   
   // Check for empty vector, return 0 if empty
   if (n <= 0) {
      result = get_zero<T_O>();
      return MSVDStatus::kSuccess;
   }

   // 2. Handle based on location
   if (x.location() == Location::kHOST) {
      // CPU implementation
      
      // CPU implementation requires all types to be the same
      if constexpr (!std::is_same_v<T_I, T_O> || !std::is_same_v<T_I, T_COMPUTE>) {
         throw std::runtime_error("For CPU norm2, input, output, and compute types must be the same");
         return MSVDStatus::kErrorBLASMixedPrecisionCPU;
      }
      
      // Use CBLAS for CPU implementation
      if constexpr (std::is_same_v<T_I, double>) {
         // Double precision
         T_O sum = cblas_dnrm2(n, x.data(), 1);
         result = sum;
      } 
      else if constexpr (std::is_same_v<T_I, float>) {
         // Single precision
         T_O sum = cblas_snrm2(n, x.data(), 1);
         result = sum;
      }
      else {
         // Half precision not supported in CPU
         throw std::runtime_error("Half precision not supported for CPU norm2");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
   } 
   else {
      // GPU implementation using cuBLAS
      try {
         cublasHandle_t handle = CUDAHandler::cublas();
         
         if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
            // Double precision
            CUBLAS_CHECK(cublasDnrm2(handle, n, x.data(), 1, &result));
         } 
         else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
            // Single precision
            CUBLAS_CHECK(cublasSnrm2(handle, n, x.data(), 1, &result));
         } 
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, float>) {
            // Half precision IO, float compute
            CUBLAS_CHECK(cublasNrm2Ex(handle, n, 
                                     x.data(), CUDA_R_16F, 1,
                                     &result, CUDA_R_16F,
                                     CUDA_R_32F));
         }
         else {
            throw std::runtime_error("Unsupported type combination for norm2");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for common type combinations
template MSVDStatus dot<double, double, double>(const Vector<double>& x, const Vector<double>& y, double& result);
template MSVDStatus dot<float, float, float>(const Vector<float>& x, const Vector<float>& y, float& result);
template MSVDStatus dot<__half, __half, float>(const Vector<__half>& x, const Vector<__half>& y, __half& result);

// Explicit instantiation for norm2
template MSVDStatus norm2<double, double, double>(const Vector<double>& x, double& result);
template MSVDStatus norm2<float, float, float>(const Vector<float>& x, float& result);
template MSVDStatus norm2<__half, __half, float>(const Vector<__half>& x, __half& result);

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus gemv(bool trans, 
                const T_COMPUTE& alpha, const Matrix<T_I>& A, const Vector<T_I>& x, 
                const T_COMPUTE& beta, Vector<T_O>& y) {
   
   // Get matrix dimensions using public getters
   const int m = A.rows();
   const int n = A.cols();
   const int lda = A.ld();
   
   // Initial checks
   // Check that both vectors and matrix are in the same location
   if (x.location() != y.location() || x.location() != A.location()) {
      throw std::runtime_error("Vectors and matrix must all be in the same location (CPU or GPU)");
      return MSVDStatus::kErrorBLASInvalidLocation;
   }
   
   // Check dimensions
   const int x_dim = trans ? m : n;
   const int y_dim = trans ? n : m;
   
   if (x.length() < x_dim || y.length() < y_dim) {
      throw std::runtime_error("Vector dimensions do not match matrix dimensions");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   // Process based on location
   if (x.location() == Location::kHOST) {
      // CPU implementation using CBLAS
      
      // CPU implementation requires all types to be the same
      if constexpr (!std::is_same_v<T_I, T_O> || !std::is_same_v<T_I, T_COMPUTE>) {
         throw std::runtime_error("For CPU gemv, input, output, and compute types must be the same");
         return MSVDStatus::kErrorBLASMixedPrecisionCPU;
      }
      
      // Half precision not supported on CPU
      if constexpr (std::is_same_v<T_I, __half>) {
         throw std::runtime_error("Half precision not supported for CPU gemv operation");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
      
      CBLAS_TRANSPOSE trans_op = get_cblas_transpose(trans);
      if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
         // Double precision
         cblas_dgemv(
               CblasColMajor,
               trans_op,
               m, n,
               alpha,
               A.data(), lda,
               x.data(), 1,
               beta,
               y.data(), 1
         );
      } 
      else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
         // Single precision
         cblas_sgemv(
               CblasColMajor,
               trans_op,
               m, n,
               alpha,
               A.data(), lda,
               x.data(), 1,
               beta,
               y.data(), 1
         );
      }
      else {
         throw std::runtime_error("Unsupported type combination for CPU gemv");
         return MSVDStatus::kErrorBLASUnsupportedType;
      }
   } 
   else {
      // GPU implementation using cuBLAS
      try {
         cublasHandle_t handle = CUDAHandler::cublas();
         cublasOperation_t trans_op = get_cublas_transpose(trans);
         
         if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
            // Double precision
            CUBLAS_CHECK(cublasDgemv(
               handle,
               trans_op,
               m, n,
               &alpha,
               A.data(), lda,
               x.data(), 1,
               &beta,
               y.data(), 1
            ));
         } 
         else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
            // Single precision
            CUBLAS_CHECK(cublasSgemv(
               handle,
               trans_op,
               m, n,
               &alpha,
               A.data(), lda,
               x.data(), 1,
               &beta,
               y.data(), 1
            ));
         }
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, float>) {
            // Half precision IO, float compute on GPU
            // Implement GEMV as a GEMM with a single column/row in second matrix
            cublasOperation_t vec_trans_op = CUBLAS_OP_N; // Vector is always non-transposed
            cudaDataType_t i_type = get_cublas_dtype<T_I>();
            cudaDataType_t o_type = get_cublas_dtype<T_O>();
            cublasComputeType_t c_type = get_cublas_compute_type<T_COMPUTE>();
            
            // For GEMM: C(m,1) = alpha * A(m,k) * B(k,1) + beta * C(m,1)
            if (trans) {
               // y(n,1) = alpha * A^T(n,m) * x(m,1) + beta * y(n,1)
               CUBLAS_CHECK(cublasGemmEx(
                  handle,
                  trans_op,      // transA = T
                  vec_trans_op,  // transB = N (vector as column)
                  y_dim, 1, x_dim,  // m=y_dim, n=1, k=x_dim
                  &alpha,
                  A.data(), i_type, lda,
                  x.data(), i_type, x_dim,  // leading dimension of x as a matrix
                  &beta,
                  y.data(), o_type, y_dim,  // leading dimension of y as a matrix
                  c_type,
                  CUBLAS_GEMM_DEFAULT
               ));
            } else {
               // y(m,1) = alpha * A(m,n) * x(n,1) + beta * y(m,1)
               CUBLAS_CHECK(cublasGemmEx(
                  handle,
                  trans_op,      // transA = N
                  vec_trans_op,  // transB = N (vector as column)
                  y_dim, 1, x_dim,  // m=y_dim, n=1, k=x_dim
                  &alpha,
                  A.data(), i_type, lda,
                  x.data(), i_type, x_dim,  // leading dimension of x as a matrix
                  &beta,
                  y.data(), o_type, y_dim,  // leading dimension of y as a matrix
                  c_type,
                  CUBLAS_GEMM_DEFAULT
               ));
            }
         }
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, __half>) {
            // Half precision IO, half compute on GPU
            cublasOperation_t vec_trans_op = CUBLAS_OP_N; // Vector is always non-transposed
            cudaDataType_t i_type = get_cublas_dtype<T_I>();
            cudaDataType_t o_type = get_cublas_dtype<T_O>();
            cublasComputeType_t c_type = get_cublas_compute_type<T_COMPUTE>();
            
            // For GEMM: C(m,1) = alpha * A(m,k) * B(k,1) + beta * C(m,1)
            if (trans) {
               // y(n,1) = alpha * A^T(n,m) * x(m,1) + beta * y(n,1)
               CUBLAS_CHECK(cublasGemmEx(
                  handle,
                  trans_op,      // transA = T
                  vec_trans_op,  // transB = N (vector as column)
                  y_dim, 1, x_dim,  // m=y_dim, n=1, k=x_dim
                  &alpha,
                  A.data(), i_type, lda,
                  x.data(), i_type, x_dim,  // leading dimension of x as a matrix
                  &beta,
                  y.data(), o_type, y_dim,  // leading dimension of y as a matrix
                  c_type,
                  CUBLAS_GEMM_DEFAULT
               ));
            } else {
               // y(m,1) = alpha * A(m,n) * x(n,1) + beta * y(m,1)
               CUBLAS_CHECK(cublasGemmEx(
                  handle,
                  trans_op,      // transA = N
                  vec_trans_op,  // transB = N (vector as column)
                  y_dim, 1, x_dim,  // m=y_dim, n=1, k=x_dim
                  &alpha,
                  A.data(), i_type, lda,
                  x.data(), i_type, x_dim,  // leading dimension of x as a matrix
                  &beta,
                  y.data(), o_type, y_dim,  // leading dimension of y as a matrix
                  c_type,
                  CUBLAS_GEMM_DEFAULT
               ));
            }
         }
         else {
            throw std::runtime_error("Unsupported type combination for GPU gemv");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus gemv<double, double, double>(
   bool trans, 
   const double& alpha, const Matrix<double>& A, const Vector<double>& x, 
   const double& beta, Vector<double>& y
);

template MSVDStatus gemv<float, float, float>(
   bool trans, 
   const float& alpha, const Matrix<float>& A, const Vector<float>& x, 
   const float& beta, Vector<float>& y
);

template MSVDStatus gemv<__half, __half, float>(
   bool trans, 
   const float& alpha, const Matrix<__half>& A, const Vector<__half>& x, 
   const float& beta, Vector<__half>& y
);

template MSVDStatus gemv<__half, __half, __half>(
   bool trans, 
   const __half& alpha, const Matrix<__half>& A, const Vector<__half>& x, 
   const __half& beta, Vector<__half>& y
);

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus gemm(bool transA, bool transB,
                const T_COMPUTE& alpha, const Matrix<T_I>& A, const Matrix<T_I>& B,
                const T_COMPUTE& beta, Matrix<T_O>& C) {
   
   // Get matrix dimensions
   const int m = transA ? A.cols() : A.rows();
   const int k = transA ? A.rows() : A.cols();
   const int n = transB ? B.rows() : B.cols();
   const int k2 = transB ? B.cols() : B.rows();
   
   // Initial checks
   // Check that all matrices are in the same location
   if (A.location() != B.location() || A.location() != C.location()) {
      throw std::runtime_error("All matrices must be in the same location (CPU or GPU)");
      return MSVDStatus::kErrorBLASInvalidLocation;
   }
   
   // Check dimensions
   if (k != k2) {
      throw std::runtime_error("Matrix dimensions do not match for multiplication");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   if (C.rows() != m || C.cols() != n) {
      throw std::runtime_error("Output matrix C has wrong dimensions");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   // Process based on location
   if (A.location() == Location::kHOST) {
      // CPU implementation using CBLAS
      // CPU implementation requires all types to be the same
      if constexpr (!std::is_same_v<T_I, T_O> || !std::is_same_v<T_I, T_COMPUTE>) {
         throw std::runtime_error("For CPU GEMM, input, output, and compute types must be the same");
         return MSVDStatus::kErrorBLASMixedPrecisionCPU;
      }

      CBLAS_TRANSPOSE transA_op = get_cblas_transpose(transA);
      CBLAS_TRANSPOSE transB_op = get_cblas_transpose(transB);
      
      // Check if types are supported
      if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
         // Double precision
         cblas_dgemm(
            CblasColMajor,
            transA_op,
            transB_op,
            m, n, k,
            alpha,
            A.data(), A.ld(),
            B.data(), B.ld(),
            beta,
            C.data(), C.ld()
         );
      } 
      else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
         // Single precision
         cblas_sgemm(
            CblasColMajor,
            transA_op,
            transB_op,
            m, n, k,
            alpha,
            A.data(), A.ld(),
            B.data(), B.ld(),
            beta,
            C.data(), C.ld()
         );
      }
      else {
         throw std::runtime_error("Unsupported type combination for CPU GEMM");
         return MSVDStatus::kErrorBLASUnsupportedType;
      }
   } 
   else {
      // GPU implementation using cuBLAS
      try {
         cublasHandle_t handle = CUDAHandler::cublas();
         cublasOperation_t transA_op = get_cublas_transpose(transA);
         cublasOperation_t transB_op = get_cublas_transpose(transB);
         
         // Double precision on GPU
         if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
            CUBLAS_CHECK(cublasDgemm(
               handle,
               transA_op,
               transB_op,
               m, n, k,
               &alpha,
               A.data(), A.ld(),
               B.data(), B.ld(),
               &beta,
               C.data(), C.ld()
            ));
         }
         // Single precision on GPU
         else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
            CUBLAS_CHECK(cublasSgemm(
               handle,
               transA_op,
               transB_op,
               m, n, k,
               &alpha,
               A.data(), A.ld(),
               B.data(), B.ld(),
               &beta,
               C.data(), C.ld()
            ));
         }
         // Half precision IO, float compute on GPU
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, float>) {
            cudaDataType_t i_type = get_cublas_dtype<T_I>();
            cudaDataType_t o_type = get_cublas_dtype<T_O>();
            cublasComputeType_t c_type = get_cublas_compute_type<T_COMPUTE>();
            CUBLAS_CHECK(cublasGemmEx(
               handle,
               transA_op,
               transB_op,
               m, n, k,
               &alpha,
               A.data(), i_type, A.ld(),
               B.data(), i_type, B.ld(),
               &beta,
               C.data(), o_type, C.ld(),
               c_type,
               CUBLAS_GEMM_DEFAULT
            ));
         }
         // Half precision IO, half compute on GPU
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, __half>) {
            cudaDataType_t i_type = get_cublas_dtype<T_I>();
            cudaDataType_t o_type = get_cublas_dtype<T_O>();
            cublasComputeType_t c_type = get_cublas_compute_type<T_COMPUTE>();
            CUBLAS_CHECK(cublasGemmEx(
               handle,
               transA_op,
               transB_op,
               m, n, k,
               &alpha,
               A.data(), i_type, A.ld(),
               B.data(), i_type, B.ld(),
               &beta,
               C.data(), o_type, C.ld(),
               c_type,
               CUBLAS_GEMM_DEFAULT
            ));
         }
         else {
            throw std::runtime_error("Unsupported type combination for GPU GEMM");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus gemm<double, double, double>(
   bool transA, bool transB,
   const double& alpha, const Matrix<double>& A, const Matrix<double>& B,
   const double& beta, Matrix<double>& C
);

template MSVDStatus gemm<float, float, float>(
   bool transA, bool transB,
   const float& alpha, const Matrix<float>& A, const Matrix<float>& B,
   const float& beta, Matrix<float>& C
);

template MSVDStatus gemm<__half, __half, float>(
   bool transA, bool transB,
   const float& alpha, const Matrix<__half>& A, const Matrix<__half>& B,
   const float& beta, Matrix<__half>& C
);

template MSVDStatus gemm<__half, __half, __half>(
   bool transA, bool transB,
   const __half& alpha, const Matrix<__half>& A, const Matrix<__half>& B,
   const __half& beta, Matrix<__half>& C
);

template <unsigned int blockSize, typename T>
__device__ void reduceMaxAbsInBlock( T* sdata, int* sindices) {
   unsigned int tid = threadIdx.x;

   if constexpr (blockSize >= 1024) {
      if (tid < 512) {
         if constexpr (std::is_same_v<T, __half>) {
            if (__hgt(sdata[tid + 512], sdata[tid])) {
               sdata[tid] = sdata[tid + 512];
               sindices[tid] = sindices[tid + 512];
            }
         } else {
            if (sdata[tid + 512] > sdata[tid]) {
               sdata[tid] = sdata[tid + 512];
               sindices[tid] = sindices[tid + 512];
            }
         }
      }
      __syncthreads();
   }
   if constexpr (blockSize >= 512) {
      if (tid < 256) {
         if constexpr (std::is_same_v<T, __half>) {
            if (__hgt(sdata[tid + 256], sdata[tid])) {
               sdata[tid] = sdata[tid + 256];
               sindices[tid] = sindices[tid + 256];
            }
         } else {
            if (sdata[tid + 256] > sdata[tid]) {
               sdata[tid] = sdata[tid + 256];
               sindices[tid] = sindices[tid + 256];
            }
         }
      }
      __syncthreads();
   }
   if constexpr (blockSize >= 256) {
      if (tid < 128) {
         if constexpr (std::is_same_v<T, __half>) {
            if (__hgt(sdata[tid + 128], sdata[tid])) {
               sdata[tid] = sdata[tid + 128];
               sindices[tid] = sindices[tid + 128];
            }
         } else {
            if (sdata[tid + 128] > sdata[tid]) {
               sdata[tid] = sdata[tid + 128];
               sindices[tid] = sindices[tid + 128];
            }
         }
      }
      __syncthreads();
   }
   if constexpr (blockSize >= 128) {
      if (tid < 64) {
         if constexpr (std::is_same_v<T, __half>) {
            if (__hgt(sdata[tid + 64], sdata[tid])) {
               sdata[tid] = sdata[tid + 64];
               sindices[tid] = sindices[tid + 64];
            }
         } else {
            if (sdata[tid + 64] > sdata[tid]) {
               sdata[tid] = sdata[tid + 64];
               sindices[tid] = sindices[tid + 64];
            }
         }
      }
      __syncthreads();
   }

   if (tid < 32) {
      if constexpr (blockSize >= 64) { 
         if constexpr (std::is_same_v<T, __half>) {
            if (__hgt(sdata[tid + 32], sdata[tid])) {
               sdata[tid] = sdata[tid + 32];
               sindices[tid] = sindices[tid + 32];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
               __half other_val = __shfl_down_sync(0xffffffff, sdata[tid], offset);
               int other_idx = __shfl_down_sync(0xffffffff, sindices[tid], offset);
               if (__hgt(other_val, sdata[tid])) {
                     sdata[tid] = other_val;
                     sindices[tid] = other_idx;
               }
            }
         } else {
            if (sdata[tid + 32] > sdata[tid]) {
               sdata[tid] = sdata[tid + 32];
               sindices[tid] = sindices[tid + 32];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
               T other_val = __shfl_down_sync(0xffffffff, sdata[tid], offset);
               int other_idx = __shfl_down_sync(0xffffffff, sindices[tid], offset);
               if (other_val > sdata[tid]) {
                     sdata[tid] = other_val;
                     sindices[tid] = other_idx;
               }
            }
         }
      }
   }
}

// First level reduction kernel for half precision
// blockSize must be a power of 2!
template <unsigned int blockSize, typename T, typename T_COMPUTE>
__global__ void findMaxAbsKernel(const T* input, T* partial_max, int* partial_indices, int n, T_COMPUTE* final_val, bool write_back) {
   __shared__ T sdata[blockSize];
   __shared__ int sindices[blockSize];
   
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockSize + tid;
   
   T thread_max;
   if constexpr (std::is_same_v<T, __half>) {
      thread_max = __half(0.0f);
   } else {
      thread_max = T(0.0f);
   }
   int thread_idx = -1;
   
   // Data loading
   if (i < n) {
      thread_max = abs_val(input[i]);
      thread_idx = i;
   }
   
   sdata[tid] = thread_max;
   sindices[tid] = thread_idx;
   __syncthreads();
   
   reduceMaxAbsInBlock<blockSize, T>(sdata, sindices);
   
   // Write result for this block to global memory
   // shift index by one to make it 1-based
   if (tid == 0) {
      partial_max[blockIdx.x] = sdata[0];
      partial_indices[blockIdx.x] = sindices[0] + 1;
      if (write_back) {
         if constexpr (std::is_same_v<T, __half> && std::is_same_v<T_COMPUTE, float>) {
            *final_val = __half2float(input[sindices[0]]);
         } else {
            *final_val = input[sindices[0]];
         }
      }
   }
}

// Note that in this kernel no absolute value is needed
// also no need to shift index by one
template <unsigned int blockSize, typename T, typename T_COMPUTE>
__global__ void findMaxAbsKernel_level2(const T* input, const T* partial_max, const int* partial_indices, 
                                           int n, T* final_max, int* final_index,
                                           T_COMPUTE* final_val, bool write_back) {
   __shared__ T sdata[blockSize];
   __shared__ int sindices[blockSize];
   
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockSize + tid;
   
   T thread_max;
   if constexpr (std::is_same_v<T, __half>) {
      thread_max = __half(0.0f);
   } else {
      thread_max = T(0.0f);
   }
   int max_idx = -1;
   
   if (i < n) {
      thread_max = partial_max[i];
      max_idx = partial_indices[i];
   }
   
   sdata[tid] = thread_max;
   sindices[tid] = max_idx;
   __syncthreads();
   
   reduceMaxAbsInBlock<blockSize, T>(sdata, sindices);

   if (tid == 0) {
      final_max[blockIdx.x] = sdata[0];
      final_index[blockIdx.x] = sindices[0];
      if (write_back) {
         if constexpr (std::is_same_v<T, __half> && std::is_same_v<T_COMPUTE, float>) {
            *final_val = __half2float(input[sindices[0] - 1]);
         } else {
            *final_val = input[sindices[0] - 1];
         }
      }
   }
}

// Note: this version only handels case when threadPerBlock is some power of 2
template <typename T, typename T_COMPUTE>
int findMaxAbsIndex(const T* d_input, int n, int* result, T_COMPUTE* result_val, Location location) {
   // Can be modified to see if better performance can be achieved on different GPUs
   // For our paper we fixed this to 1024
   // result_val must be device pointer
   constexpr int threadsPerBlock = 1024;

   if (threadsPerBlock <= 0 || ((threadsPerBlock & (threadsPerBlock - 1)) != 0)) {
      throw std::runtime_error("threadsPerBlock must be a power of 2 and greater than 0.");
      return 0;
   }
   
   // First level reduction, note that 
   // since n at most is 2^31 - 1, we won't exceed the max number of blocks
   // for cuda launch, so no any check here.
   int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
   
   T_COMPUTE* d_result_val;
   if (location == Location::kDEVICE) {
      d_result_val = result_val;
   } else {
      CUDA_CHECK(cudaMalloc(&d_result_val, sizeof(T_COMPUTE)));
   }
   T* d_partial_max = nullptr;
   int* d_partial_indices = nullptr;
   CUDA_CHECK(cudaMalloc(&d_partial_max, numBlocks * sizeof(T)));
   CUDA_CHECK(cudaMalloc(&d_partial_indices, numBlocks * sizeof(int)));

   // Launch first level reduction kernel
   // each block puts its max abs value and index in d_partial_max
   // and the corresponding index in d_partial_indices
   findMaxAbsKernel<threadsPerBlock, T, T_COMPUTE><<<numBlocks, threadsPerBlock>>>(
      d_input, d_partial_max, d_partial_indices, n, d_result_val, numBlocks == 1);
   
   // If we have more than one block, perform other levels of reduction
   // in this level we no longer need absolute value so we use a different kernel
   if (numBlocks > 1) {
      // Continue reducing until we have only one block
      while (numBlocks > 1) {
         int secondLevelBlocks = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
         secondLevelBlocks = std::max(1, secondLevelBlocks);

         T* d_second_max = nullptr;
         int* d_second_indices = nullptr;
         
         CUDA_CHECK(cudaMalloc(&d_second_max, secondLevelBlocks * sizeof(T)));
         CUDA_CHECK(cudaMalloc(&d_second_indices, secondLevelBlocks * sizeof(int)));
         
         // Launch second level reduction kernel
         findMaxAbsKernel_level2<threadsPerBlock, T, T_COMPUTE><<<secondLevelBlocks, threadsPerBlock>>>(
            d_input, d_partial_max, d_partial_indices, numBlocks, d_second_max, d_second_indices, d_result_val, secondLevelBlocks == 1);
         
         // Update pointers
         CUDA_CHECK(cudaFree(d_partial_max));
         CUDA_CHECK(cudaFree(d_partial_indices));
         d_partial_max = d_second_max;
         d_partial_indices = d_second_indices;
         numBlocks = secondLevelBlocks;
      }
   }
   
   // Handle based on desired output location
   if (location == Location::kHOST) {
      // Copy result to host
      CUDA_CHECK(cudaMemcpy(result, d_partial_indices, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(result_val, d_result_val, sizeof(T_COMPUTE), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_partial_max));
      CUDA_CHECK(cudaFree(d_partial_indices));
      CUDA_CHECK(cudaFree(d_result_val));
      return *result;
   } else {
      CUDA_CHECK(cudaMemcpy(result, d_partial_indices, sizeof(int), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaFree(d_partial_max));
      CUDA_CHECK(cudaFree(d_partial_indices));
      
      return 0; // When output is on device, return 0 as success indicator
   }
}

// Updated iamax function that supports device output
template<typename T, typename T_O>
MSVDStatus iamax(const Vector<T>& x, int& result, T_O& result_val, Location location) {
   // Check for empty vector
   if (x.length() <= 0) {
      throw std::runtime_error("Empty vector not allowed for iamax operation");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   // Process based on location
   if (x.location() == Location::kHOST) {
      if (location != Location::kHOST) {
         throw std::runtime_error("result must be on host when input vector is on host");
         return MSVDStatus::kErrorInvalidArgument;
      }
      
      // CPU implementation using STL
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
         // For double and float, use std::max_element with custom comparison
         auto abs_compare = [](const T& a, const T& b) {
            return std::abs(a) < std::abs(b);
         };
         
         auto max_iter = std::max_element(x.data(), x.data() + x.length(), abs_compare);
         result = static_cast<int>(max_iter - x.data()) + 1; // Convert to 1-based indexing
         result_val = *max_iter;
      }
      else {
         // Half precision not supported on CPU
         throw std::runtime_error("Half precision not supported for CPU iamax operation");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
   } 
   else {
      // GPU implementation using cuBLAS or custom kernels
      try {
         if constexpr (std::is_same_v<T, double>) {
            if (location == Location::kHOST) {
               cublasHandle_t handle = CUDAHandler::cublas();
               CUBLAS_CHECK(cublasIdamax(handle, x.length(), x.data(), 1, &result));
               result_val = 0.0; // won't be written back in this case
            } else {
               findMaxAbsIndex<double, double>(x.data(), x.length(), &result, &result_val, location);
            }
         } 
         else if constexpr (std::is_same_v<T, float>) {
            if (location == Location::kHOST) {
               cublasHandle_t handle = CUDAHandler::cublas();
               CUBLAS_CHECK(cublasIsamax(handle, x.length(), x.data(), 1, &result));
               result_val = 0.0; // won't be written back in this case
            } else {
               findMaxAbsIndex<float, float>(x.data(), x.length(), &result, &result_val, location);
            }
         } 
         else if constexpr (std::is_same_v<T, __half>) {
            findMaxAbsIndex<__half, T_O>(x.data(), x.length(), &result, &result_val, location);
         }  
         else {
            throw std::runtime_error("Unsupported data type for iamax");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         printf("iamax: error: %s\n", e.what());
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus iamax<double, double>(const Vector<double>& x, int& result, double& result_val, Location location);
template MSVDStatus iamax<float, float>(const Vector<float>& x, int& result, float& result_val, Location location);
template MSVDStatus iamax<__half, __half>(const Vector<__half>& x, int& result, __half& result_val, Location location);
template MSVDStatus iamax<__half, float>(const Vector<__half>& x, int& result, float& result_val, Location location);

// CUDA kernel for finding max abs value indices for each column in double precision
__global__ void matrix_iamax_kernel_double(const double* A, int* results, int rows, int cols, int ld) {
   // Each block handles one column
   int col = blockIdx.x;
   
   if (col >= cols) return;
   
   // Shared memory for partial results
   extern __shared__ double s[];
   double* s_max_vals = s;
   int* s_max_indices = (int*)&s_max_vals[blockDim.x];
   
   // Initialize shared memory
   int tid = threadIdx.x;
   s_max_vals[tid] = 0.0;
   s_max_indices[tid] = -1;
   
   // Each thread processes multiple elements in the column with stride
   for (int row = tid; row < rows; row += blockDim.x) {
      double val = A[col * ld + row];
      double val_abs = abs_val(val);
      
      if (val_abs > s_max_vals[tid]) {
         s_max_vals[tid] = val_abs;
         s_max_indices[tid] = row;
      }
   }
   
   __syncthreads();
   
   // Reduction within the block to find the maximum
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
         if (s_max_vals[tid] < s_max_vals[tid + stride]) {
            s_max_vals[tid] = s_max_vals[tid + stride];
            s_max_indices[tid] = s_max_indices[tid + stride];
         }
      }
      __syncthreads();
   }
   
   // Write the result for this column
   if (tid == 0) {
      results[col] = s_max_indices[0];
   }
}

// CUDA kernel for finding max abs value indices for each column in single precision
__global__ void matrix_iamax_kernel_float(const float* A, int* results, int rows, int cols, int ld) {
   // Each block handles one column
   int col = blockIdx.x;
   
   if (col >= cols) return;
   
   // Shared memory for partial results
   extern __shared__ float s_float[];
   float* s_max_vals = s_float;
   int* s_max_indices = (int*)&s_max_vals[blockDim.x];
   
   // Initialize shared memory
   int tid = threadIdx.x;
   s_max_vals[tid] = 0.0f;
   s_max_indices[tid] = -1;
   
   // Each thread processes multiple elements in the column with stride
   for (int row = tid; row < rows; row += blockDim.x) {
      float val = A[col * ld + row];
      float val_abs = abs_val(val);
      
      if (val_abs > s_max_vals[tid]) {
         s_max_vals[tid] = val_abs;
         s_max_indices[tid] = row;
      }
   }
   
   __syncthreads();
   
   // Reduction within the block to find the maximum
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
         if (s_max_vals[tid] < s_max_vals[tid + stride]) {
            s_max_vals[tid] = s_max_vals[tid + stride];
            s_max_indices[tid] = s_max_indices[tid + stride];
         }
      }
      __syncthreads();
   }
   
   // Write the result for this column
   if (tid == 0) {
      results[col] = s_max_indices[0];
   }
}

// CUDA kernel for finding max abs value indices for each column in half precision
__global__ void matrix_iamax_kernel_half(const __half* A, int* results, int rows, int cols, int ld) {
   // Each block handles one column
   int col = blockIdx.x;
   
   if (col >= cols) return;
   
   // Shared memory for partial results - using float for intermediate values
   extern __shared__ float s_half[];
   float* s_max_vals = s_half;
   int* s_max_indices = (int*)&s_max_vals[blockDim.x];
   
   // Initialize shared memory
   int tid = threadIdx.x;
   s_max_vals[tid] = 0.0f;
   s_max_indices[tid] = -1;
   
   // Each thread processes multiple elements in the column with stride
   for (int row = tid; row < rows; row += blockDim.x) {
      __half val = A[col * ld + row];
      // Use unified interface from type_utils.hpp
      float val_abs = __half2float(abs_val(val));
      
      if (val_abs > s_max_vals[tid]) {
         s_max_vals[tid] = val_abs;
         s_max_indices[tid] = row;
      }
   }
   
   __syncthreads();
   
   // Reduction within the block to find the maximum
   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
         if (s_max_vals[tid] < s_max_vals[tid + stride]) {
            s_max_vals[tid] = s_max_vals[tid + stride];
            s_max_indices[tid] = s_max_indices[tid + stride];
         }
      }
      __syncthreads();
   }
   
   // Write the result for this column
   if (tid == 0) {
      results[col] = s_max_indices[0];
   }
}

template<typename T>
MSVDStatus matrix_iamax(const Matrix<T>& A, int* results, Location location) {
   // Check for empty matrix
   if (A.rows() <= 0 || A.cols() <= 0) {
      throw std::runtime_error("Empty matrix not allowed for matrix_iamax operation");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   // Check if results array is provided
   if (results == nullptr) {
      throw std::runtime_error("Results array cannot be null");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   const int rows = A.rows();
   const int cols = A.cols();
   const int ld = A.ld();  // Leading dimension (stride between columns)
   
   // Process based on location
   if (A.location() == Location::kHOST) {
      // CPU implementation using a loop over columns
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
         // For each column
         for (int col = 0; col < cols; ++col) {
            int max_idx = -1;
            T max_val = static_cast<T>(0);
            
            // Find max abs value in this column
            for (int row = 0; row < rows; ++row) {
               T val = A(row, col);
               T abs_val = std::abs(val);
               
               if (abs_val > max_val) {
                  max_val = abs_val;
                  max_idx = row;
               }
            }
            
            results[col] = max_idx;
         }
      }
      else {
         // Half precision not supported on CPU
         throw std::runtime_error("Half precision not supported for CPU matrix_iamax operation");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
   } 
   else {
      // GPU implementation using custom CUDA kernels
      try {
         // Number of threads per block - adjust based on device capabilities
         int threadsPerBlock = 256;
         
         // Allocate device memory for results if needed
         int* d_results = results;
         bool allocated_d_results = false;
         if (location == Location::kHOST) {
            CUDA_CHECK(cudaMalloc(&d_results, cols * sizeof(int)));
            allocated_d_results = true;
         }
         
         // Calculate shared memory size (values + indices)
         size_t sharedMemSize = threadsPerBlock * sizeof(float) + threadsPerBlock * sizeof(int);
         if constexpr (std::is_same_v<T, double>) {
            sharedMemSize = threadsPerBlock * sizeof(double) + threadsPerBlock * sizeof(int);
         }
         
         // Launch appropriate kernel based on data type
         if constexpr (std::is_same_v<T, double>) {
            matrix_iamax_kernel_double<<<cols, threadsPerBlock, sharedMemSize>>>(
               A.data(), d_results, rows, cols, ld);
         }
         else if constexpr (std::is_same_v<T, float>) {
            matrix_iamax_kernel_float<<<cols, threadsPerBlock, sharedMemSize>>>(
               A.data(), d_results, rows, cols, ld);
         }
         else if constexpr (std::is_same_v<T, __half>) {
            matrix_iamax_kernel_half<<<cols, threadsPerBlock, sharedMemSize>>>(
               A.data(), d_results, rows, cols, ld);
         }
         else {
            if (allocated_d_results) {
               CUDA_CHECK(cudaFree(d_results));
            }
            throw std::runtime_error("Unsupported data type for GPU matrix_iamax");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
         
         // Check for kernel execution errors
         CUDA_CHECK(cudaGetLastError());
         
         // Copy results back to host if needed
         if (allocated_d_results) {
            CUDA_CHECK(cudaMemcpy(results, d_results, cols * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_results));
         }
      } 
      catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus matrix_iamax<double>(const Matrix<double>& A, int* results, Location location);
template MSVDStatus matrix_iamax<float>(const Matrix<float>& A, int* results, Location location);
template MSVDStatus matrix_iamax<__half>(const Matrix<__half>& A, int* results, Location location);

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus axpy(const T_COMPUTE& alpha, const Vector<T_I>& x, Vector<T_O>& y) {
   // 1. Check that both vectors are in the same location (CPU or GPU)
   if (x.location() != y.location()) {
      throw std::runtime_error("Both vectors must be in the same location (CPU or GPU)");
      return MSVDStatus::kErrorBLASInvalidLocation;
   }
   
   // Use the minimum length of the two vectors
   int n = std::min(x.length(), y.length());
   
   // Check for empty vectors
   if (n <= 0) {
      throw std::runtime_error("Empty vectors not allowed for axpy operation");
      return MSVDStatus::kErrorInvalidArgument;
   }

   // 2. Handle based on location
   if (x.location() == Location::kHOST) {
      // CPU implementation using CBLAS
      
      // CPU implementation requires input and output to be the same type
      if constexpr (!std::is_same_v<T_I, T_O>) {
         throw std::runtime_error("For CPU axpy, input and output types must be the same");
         return MSVDStatus::kErrorBLASMixedPrecisionCPU;
      }
      
      // Use CBLAS for CPU implementation
      if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_COMPUTE, double>) {
         // Double precision
         cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
      } 
      else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_COMPUTE, float>) {
         // Single precision
         cblas_saxpy(n, alpha, x.data(), 1, y.data(), 1);
      }
      else {
         // Half precision or mixed precision not supported in CPU
         throw std::runtime_error("Half precision or mixed precision not supported for CPU axpy");
         return MSVDStatus::kErrorBLASUnsupportedType;
      }
   } 
   else {
      // GPU implementation using cuBLAS
      try {
         cublasHandle_t handle = CUDAHandler::cublas();
         
         if constexpr (std::is_same_v<T_I, double> && std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
            // Double precision
            CUBLAS_CHECK(cublasDaxpy(handle, n, &alpha, x.data(), 1, y.data(), 1));
         } 
         else if constexpr (std::is_same_v<T_I, float> && std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
            // Single precision
            CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, x.data(), 1, y.data(), 1));
         } 
         else if constexpr (std::is_same_v<T_I, __half> && std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, float>) {
            // Half precision IO with float compute
            float alpha_float = static_cast<float>(alpha);
            CUBLAS_CHECK(cublasAxpyEx(handle, n,
                                    &alpha_float, CUDA_R_32F,
                                    x.data(), CUDA_R_16F, 1,
                                    y.data(), CUDA_R_16F, 1,
                                    CUDA_R_32F));
         }
         else {
            throw std::runtime_error("Unsupported type combination for axpy");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for common type combinations
template MSVDStatus axpy<double, double, double>(const double& alpha, const Vector<double>& x, Vector<double>& y);
template MSVDStatus axpy<float, float, float>(const float& alpha, const Vector<float>& x, Vector<float>& y);
template MSVDStatus axpy<__half, __half, float>(const float& alpha, const Vector<__half>& x, Vector<__half>& y);

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus scale(const T_COMPUTE& alpha, const Vector<T_I>& x, Vector<T_O>& y) {
   // 1. Check that both vectors are in the same location (CPU or GPU)
   if (x.location() != y.location()) {
      throw std::runtime_error("Both vectors must be in the same location (CPU or GPU)");
      return MSVDStatus::kErrorBLASInvalidLocation;
   }
   
   // Use the minimum length of the two vectors
   int n = std::min(x.length(), y.length());
   
   // Check for empty vectors
   if (n <= 0) {
      throw std::runtime_error("Empty vectors not allowed for scale operation");
      return MSVDStatus::kErrorInvalidArgument;
   }

   // 2. Handle based on location
   if (x.location() == Location::kHOST) {
      // CPU implementation using CBLAS
      
      // CPU implementation has restrictions on input and output types
      if constexpr (!std::is_same_v<T_I, T_O>) {
         throw std::runtime_error("For CPU scale, input and output types must be the same");
         return MSVDStatus::kErrorBLASMixedPrecisionCPU;
      }
      
      // For CPU, we need to copy x to y and then scale y
      // First copy x to y
      for (size_t i = 0; i < n; ++i) {
         y.data()[i] = static_cast<T_O>(x.data()[i]);
      }
      
      // Then scale y using CBLAS
      if constexpr (std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
         // Double precision
         cblas_dscal(n, alpha, y.data(), 1);
      } 
      else if constexpr (std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
         // Single precision
         cblas_sscal(n, alpha, y.data(), 1);
      }
      else {
         // Half precision or mixed precision not supported in CPU
         throw std::runtime_error("Half precision or mixed precision not supported for CPU scale");
         return MSVDStatus::kErrorBLASUnsupportedType;
      }
   } 
   else {
      // GPU implementation using cuBLAS
      try {
         cublasHandle_t handle = CUDAHandler::cublas();
         
         // First copy x to y
         if constexpr (std::is_same_v<T_I, T_O>) {
            // If types match, use cudaMemcpy
            CUDA_CHECK(cudaMemcpy(y.data(), x.data(), n * sizeof(T_I), cudaMemcpyDeviceToDevice));
         } 
         else {
            // For type conversion, we need a custom kernel
            // This is a simplified approach - might need a proper type conversion kernel
            throw std::runtime_error("Type conversion between input and output not supported for scale operation");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
         
         // Then scale y
         if constexpr (std::is_same_v<T_O, double> && std::is_same_v<T_COMPUTE, double>) {
            // Double precision
            CUBLAS_CHECK(cublasDscal(handle, n, &alpha, y.data(), 1));
         } 
         else if constexpr (std::is_same_v<T_O, float> && std::is_same_v<T_COMPUTE, float>) {
            // Single precision
            CUBLAS_CHECK(cublasSscal(handle, n, &alpha, y.data(), 1));
         } 
         else if constexpr (std::is_same_v<T_O, __half> && std::is_same_v<T_COMPUTE, float>) {
            // Half precision IO with float compute
            // Make sure alpha is converted to float for computation
            float alpha_float = static_cast<float>(alpha);
            CUBLAS_CHECK(cublasScalEx(handle, n,
                                    &alpha_float, CUDA_R_32F,
                                    y.data(), CUDA_R_16F, 1,
                                    CUDA_R_32F));
         }
         else {
            throw std::runtime_error("Unsupported type combination for scale");
            return MSVDStatus::kErrorBLASUnsupportedType;
         }
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorBLASCuBLASFailure;
      }
   }
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for common type combinations
template MSVDStatus scale<double, double, double>(const double& alpha, const Vector<double>& x, Vector<double>& y);
template MSVDStatus scale<float, float, float>(const float& alpha, const Vector<float>& x, Vector<float>& y);
template MSVDStatus scale<__half, __half, float>(const float& alpha, const Vector<__half>& x, Vector<__half>& y);

// CUDA kernel for checking NaN and Inf values
template <typename T>
__global__ void check_special_values_kernel(const T* data, int n, int* has_nan, int* has_inf) {
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   
   for (size_t i = idx; i < n; i += stride) {
      T val = data[i];
      
      // Use unified interfaces from type_utils.hpp
      if (isnan_val(val)) {
         atomicExch(has_nan, 1);
      } 
      else if (isinf_val(val)) {
         atomicExch(has_inf, 1);
      }
   }
}

template<typename T>
MSVDStatus check_special_values(const Vector<T>& x) {
   // Check for empty vector
   if (x.length() <= 0) {
      return MSVDStatus::kSuccess; // Empty vector has no special values
   }
   
   // If vector is on the host
   if (x.location() == Location::kHOST) {
      bool has_nan = false;
      bool has_inf = false;
      
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
         // Check each element for NaN or Inf
         for (size_t i = 0; i < x.length(); ++i) {
            if (std::isnan(x[i])) {
               has_nan = true;
               break;
            } 
            else if (std::isinf(x[i])) {
               has_inf = true;
               break;
            }
         }
      }
      else if constexpr (std::is_same_v<T, __half>) {
         throw std::runtime_error("Half precision not supported for CPU check_special_values");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
      
      // Return appropriate status
      if (has_nan) {
         return MSVDStatus::kErrorNaN;
      } 
      else if (has_inf) {
         return MSVDStatus::kErrorInf;
      }
      
      return MSVDStatus::kSuccess;
   }
   // If vector is on the device
   else {
      try {
         // Allocate device memory for flags
         int* d_has_nan;
         int* d_has_inf;
         CUDA_CHECK(cudaMalloc(&d_has_nan, sizeof(int)));
         CUDA_CHECK(cudaMalloc(&d_has_inf, sizeof(int)));
         
         // Initialize flags to 0
         CUDA_CHECK(cudaMemset(d_has_nan, 0, sizeof(int)));
         CUDA_CHECK(cudaMemset(d_has_inf, 0, sizeof(int)));
         
         // Calculate grid and block dimensions
         int blockSize = 256;
         int numBlocks = (x.length() + blockSize - 1) / blockSize;
         numBlocks = std::min(numBlocks, 1024); // Limit number of blocks
         
         // Launch kernel to check for NaN and Inf
         check_special_values_kernel<T><<<numBlocks, blockSize>>>(x.data(), x.length(), d_has_nan, d_has_inf);
         
         // Check for kernel launch errors
         CUDA_CHECK(cudaGetLastError());
         CUDA_CHECK(cudaDeviceSynchronize());
         
         // Copy results back to host
         int h_has_nan = 0;
         int h_has_inf = 0;
         CUDA_CHECK(cudaMemcpy(&h_has_nan, d_has_nan, sizeof(int), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaMemcpy(&h_has_inf, d_has_inf, sizeof(int), cudaMemcpyDeviceToHost));
         
         // Clean up device memory
         CUDA_CHECK(cudaFree(d_has_nan));
         CUDA_CHECK(cudaFree(d_has_inf));
         
         // Return appropriate status
         if (h_has_nan) {
            return MSVDStatus::kErrorNaN;
         } 
         else if (h_has_inf) {
            return MSVDStatus::kErrorInf;
         }
         
         return MSVDStatus::kSuccess;
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorCUDAGeneral;
      }
   }
}

template<typename T>
MSVDStatus check_special_values(const Matrix<T>& A) {
   // Check for empty matrix
   if (A.rows() <= 0 || A.cols() <= 0) {
      return MSVDStatus::kSuccess; // Empty matrix has no special values
   }
   
   const int total_elements = A.rows() * A.cols();
   
   // If matrix is on the host
   if (A.location() == Location::kHOST) {
      bool has_nan = false;
      bool has_inf = false;
      
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
         // Check each element for NaN or Inf
         for (size_t j = 0; j < A.cols(); ++j) {
            for (size_t i = 0; i < A.rows(); ++i) {
               const T& val = A(i, j);
               if (std::isnan(val)) {
                  has_nan = true;
                  break;
               } 
               else if (std::isinf(val)) {
                  has_inf = true;
                  break;
               }
            }
            if (has_nan || has_inf) break;
         }
      }
      else if constexpr (std::is_same_v<T, __half>) {
         throw std::runtime_error("Half precision not supported for CPU check_special_values");
         return MSVDStatus::kErrorBLASHalfPrecisionCPU;
      }
      
      // Return appropriate status
      if (has_nan) {
         return MSVDStatus::kErrorNaN;
      } 
      else if (has_inf) {
         return MSVDStatus::kErrorInf;
      }
      
      return MSVDStatus::kSuccess;
   }
   // If matrix is on the device
   else {
      try {
         // Allocate device memory for flags
         int* d_has_nan;
         int* d_has_inf;
         CUDA_CHECK(cudaMalloc(&d_has_nan, sizeof(int)));
         CUDA_CHECK(cudaMalloc(&d_has_inf, sizeof(int)));
         
         // Initialize flags to 0
         CUDA_CHECK(cudaMemset(d_has_nan, 0, sizeof(int)));
         CUDA_CHECK(cudaMemset(d_has_inf, 0, sizeof(int)));
         
         // Calculate grid and block dimensions
         int blockSize = 256;
         int numBlocks = (total_elements + blockSize - 1) / blockSize;
         numBlocks = std::min(numBlocks, 1024); // Limit number of blocks
         
         // Launch kernel to check for NaN and Inf
         // Use the kernel defined for Vector, as the check is the same
         check_special_values_kernel<T><<<numBlocks, blockSize>>>(A.data(), total_elements, d_has_nan, d_has_inf);
         
         // Check for kernel launch errors
         CUDA_CHECK(cudaGetLastError());
         CUDA_CHECK(cudaDeviceSynchronize());
         
         // Copy results back to host
         int h_has_nan = 0;
         int h_has_inf = 0;
         CUDA_CHECK(cudaMemcpy(&h_has_nan, d_has_nan, sizeof(int), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaMemcpy(&h_has_inf, d_has_inf, sizeof(int), cudaMemcpyDeviceToHost));
         
         // Clean up device memory
         CUDA_CHECK(cudaFree(d_has_nan));
         CUDA_CHECK(cudaFree(d_has_inf));
         
         // Return appropriate status
         if (h_has_nan) {
            return MSVDStatus::kErrorNaN;
         } 
         else if (h_has_inf) {
            return MSVDStatus::kErrorInf;
         }
         
         return MSVDStatus::kSuccess;
      } catch (const std::runtime_error& e) {
         return MSVDStatus::kErrorCUDAGeneral;
      }
   }
}

// Explicit instantiation for supported types
template MSVDStatus check_special_values<double>(const Vector<double>& x);
template MSVDStatus check_special_values<float>(const Vector<float>& x);
template MSVDStatus check_special_values<__half>(const Vector<__half>& x);
template MSVDStatus check_special_values<double>(const Matrix<double>& A);
template MSVDStatus check_special_values<float>(const Matrix<float>& A);
template MSVDStatus check_special_values<__half>(const Matrix<__half>& A);

template<typename T>
MSVDStatus syev( Matrix<T>& A, Vector<T>& W, T* work, int lwork)
{
   size_t n = A.rows();
   if(n != A.cols()) {
      throw std::runtime_error("A must be a square matrix");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   // Check, all matrices must be on host
   if(A.location() != Location::kHOST || W.location() != Location::kHOST) {
      throw std::runtime_error("All matrices/vectors must be on host");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   if constexpr (std::is_same_v<T, double>) {
      int info = LAPACKE_dsyev_work(LAPACK_COL_MAJOR, 'V', 'U', n, 
                                       A.data(), A.ld(), W.data(),
                                       work, lwork);
   }
   else if constexpr (std::is_same_v<T, float>) {
      int info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, 'V', 'U', n, 
                                       A.data(), A.ld(), W.data(),
                                       work, lwork);
   }

   return MSVDStatus::kSuccess;

}

template MSVDStatus syev<double>( Matrix<double>& A, Vector<double>& W, double* work, int lwork);
template MSVDStatus syev<float>( Matrix<float>& A, Vector<float>& W, float* work, int lwork);

template<typename T>
MSVDStatus sygv( Matrix<T>& A, Matrix<T>& B, Vector<T>& W, T* work, int lwork)
{
   size_t n = A.rows();
   if(n != A.cols() || n != B.rows() || n != B.cols()) {
      throw std::runtime_error("A and B must be square matrices of same size");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   // Check, all matrices must be on host
   if(A.location() != Location::kHOST || B.location() != Location::kHOST || W.location() != Location::kHOST) {
      throw std::runtime_error("All matrices/vectors must be on host");
      return MSVDStatus::kErrorInvalidArgument;
   }
   
   if constexpr (std::is_same_v<T, double>) {
      int info = LAPACKE_dsygv_work(LAPACK_COL_MAJOR, 1, 'V', 'U', n, 
                                       A.data(), A.ld(), B.data(), B.ld(), W.data(),
                                       work, lwork);
   }
   else if constexpr (std::is_same_v<T, float>) {
      int info = LAPACKE_ssygv_work(LAPACK_COL_MAJOR, 1, 'V', 'U', n, 
                                       A.data(), A.ld(), B.data(), B.ld(), W.data(),
                                       work, lwork);
   }

   return MSVDStatus::kSuccess;
}

template MSVDStatus sygv<double>( Matrix<double>& A, Matrix<double>& B, Vector<double>& W, double* work, int lwork);
template MSVDStatus sygv<float>( Matrix<float>& A, Matrix<float>& B, Vector<float>& W, float* work, int lwork);

} // namespace msvd