#include "hessenberg.hpp"
#include "../../linalg/blas/mvops.hpp"
#include "../../core/utils/cuda_handler.hpp"
#include "../../core/utils/type_utils.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace msvd {

constexpr int SCALE_KERNEL_BLOCK_SIZE = 1024;

constexpr int CONSTANT_BLOCI_SIZE_DOUBLE = 2048;
constexpr int CONSTANT_BLOCI_SIZE_FLOAT = 4096;
constexpr int CONSTANT_BLOCI_SIZE_HALF = 8192;

__constant__ double perm_val_double;
__constant__ float perm_val_float;
__constant__ __half perm_val_half;

__constant__ int perm_idx;

__constant__ double scale_double[CONSTANT_BLOCI_SIZE_DOUBLE];
__constant__ float scale_float[CONSTANT_BLOCI_SIZE_FLOAT];
__constant__ __half scale_half[CONSTANT_BLOCI_SIZE_HALF];

/**
 * @brief This kernel scales the current column
 * @details This kernel scales the current column. \n
 *          We need to use current column to update all columns after it.
 */
template<typename T_O, typename T_COMPUTE>
__global__ void hessenberg_colscale_kernel(T_O* q_i, int m, T_COMPUTE select_tol, int *skip) {
   int tid = threadIdx.x;
   int row = blockIdx.x * blockDim.x + tid;

   if (row >= m) {
      return;
   }

   if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         if(__habs(perm_val_half) < select_tol) {
            if(row == 0) {
               skip[0] = 1;
            }
            return;
         }
         else {
            q_i[row] = __hdiv(q_i[row], perm_val_half);
         }
      }
      else {
         if(abs(perm_val_float) < select_tol) {
            if(row == 0) {
               skip[0] = 1;
            }
            return;
         }
         else {
            q_i[row] = __float2half(__half2float(q_i[row]) / perm_val_float);
         }
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      if(abs(perm_val_float) < select_tol) {
         if(row == 0) {
            skip[0] = 1;
         }
         return;
      }
      else {
         q_i[row] = q_i[row] / perm_val_float;
      }
   } else if constexpr (std::is_same_v<T_O, double>) {
      if(abs(perm_val_double) < select_tol) {
         if(row == 0) {
            skip[0] = 1;
         }
         return;
      }
      else {
         q_i[row] = q_i[row] / perm_val_double;
      }
   }
}

/**
 * @brief This kernel computes the scale factor for the current column
 * @details This kernel computes the scale factor for the current column. \n
 *          We need to use current column to update all columns after it.
 */
template<typename T_O, typename T_COMPUTE>
__global__ void hessenberg_compute_scale_kernel(T_O* Q, int n, int ldq, T_COMPUTE* scale_factor) {
   int tid = threadIdx.x;
   int col = blockIdx.x * blockDim.x + tid;

   if(col >= n) {
      return;
   }

   if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         scale_factor[col] = (Q[col * ldq + perm_idx - 1]);
      } else {
         scale_factor[col] = __half2float(Q[col * ldq + perm_idx - 1]);
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      scale_factor[col] = Q[col * ldq + perm_idx - 1];
   } else if constexpr (std::is_same_v<T_O, double>) {
      scale_factor[col] = Q[col * ldq + perm_idx - 1];
   }

   return;
}

/**
 * @brief This kernel scales the columns starting from the current Q pointer
 * @details This kernel scales the columns starting from the current Q pointer. \n
 *          We should already have scale factor in the __constant__ memory. \n
 *          We have a 2D block with 1D grid.
 */
template<typename T_O, typename T_COMPUTE>
__global__ void hessenberg_scale_kernel(
   T_O* Q, int m, int ldq
) {
   int tid = threadIdx.x;
   int row = blockIdx.x * blockDim.x + tid;
   int col = blockIdx.y;

   if (row >= m) {
      return;
   }

   if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         Q[col * ldq + row] = __hsub(Q[col * ldq + row], __hmul(Q[ - ldq + row], scale_half[col]));
      } else {
         Q[col * ldq + row] -= __float2half(__half2float(Q[ - ldq + row]) * scale_float[col]);
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      Q[col * ldq + row] -= Q[ - ldq + row] * scale_float[col];
   } else if constexpr (std::is_same_v<T_O, double>) {
      Q[col * ldq + row] -= Q[ - ldq + row] * scale_double[col];
   }
}

/**
 * @brief This kernel computes the scale factor for the current column
 * @details This kernel computes the scale factor for only one column.
 */
template<typename T_O, typename T_COMPUTE>
__global__ void hessenberg_compute_scale_kernel_v3(T_O* Q, int* perm_idx, T_COMPUTE* scale_factor) {
   if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         scale_factor[0] = (Q[perm_idx[0] - 1]);
      } else {
         scale_factor[0] = __half2float(Q[perm_idx[0] - 1]);
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      scale_factor[0] = Q[perm_idx[0] - 1];
   } else if constexpr (std::is_same_v<T_O, double>) {
      scale_factor[0] = Q[perm_idx[0] - 1];
   }
   return;
}

/**
 * @brief This kernel computes the scale factor for the current column
 * @details This kernel computes the scale factor for the current column. \n
 *          We simply extract the values from previous permutation index in the current column.
 */
template<typename T_O, typename T_COMPUTE>
__global__ void hessenberg_scale_kernel_v3(T_O* q_j, T_O* q_i, int n) {
   int tid = threadIdx.x;
   int row = blockIdx.x * blockDim.x + tid;

   if(row >= n) {
      return;
   }
   if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         q_i[row] = __hsub(q_i[row], __hmul(q_j[row], perm_val_half));
      } else {
         q_i[row] = __float2half(__half2float(q_i[row]) - __half2float(q_j[row]) * perm_val_float);
      }
   }
   else if constexpr (std::is_same_v<T_O, float>) {
      q_i[row] = q_i[row] - q_j[row] * perm_val_float;
   }
   else if constexpr (std::is_same_v<T_O, double>) {
      q_i[row] = q_i[row] - q_j[row] * perm_val_double;
   }
   return;
}

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> hessenberg_v2(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol,
   T_O orth_tol,
   T_O reorth_tol
) {
   // Get dimensions
   size_t m = A.rows();
   size_t n = A.cols();
   
   // Check input/output locations
   if (A.location() != Location::kDEVICE) {
      throw std::runtime_error("Input matrix A must be on device");
   }
   if (Q.location() != Location::kDEVICE) {
      throw std::runtime_error("Output matrix Q must be on device");
   }
   if (R.location() != Location::kHOST) {
      throw std::runtime_error("Output matrix R must be on host");
   }
   
   // Check dimensions
   if (Q.rows() != m || Q.cols() != n) {
      throw std::runtime_error("Output matrix Q has incorrect dimensions");
   }
   if (R.rows() != n || R.cols() != n) {
      throw std::runtime_error("Output matrix R has incorrect dimensions");
   }
   
   // Copy A to Q for in-place modifications
   if constexpr (std::is_same_v<T_I, T_O>) {
      if (A.ld() == Q.ld()) {
         CUDA_CHECK(cudaMemcpy(Q.data(), A.data(), A.ld() * n * sizeof(T_I), cudaMemcpyDeviceToDevice));
      }
      else {
         for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpy(&Q(0, i), &A(0, i), m * sizeof(T_I), cudaMemcpyDeviceToDevice));
         }
      }
   } 
   else {
      Vector<T_I> A_data(A.data(), A.ld() * n, Location::kDEVICE);
      Vector<T_O> Q_data = A_data.template cast<T_O>();
      if (Q.ld() == A.ld()) {
         CUDA_CHECK(cudaMemcpy(Q.data(), Q_data.data(), A.ld() * n * sizeof(T_O), cudaMemcpyDeviceToDevice));
      }
      else {
         for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpy(&Q(0, i), &Q_data(0, i), m * sizeof(T_O), cudaMemcpyDeviceToDevice));
         }
      }
   }

   // Initialize R to zeros
   R.fill(get_zero<T_O>());
   
   // Vector to track skipped columns
   std::vector<int> skip(n);
   int* skip_d;
   CUDA_CHECK(cudaMalloc(&skip_d, n * sizeof(int)));
   CUDA_CHECK(cudaMemset(skip_d, 0, n * sizeof(int)));
   // We need a memory on host pinned memory to fast check if we should skip the current column
   int* skip_h;
   CUDA_CHECK(cudaMallocHost(&skip_h, sizeof(int)));
   
   // Vector to track permutation of columns (which rows have max elements)
   // Note that we use 1-based indexing, and stored on device
   int* constant_perm_idx_pointer = nullptr;
   CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_idx_pointer, perm_idx));
   T_COMPUTE* constant_perm_val_pointer = nullptr;
   T_COMPUTE* scale_pointer = nullptr;
   if constexpr (std::is_same_v<T_COMPUTE, double>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_double));
      CUDA_CHECK(cudaGetSymbolAddress((void**)&scale_pointer, scale_double));
   } else if constexpr (std::is_same_v<T_COMPUTE, float>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_float));
      CUDA_CHECK(cudaGetSymbolAddress((void**)&scale_pointer, scale_float));
   } else if constexpr (std::is_same_v<T_COMPUTE, __half>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_half));
      CUDA_CHECK(cudaGetSymbolAddress((void**)&scale_pointer, scale_half));
   }

   int* result;
   CUDA_CHECK(cudaMalloc(&result, sizeof(int)));

   // The scalefactor array
   Matrix<T_COMPUTE> scale_factor_matrix(n, n, Location::kDEVICE);
   scale_factor_matrix.fill(get_zero<T_COMPUTE>());
   // Process each column, we use a different implementation
   for (size_t i = 0; i < n; i++) {
      // Create a reference vector to column i
      Vector<T_O> q_i(&Q(0, i), m, Location::kDEVICE);
   
      // Get max index directly to device memory index
      MSVDStatus status = iamax<T_O, T_COMPUTE>(q_i, result[0], scale_factor_matrix(i, i), Location::kDEVICE);
      if (status != MSVDStatus::kSuccess) {
         throw std::runtime_error("iamax failed for column " + std::to_string(i));
      }

      // Now we have the scaling factor for the current column
      // Following steps:
      // 1. Scale the current column so that the largest element is 1
      // 2. Compute the scale factor for all columns after it (simply the values on the permutation index)
      // 3. Scale the columns after it

      // Copy based on block size
      CUDA_CHECK(cudaMemcpy(constant_perm_idx_pointer, result, sizeof(int), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(constant_perm_val_pointer, &scale_factor_matrix(i, i), sizeof(T_COMPUTE), cudaMemcpyDeviceToDevice));

      // Step 1: Scale the current column
      hessenberg_colscale_kernel<T_O, T_COMPUTE><<<(m + SCALE_KERNEL_BLOCK_SIZE - 1) / SCALE_KERNEL_BLOCK_SIZE, SCALE_KERNEL_BLOCK_SIZE>>>(
         &Q(0, i), m, select_tol, skip_d + i
      );
      // Check if we can skip the scaleing step
      CUDA_CHECK(cudaMemcpy(skip_h, skip_d + i, sizeof(int), cudaMemcpyDeviceToHost));
      if (skip_h[0] == 1 || i == n - 1) {
         continue;
      }
      
      // Step 2: Compute the scale factor for all columns after it
      // Now the first column has its scale factor computed, 
      hessenberg_compute_scale_kernel<T_O, T_COMPUTE><<<(n - i + SCALE_KERNEL_BLOCK_SIZE - 2) / SCALE_KERNEL_BLOCK_SIZE, SCALE_KERNEL_BLOCK_SIZE>>>(
         &Q(0, i+1), n - i - 1, Q.ld(), &scale_factor_matrix(i+1, i)
      );

      // Now compute the scale for all columns after it
      int scale_idx = (int)i + 1;
      size_t kernel_size;
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         kernel_size = CONSTANT_BLOCI_SIZE_HALF;
      } else if constexpr (std::is_same_v<T_COMPUTE, float>) {
         kernel_size = CONSTANT_BLOCI_SIZE_FLOAT;
      } else if constexpr (std::is_same_v<T_COMPUTE, double>) {
         kernel_size = CONSTANT_BLOCI_SIZE_DOUBLE;
      }

      constexpr int block_size = 1024;
      // we do this block by block as the size of constant memory is limited
      while(scale_idx < n) {
         int ncols = std::min(n - scale_idx, kernel_size);
         dim3 grid_size((m + block_size - 1) / block_size, ncols);
         CUDA_CHECK(cudaMemcpy(scale_pointer, &scale_factor_matrix(scale_idx, i), ncols * sizeof(T_COMPUTE), cudaMemcpyDeviceToDevice));
         hessenberg_scale_kernel<T_O, T_COMPUTE><<<grid_size, block_size>>>(
            &Q(0, scale_idx), m, Q.ld()
         );
         scale_idx += kernel_size;
      }

   }

   // Copy skip from device to host
   CUDA_CHECK(cudaMemcpy(skip.data(), skip_d, n * sizeof(int), cudaMemcpyDeviceToHost));

   // Also copy the scale factor matrix to host
   scale_factor_matrix.to_host();

   // Note that the two matrices R and scale_factor_matrix might have different datatypes
   // and also one of them is a transposed matrix
   // For now use double for loop to copy the elements, definetly have better ways to do this
   if constexpr (std::is_same_v<T_O, double>) {
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i; j < n; j++) {
            R(i, j) = scale_factor_matrix(j, i);
         }
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i; j < n; j++) {
            R(i, j) = scale_factor_matrix(j, i);
         }
      }
   } else if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < n; j++) {
               R(i, j) = scale_factor_matrix(j, i);
            }
         }
      } else {
         for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < n; j++) {
               R(i, j) = __float2half( scale_factor_matrix(j, i));
            }
         }
      }
   }

   CUDA_CHECK(cudaFree(skip_d));
   CUDA_CHECK(cudaFreeHost(skip_h));
   CUDA_CHECK(cudaFree(result));

   return skip;
}

// Explicit template instantiations
template std::vector<int> hessenberg_v2<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&, double, double, double);

template std::vector<int> hessenberg_v2<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&, float, float, float);

template std::vector<int> hessenberg_v2<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&, __half, __half, __half);

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> hessenberg(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol,
   T_O orth_tol,
   T_O reorth_tol
) {
   // Get dimensions
   size_t m = A.rows();
   size_t n = A.cols();
   
   // Check input/output locations
   if (A.location() != Location::kDEVICE) {
      throw std::runtime_error("Input matrix A must be on device");
   }
   if (Q.location() != Location::kDEVICE) {
      throw std::runtime_error("Output matrix Q must be on device");
   }
   if (R.location() != Location::kHOST) {
      throw std::runtime_error("Output matrix R must be on host");
   }
   
   // Check dimensions
   if (Q.rows() != m || Q.cols() != n) {
      throw std::runtime_error("Output matrix Q has incorrect dimensions");
   }
   if (R.rows() != n || R.cols() != n) {
      throw std::runtime_error("Output matrix R has incorrect dimensions");
   }
   
   // Copy A to Q for in-place modifications
   if constexpr (std::is_same_v<T_I, T_O>) {
      if (A.ld() == Q.ld()) {
         CUDA_CHECK(cudaMemcpy(Q.data(), A.data(), A.ld() * n * sizeof(T_I), cudaMemcpyDeviceToDevice));
      }
      else {
         for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpy(&Q(0, i), &A(0, i), m * sizeof(T_I), cudaMemcpyDeviceToDevice));
         }
      }
   }
   else {
      Vector<T_I> A_data(A.data(), A.ld() * n, Location::kDEVICE);
      Vector<T_O> Q_data = A_data.template cast<T_O>();
      if (Q.ld() == A.ld()) {
         CUDA_CHECK(cudaMemcpy(Q.data(), Q_data.data(), A.ld() * n * sizeof(T_O), cudaMemcpyDeviceToDevice));
      }
      else {
         for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpy(&Q(0, i), &Q_data(0, i), m * sizeof(T_O), cudaMemcpyDeviceToDevice));
         }
      }
   }
   
   // Initialize R to zeros
   R.fill(get_zero<T_O>());
   
   // Vector to track skipped columns
   std::vector<int> skip(n);
   int* skip_d;
   CUDA_CHECK(cudaMalloc(&skip_d, n * sizeof(int)));
   CUDA_CHECK(cudaMemset(skip_d, 0, n * sizeof(int)));
   // We need a memory on host pinned memory to fast check if we should skip the current column
   int* skip_h;
   CUDA_CHECK(cudaMallocHost(&skip_h, sizeof(int)));
   
   // Vector to track permutation of columns (which rows have max elements)
   // Note that we use 1-based indexing, and stored on device
   int* constant_perm_idx_pointer = nullptr;
   CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_idx_pointer, perm_idx));
   T_O* constant_perm_val_pointer = nullptr;
   if constexpr (std::is_same_v<T_COMPUTE, double>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_double));
   } else if constexpr (std::is_same_v<T_COMPUTE, float>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_float));
   } else if constexpr (std::is_same_v<T_COMPUTE, __half>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_half));
   }

   int* result;
   CUDA_CHECK(cudaMalloc(&result, sizeof(int)));

   // The scalefactor array
   Matrix<T_O> scale_factor_matrix(n, n, Location::kDEVICE);
   scale_factor_matrix.fill(get_zero<T_O>());
   Vector<T_COMPUTE> scale_factor_diag(n, Location::kDEVICE);
   scale_factor_diag.fill(get_zero<T_COMPUTE>());
   // Process each column, we use a different implementation
   for (size_t i = 0; i < n; i++) {
      // Create a reference vector to column i
      Vector<T_O> q_i(&Q(0, i), m, Location::kDEVICE);
   
      // Get max index directly to device memory index
      MSVDStatus status = iamax<T_O, T_COMPUTE>(q_i, result[0], scale_factor_diag[i], Location::kDEVICE);
      if (status != MSVDStatus::kSuccess) {
         throw std::runtime_error("iamax failed for column " + std::to_string(i));
      }

      // Now we have the scaling factor for the current column
      // Following steps:
      // 1. Scale the current column so that the largest element is 1
      // 2. Compute the scale factor for all columns after it (simply the values on the permutation index)
      // 3. Scale the columns after it

      // Copy based on block size
      CUDA_CHECK(cudaMemcpy(constant_perm_idx_pointer, result, sizeof(int), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(constant_perm_val_pointer, &scale_factor_diag[i], sizeof(T_COMPUTE), cudaMemcpyDeviceToDevice));

      // Step 1: Scale the current column
      hessenberg_colscale_kernel<T_O, T_COMPUTE><<<(m + SCALE_KERNEL_BLOCK_SIZE - 1) / SCALE_KERNEL_BLOCK_SIZE, SCALE_KERNEL_BLOCK_SIZE>>>(
         &Q(0, i), m, select_tol, skip_d + i
      );
      // Check if we can skip the scaleing step
      CUDA_CHECK(cudaMemcpy(skip_h, skip_d + i, sizeof(int), cudaMemcpyDeviceToHost));
      if (skip_h[0] == 1 || i == n - 1) {
         continue;
      }
      
      // Step 2: Compute the scale factor for all columns after it
      // Now the first column has its scale factor computed,
      // WARNING: currently we only need to take function value, so T_O computation is fine
      // however, might have trouble if we use something more advanced calcluation
      hessenberg_compute_scale_kernel<T_O, T_O><<<(n - i + SCALE_KERNEL_BLOCK_SIZE - 2) / SCALE_KERNEL_BLOCK_SIZE, SCALE_KERNEL_BLOCK_SIZE>>>(
         &Q(0, i+1), n - i - 1, Q.ld(), &scale_factor_matrix(i+1, i)
      );

      // Now compute the scale for all columns after it
      // We directly use GEMM to do this
      Matrix<T_O> col_i_matrix(&Q(0, i), m, 1, Q.ld(), Location::kDEVICE);
      Matrix<T_O> scale_matrix(&scale_factor_matrix(i+1, i), 1, n - i + 1, 1, Location::kDEVICE);
      Matrix<T_O> result_matrix(&Q(0, i+1), m, n - i + 1, Q.ld(), Location::kDEVICE);
      if constexpr (std::is_same_v<T_COMPUTE, double>) {
         double alpha = -1.0;
         double beta = 1.0;
         gemm<T_O, T_O, T_COMPUTE>(false, false,
                  alpha, col_i_matrix, scale_matrix,
                  beta, result_matrix);
      } else if constexpr (std::is_same_v<T_COMPUTE, float>) {
         float alpha = -1.0f;
         float beta = 1.0f;
         gemm<T_O, T_O, T_COMPUTE>(false, false,
                  alpha, col_i_matrix, scale_matrix,
                  beta, result_matrix);
      } else 
      {
         // we don't support __half precision compute
         throw std::runtime_error("Unsupported compute type");
      }

   }

   // Copy skip from device to host
   CUDA_CHECK(cudaMemcpy(skip.data(), skip_d, n * sizeof(int), cudaMemcpyDeviceToHost));

   // Also copy the scale factor matrix to host
   scale_factor_matrix.to_host();

   // Note that the two matrices R and scale_factor_matrix might have different datatypes
   // and also one of them is a transposed matrix
   // For now use double for loop to copy the elements, definetly have better ways to do this
   scale_factor_diag.to_host();
   if constexpr (std::is_same_v<T_O, double>) {
      for (size_t i = 0; i < n; i++) {
         R(i, i) = scale_factor_diag[i];
         for (size_t j = i+1; j < n; j++) {
            R(i, j) = scale_factor_matrix(j, i);
         }
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      for (size_t i = 0; i < n; i++) {
         R(i, i) = scale_factor_diag[i];
         for (size_t j = i+1; j < n; j++) {
            R(i, j) = scale_factor_matrix(j, i);
         }
      }
   } else if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         for (size_t i = 0; i < n; i++) {
            R(i, i) = scale_factor_diag[i];
            for (size_t j = i+1; j < n; j++) {
               R(i, j) = scale_factor_matrix(j, i);
            }
         }
      } else {
         for (size_t i = 0; i < n; i++) {
            R(i, i) = __float2half(scale_factor_diag[i]);
            for (size_t j = i+1; j < n; j++) {
               R(i, j) = scale_factor_matrix(j, i);
            }
         }
      }
   }

   CUDA_CHECK(cudaFree(skip_d));
   CUDA_CHECK(cudaFreeHost(skip_h));
   CUDA_CHECK(cudaFree(result));

   return skip;
}

// Explicit template instantiations
template std::vector<int> hessenberg<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&, double, double, double);

template std::vector<int> hessenberg<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&, float, float, float);

template std::vector<int> hessenberg<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&, __half, __half, __half);

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> hessenberg_v3(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q,
   Matrix<T_O>& R,
   T_O select_tol,
   T_O orth_tol,
   T_O reorth_tol
) {
   // Get dimensions
   size_t m = A.rows();
   size_t n = A.cols();
   
   // Check input/output locations
   if (A.location() != Location::kDEVICE) {
      throw std::runtime_error("Input matrix A must be on device");
   }
   if (Q.location() != Location::kDEVICE) {
      throw std::runtime_error("Output matrix Q must be on device");
   }
   if (R.location() != Location::kHOST) {
      throw std::runtime_error("Output matrix R must be on host");
   }
   
   // Check dimensions
   if (Q.rows() != m || Q.cols() != n) {
      throw std::runtime_error("Output matrix Q has incorrect dimensions");
   }
   if (R.rows() != n || R.cols() != n) {
      throw std::runtime_error("Output matrix R has incorrect dimensions");
   }
   
   // Copy A to Q for in-place modifications
   if constexpr (std::is_same_v<T_I, T_O>) {
      if (A.ld() == Q.ld()) {
         CUDA_CHECK(cudaMemcpy(Q.data(), A.data(), A.ld() * n * sizeof(T_I), cudaMemcpyDeviceToDevice));
      }
      else {
         for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpy(&Q(0, i), &A(0, i), m * sizeof(T_I), cudaMemcpyDeviceToDevice));
         }
      }
   }
   else {
      Vector<T_I> A_data(A.data(), A.ld() * n, Location::kDEVICE);
      Vector<T_O> Q_data = A_data.template cast<T_O>();
      if (Q.ld() == A.ld()) {
         CUDA_CHECK(cudaMemcpy(Q.data(), Q_data.data(), A.ld() * n * sizeof(T_O), cudaMemcpyDeviceToDevice));
      }
      else {
         for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaMemcpy(&Q(0, i), &Q_data(0, i), m * sizeof(T_O), cudaMemcpyDeviceToDevice));
         }
      }
   }
   
   // Initialize R to zeros
   R.fill(get_zero<T_O>());
   
   // Vector to track skipped columns
   std::vector<int> skip(n);
   int* skip_d;
   CUDA_CHECK(cudaMalloc(&skip_d, n * sizeof(int)));
   CUDA_CHECK(cudaMemset(skip_d, 0, n * sizeof(int)));
   // We need a memory on host pinned memory to fast check if we should skip the current column
   int* skip_h;
   CUDA_CHECK(cudaMallocHost(&skip_h, sizeof(int)));
   
   // Vector to track permutation of columns (which rows have max elements)
   // In this version we need the entire permutation index, but not the values
   int* perm_indices_d;
   CUDA_CHECK(cudaMalloc(&perm_indices_d, n * sizeof(int)));
   T_COMPUTE* constant_perm_val_pointer = nullptr;
   if constexpr (std::is_same_v<T_COMPUTE, double>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_double));
   } else if constexpr (std::is_same_v<T_COMPUTE, float>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_float));
   } else if constexpr (std::is_same_v<T_COMPUTE, __half>) {
      CUDA_CHECK(cudaGetSymbolAddress((void**)&constant_perm_val_pointer, perm_val_half));
   }

   // The scalefactor array
   Matrix<T_COMPUTE> scale_factor_matrix(n, n, Location::kDEVICE);
   scale_factor_matrix.fill(get_zero<T_COMPUTE>());

   // Process each column, we use a different implementation
   for (size_t i = 0; i < n; i++) {
      hessenberg_v3_step<T_O, T_COMPUTE>(Q, R, skip_h, skip_d, i, perm_indices_d, scale_factor_matrix, constant_perm_val_pointer, select_tol);
   }

   // Copy skip from device to host
   CUDA_CHECK(cudaMemcpy(skip.data(), skip_d, n * sizeof(int), cudaMemcpyDeviceToHost));

   // Also copy the scale factor matrix to host
   scale_factor_matrix.to_host();

   // Note that the two matrices R and scale_factor_matrix might have different datatypes
   // and also one of them is a transposed matrix
   // For now use double for loop to copy the elements, definetly have better ways to do this
   if constexpr (std::is_same_v<T_O, double>) {
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i; j < n; j++) {
            R(i, j) = scale_factor_matrix(i, j);
         }
      }
   } else if constexpr (std::is_same_v<T_O, float>) {
      for (size_t i = 0; i < n; i++) {
         for (size_t j = i; j < n; j++) {
            R(i, j) = scale_factor_matrix(i, j);
         }
      }
   } else if constexpr (std::is_same_v<T_O, __half>) {
      if constexpr (std::is_same_v<T_COMPUTE, __half>) {
         for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < n; j++) {
               R(i, j) = scale_factor_matrix(i, j);
            }
         }
      } else {
         for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < n; j++) {
               R(i, j) = scale_factor_matrix(i, j);
            }
         }
      }
   }

   CUDA_CHECK(cudaFree(skip_d));
   CUDA_CHECK(cudaFreeHost(skip_h));
   CUDA_CHECK(cudaFree(perm_indices_d));

   return skip;
}

template<typename T_O, typename T_COMPUTE>
int hessenberg_v3_step(
   Matrix<T_O>& Q,
   Matrix<T_O>& R,
   int* skip_h,
   int* skip_d,
   size_t i,
   int* perm_indices_d,
   Matrix<T_COMPUTE>& scale_factor_matrix,
   T_COMPUTE* constant_perm_val_pointer,
   T_O select_tol
) {
   size_t m = Q.rows();
   size_t n = Q.cols();

   // Create a reference vector to column i
   Vector<T_O> q_i(&Q(0, i), m, Location::kDEVICE);

   if(i > 0)
   {
      constexpr int block_size = 1024;
      for(size_t j = 0; j < i; j++) {
         hessenberg_compute_scale_kernel_v3<T_O, T_COMPUTE><<<1,1>>>(
            &Q(0,i), perm_indices_d + j, &scale_factor_matrix(j, i)
         );
         CUDA_CHECK(cudaMemcpy(constant_perm_val_pointer, &scale_factor_matrix(j, i), sizeof(T_COMPUTE), cudaMemcpyDeviceToDevice));
         hessenberg_scale_kernel_v3<T_O, T_COMPUTE><<<(m + block_size - 1) / block_size, block_size>>>(
            &Q(0,j), &Q(0,i), m
         );
      }
   }

   // Get max index directly to device memory index
   MSVDStatus status = iamax<T_O, T_COMPUTE>(q_i, perm_indices_d[i], scale_factor_matrix(i, i), Location::kDEVICE);
   if (status != MSVDStatus::kSuccess) {
      throw std::runtime_error("iamax failed for column " + std::to_string(i));
   }

   CUDA_CHECK(cudaMemcpy(constant_perm_val_pointer, &scale_factor_matrix(i, i), sizeof(T_COMPUTE), cudaMemcpyDeviceToDevice));

   // Scale the current column
   hessenberg_colscale_kernel<T_O, T_COMPUTE><<<(m + SCALE_KERNEL_BLOCK_SIZE - 1) / SCALE_KERNEL_BLOCK_SIZE, SCALE_KERNEL_BLOCK_SIZE>>>(
      &Q(0, i), m, select_tol, skip_d + i
   );
   // Check if we can skip the scaleing step
   CUDA_CHECK(cudaMemcpy(skip_h, skip_d + i, sizeof(int), cudaMemcpyDeviceToHost));
   if (skip_h[0] == 1) {
      return 1;
   }

   return 0;
}

// Explicit template instantiations
template std::vector<int> hessenberg_v3<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&, double, double, double);

template std::vector<int> hessenberg_v3<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&, float, float, float);

template std::vector<int> hessenberg_v3<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&, __half, __half, __half);

template int hessenberg_v3_step<double, double>(
   Matrix<double>&, Matrix<double>&, int*, int*, size_t, int*, Matrix<double>&, double*, double);

template int hessenberg_v3_step<float, float>(
   Matrix<float>&, Matrix<float>&, int*, int*, size_t, int*, Matrix<float>&, float*, float);

template int hessenberg_v3_step<__half, float>(
   Matrix<__half>&, Matrix<__half>&, int*, int*, size_t, int*, Matrix<float>&, float*, __half);

} // namespace msvd 
