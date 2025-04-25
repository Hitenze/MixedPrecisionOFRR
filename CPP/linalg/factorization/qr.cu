#include "qr.hpp"
#include "../../linalg/blas/mvops.hpp"
#include "../../core/utils/cuda_handler.hpp"
#include "../../core/utils/type_utils.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace msvd {

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> mgs(
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
   // Create a pointer to the data for direct copying
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
   std::vector<int> skip(n, 0);
   
   // Process each column
   for (size_t i = 0; i < n; i++) {
      mgs_step<T_O, T_COMPUTE>(Q, R, skip, i, select_tol, orth_tol, reorth_tol);
   }
   
   return skip;
}


template<typename T_O, typename T_COMPUTE>
int mgs_step(
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   std::vector<int>& skip,
   size_t i,
   T_O select_tol,
   T_O orth_tol,
   T_O reorth_tol
) {
   // Get dimensions
   size_t m = Q.rows();
   size_t n = Q.cols();
   
   // Create vector objects for column operations
   Vector<T_O> q_col(&Q(0, i), m, Location::kDEVICE);
   
   // Check initial norm of column i
   T_O normv;
   norm2<T_O, T_O, T_COMPUTE>(q_col, normv);
   
   // Skip if column is too small
   if (normv < select_tol) {
      skip[i] = 1;
      return 1;
   }
   
   // Orthogonalize against previous columns
   for (size_t j = 0; j < i; j++) {
      if (skip[j]) continue;
      
      // Create vector reference for column j
      Vector<T_O> q_j(&Q(0, j), m, Location::kDEVICE);
      
      // Compute projection
      T_O r_val;
      dot<T_O, T_O, T_COMPUTE>(q_col, q_j, r_val);
      
      // Store in R
      R(j, i) = r_val;
      
      // Orthogonalize: v = v - r_val * Q(:,j)
      if constexpr (std::is_same_v<T_O, __half>) {
         if constexpr (std::is_same_v<T_COMPUTE, float>) {
            T_COMPUTE neg_r_val = __half2float(-r_val);
            axpy<T_O, T_O, T_COMPUTE>(neg_r_val, q_j, q_col);
         }
         else {
            T_COMPUTE neg_r_val = __hneg(r_val);
            axpy<T_O, T_O, T_COMPUTE>(neg_r_val, q_j, q_col);
         }
      }
      else if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
         T_COMPUTE neg_r_val = -r_val;
         axpy<T_O, T_O, T_COMPUTE>(neg_r_val, q_j, q_col);
      }
      else {
         T_COMPUTE neg_r_val = static_cast<T_COMPUTE>(-r_val);
         axpy<T_O, T_O, T_COMPUTE>(neg_r_val, q_j, q_col);
      }
   }
   
   // Reorthogonalization if needed
   T_O t;
   norm2<T_O, T_O, T_COMPUTE>(q_col, t);

   bool reorth_flag = false;
   if constexpr (std::is_same_v<T_O, __half>) {
      reorth_flag = __half2float(reorth_tol) > 0.0f;
   }
   else {
      reorth_flag = reorth_tol > get_zero<T_O>();
   }

   while (reorth_flag && t >= orth_tol && t < reorth_tol * normv) {
      normv = t;
      
      // Reorthogonalize against all previous columns
      for (size_t j = 0; j < i; j++) {
         if (skip[j]) continue;
         
         // Create vector reference for column j
         Vector<T_O> q_j(&Q(0, j), m, Location::kDEVICE);
         
         // Compute projection
         T_O r;
         dot<T_O, T_O, T_COMPUTE>(q_col, q_j, r);
         
         // Update R
         // only supports FP32 computation for FP16 R
         if constexpr (std::is_same_v<T_O, __half>) {
            if constexpr (std::is_same_v<T_COMPUTE, float>) {
               R(j, i) = __float2half(__half2float(R(j, i)) + __half2float(r));
            }
            else {
               R(j, i) = __hadd(R(j, i), r);
            }
         }
         else {
            if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
               R(j, i) = R(j, i) + r;
            }
            else {
               R(j, i) = static_cast<T_O>(static_cast<T_COMPUTE>(R(j, i)) + static_cast<T_COMPUTE>(r));
            }
         }
         
         // Orthogonalize: v = v - r * Q(:,j)
         if constexpr (std::is_same_v<T_O, __half>) {
            if constexpr (std::is_same_v<T_COMPUTE, float>) {
               T_COMPUTE neg_r = -__half2float(r);
               axpy<T_O, T_O, T_COMPUTE>(neg_r, q_j, q_col);
            }
            else {
               T_COMPUTE neg_r = __hneg(r);
               axpy<T_O, T_O, T_COMPUTE>(neg_r, q_j, q_col);
            }
         }
         else if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
            T_COMPUTE neg_r = -r;
            axpy<T_O, T_O, T_COMPUTE>(neg_r, q_j, q_col);
         }
         else {
            T_COMPUTE neg_r = static_cast<T_COMPUTE>(-r);
            axpy<T_O, T_O, T_COMPUTE>(neg_r, q_j, q_col);
         }
      }
      
      // Recompute norm
      norm2<T_O, T_O, T_COMPUTE>(q_col, t);
   }

   // Check if column became too small after orthogonalization
   if (t < select_tol) {
      skip[i] = 1;
   } else {
      // Normalize and set R(i,i)
      T_COMPUTE one_over_t;
      if constexpr (std::is_same_v<T_O, __half>) {
         if constexpr (std::is_same_v<T_COMPUTE, float>) {
            one_over_t = 1.0f / __half2float(t);
         }
         else {
            one_over_t = __hdiv(get_one<T_COMPUTE>(), t);
         }
      }
      else {
         if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
            one_over_t = get_one<T_COMPUTE>() / t;
         }
         else {
            one_over_t = get_one<T_COMPUTE>() / static_cast<T_COMPUTE>(t);
         }
      }
      
      // Scale the column
      scale<T_O, T_O, T_COMPUTE>(one_over_t, q_col, q_col);
      
      // Copy back to Q
      CUDA_CHECK(cudaMemcpy(&Q(0, i), q_col.data(), m * sizeof(T_O), cudaMemcpyDeviceToDevice));
      
      R(i, i) = t;
   }
   
   return skip[i];
}

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> mgs_v2(
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
   
   // Vector to track skipped columns
   std::vector<int> skip(n, 0);

   // Helper vector and matrix
   Matrix<T_O> R_device(n, n, Location::kDEVICE);
   Vector<T_O> r_diag(n, Location::kHOST);
   // Initialize R to zeros
   R_device.fill(get_zero<T_O>());
   r_diag.fill(get_one<T_O>());
   
   // Constants
   T_COMPUTE alpha_neg_one = get_negone<T_COMPUTE>();
   T_COMPUTE alpha_one = get_one<T_COMPUTE>();
   T_COMPUTE beta_zero = get_zero<T_COMPUTE>();
   T_COMPUTE beta_one = get_one<T_COMPUTE>();
   
   // Process each column
   for (size_t i = 0; i < n; i++) {
      // Extract column i to Q_current
      Vector<T_O> q_col(&Q(0, i), m, Location::kDEVICE);
      
      // Check initial norm
      T_O t;
      norm2<T_O, T_O, T_COMPUTE>(q_col, t);
      
      // Skip if column is too small
      if (t < select_tol) {
         skip[i] = 1;
         continue;
      }

      r_diag[i] = t;

      // normalize column
      T_COMPUTE one_over_t;
      if constexpr (std::is_same_v<T_O, __half>) {
         if constexpr (std::is_same_v<T_COMPUTE, float>) {
            one_over_t = 1.0f / __half2float(t);
         }
         else {
            one_over_t = __hdiv(get_one<T_COMPUTE>(), t);
         }
      } else {
         if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
            one_over_t = get_one<T_COMPUTE>() / t;
         }
         else {
            one_over_t = get_one<T_COMPUTE>() / static_cast<T_COMPUTE>(t);
         }
      }
      // scale q_col to 1/t (in-place operation)
      scale<T_O, T_O, T_COMPUTE>(one_over_t, q_col, q_col);
      
      // Orthogonalize against all columns after it (note that this is different from CGS)
      if (i < n-1) {
         // Create reference to columns after q_col
         Matrix<T_O> Q_after(&Q(0, i+1), m, n-i-1, m, Location::kDEVICE);
         
         // Create reference to row of R
         Matrix<T_O> r_row(&R_device(i, i+1), 1, n-i-1, n, Location::kDEVICE);
         Matrix<T_O> q_col_mat(&Q(0, i), m, 1, m, Location::kDEVICE);

         // Compute projection coefficients: r_row = q_col^T * Q_after
         gemm<T_O, T_O, T_COMPUTE>(true, false, alpha_one, q_col_mat, Q_after, beta_zero, r_row);
         
         // Compute rank-1 update: Q_after = Q_after - q_col * r_row
         gemm<T_O, T_O, T_COMPUTE>(false, false, alpha_neg_one, q_col_mat, r_row, beta_one, Q_after);
      }
   }

   // Copy R_device to R
   if(R.ld() != n) {
      for(size_t i = 0; i < n; i++) {
         CUDA_CHECK(cudaMemcpy(&R(0, i), &R_device(0, i), n * sizeof(T_O), cudaMemcpyDeviceToHost));
      }
   }
   else {
      CUDA_CHECK(cudaMemcpy(R.data(), R_device.data(), n * n * sizeof(T_O), cudaMemcpyDeviceToHost));
   }

   // Set diagonal elements of R
   for (size_t i = 0; i < n; i++) {
      R(i, i) = r_diag[i];
   }
   
   return skip;
}

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> cgs(
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
   
   // Vector to track skipped columns
   std::vector<int> skip(n, 0);

   // Helper vector and matrix
   Matrix<T_O> R_device(n, n, Location::kDEVICE);
   Vector<T_O> r_diag(n, Location::kHOST);
   // Initialize R to zeros
   R_device.fill(get_zero<T_O>());
   r_diag.fill(get_one<T_O>());
   
   // Constants
   T_COMPUTE alpha_one = get_one<T_COMPUTE>();
   T_COMPUTE alpha_neg_one = get_negone<T_COMPUTE>();
   T_COMPUTE beta_zero = get_zero<T_COMPUTE>();
   
   // Process each column
   for (size_t i = 0; i < n; i++) {
      cgs_step<T_O, T_COMPUTE>(Q, R_device, r_diag, skip, i, select_tol);
   }

   // Copy R_device to R
   if(R.ld() != n) {
      for(size_t i = 0; i < n; i++) {
         CUDA_CHECK(cudaMemcpy(&R(0, i), &R_device(0, i), n * sizeof(T_O), cudaMemcpyDeviceToHost));
      }
   }
   else {
      CUDA_CHECK(cudaMemcpy(R.data(), R_device.data(), n * n * sizeof(T_O), cudaMemcpyDeviceToHost));
   }

   // Set diagonal elements of R
   for (size_t i = 0; i < n; i++) {
      R(i, i) = r_diag[i];
   }
   
   return skip;
}

template<typename T_O, typename T_COMPUTE>
int cgs_step(
   Matrix<T_O>& Q,
   Matrix<T_O>& R_device,
   Vector<T_O>& r_diag,
   std::vector<int>& skip,
   size_t i,
   T_O select_tol
) {
   // Get dimensions
   size_t m = Q.rows();
   size_t n = Q.cols();
   
   // Constants
   T_COMPUTE alpha_one = get_one<T_COMPUTE>();
   T_COMPUTE alpha_neg_one = get_negone<T_COMPUTE>();
   T_COMPUTE beta_zero = get_zero<T_COMPUTE>();
   
   // Extract column i to Q_current
   Vector<T_O> q_col(&Q(0, i), m, Location::kDEVICE);
   
   // Check initial norm
   T_O normv;
   norm2<T_O, T_O, T_COMPUTE>(q_col, normv);
   
   // Skip if column is too small
   if (normv < select_tol) {
      skip[i] = 1;
      return 1;
   }
   
   // Orthogonalize against previous columns (first CGS)
   if (i > 0) {
      // Create reference to previous i columns
      Matrix<T_O> Q_prev(&Q(0, 0), m, i, m, Location::kDEVICE);
      
      // Create reference to helper vectors
      Vector<T_O> r_col(&R_device(0, i), i, Location::kDEVICE);
      
      // Compute projection coefficients: r_col = Q_prev^T * q_col
      gemv<T_O, T_O, T_COMPUTE>(true, alpha_one, Q_prev, q_col, beta_zero, r_col);
      
      // Subtract projection: q_col = q_col - Q_prev * r_col
      gemv<T_O, T_O, T_COMPUTE>(false, alpha_neg_one, Q_prev, r_col, alpha_one, q_col);
   }
   
   // Compute norm
   T_O t;
   norm2<T_O, T_O, T_COMPUTE>(q_col, t);
   
   // Check if column became too small after orthogonalization
   if (t < select_tol) {
      skip[i] = 1;
   } else {
      // save diagonal element
      r_diag[i] = t;
      
      // normalize column
      T_COMPUTE one_over_t;
      if constexpr (std::is_same_v<T_O, __half>) {
         if constexpr (std::is_same_v<T_COMPUTE, float>) {
            one_over_t = 1.0f / __half2float(t);
         }
         else {
            one_over_t = __hdiv(get_one<T_COMPUTE>(), t);
         }
      } else {
         if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
            one_over_t = get_one<T_COMPUTE>() / t;
         }
         else {
            one_over_t = get_one<T_COMPUTE>() / static_cast<T_COMPUTE>(t);
         }
      }
      
      // scale q_col to 1/t (in-place operation)
      scale<T_O, T_O, T_COMPUTE>(one_over_t, q_col, q_col);
   }

   return skip[i];
}

template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> cgs2(
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
   
   // Vector to track skipped columns
   std::vector<int> skip(n, 0);

   // Helper vector and matrix
   Matrix<T_O> R_device(n, n, Location::kDEVICE);
   Vector<T_O> r_correction(n, Location::kDEVICE);
   Vector<T_O> r_diag(n, Location::kHOST);
   // Initialize R to zeros
   R_device.fill(get_zero<T_O>());
   r_diag.fill(get_one<T_O>());
   
   // Constants
   T_COMPUTE alpha_one = get_one<T_COMPUTE>();
   T_COMPUTE alpha_neg_one = get_negone<T_COMPUTE>();
   T_COMPUTE beta_zero = get_zero<T_COMPUTE>();
   
   // Process each column
   for (size_t i = 0; i < n; i++) {
      cgs2_step<T_O, T_COMPUTE>(Q, R_device, r_correction, r_diag, skip, i, select_tol);
   }

   // Copy R_device to R
   if(R.ld() != n) {
      for(size_t i = 0; i < n; i++) {
         CUDA_CHECK(cudaMemcpy(&R(0, i), &R_device(0, i), n * sizeof(T_O), cudaMemcpyDeviceToHost));
      }
   }
   else {
      CUDA_CHECK(cudaMemcpy(R.data(), R_device.data(), n * n * sizeof(T_O), cudaMemcpyDeviceToHost));
   }

   // Set diagonal elements of R
   for (size_t i = 0; i < n; i++) {
      R(i, i) = r_diag[i];
   }
   
   return skip;
}

template<typename T_O, typename T_COMPUTE>
int cgs2_step(
   Matrix<T_O>& Q, 
   Matrix<T_O>& R_device,
   Vector<T_O>& r_correction,
   Vector<T_O>& r_diag,
   std::vector<int>& skip,
   size_t i,
   T_O select_tol
) {
   // Get dimensions
   size_t m = Q.rows();
   size_t n = Q.cols();
   
   // Constants
   T_COMPUTE alpha_one = get_one<T_COMPUTE>();
   T_COMPUTE alpha_neg_one = get_negone<T_COMPUTE>();
   T_COMPUTE beta_zero = get_zero<T_COMPUTE>();
   
   // Extract column i to Q_current
   Vector<T_O> q_col(&Q(0, i), m, Location::kDEVICE);
   
   // Check initial norm
   T_O normv;
   norm2<T_O, T_O, T_COMPUTE>(q_col, normv);
   
   // Skip if column is too small
   if (normv < select_tol) {
      skip[i] = 1;
      return 1;
   }
   
   // Orthogonalize against previous columns (first CGS)
   if (i > 0) {
      // Create reference to previous i columns
      Matrix<T_O> Q_prev(&Q(0, 0), m, i, m, Location::kDEVICE);
      
      // Create reference to helper vectors
      Vector<T_O> r_col(&R_device(0, i), i, Location::kDEVICE);
      Vector<T_O> r_col_tmp(r_correction.data(), i, Location::kDEVICE);
      
      // Compute projection coefficients: r_col = Q_prev^T * q_col
      gemv<T_O, T_O, T_COMPUTE>(true, alpha_one, Q_prev, q_col, beta_zero, r_col);
      
      // Subtract projection: q_col = q_col - Q_prev * r_col
      gemv<T_O, T_O, T_COMPUTE>(false, alpha_neg_one, Q_prev, r_col, alpha_one, q_col);

      // Reorthogonalize
      gemv<T_O, T_O, T_COMPUTE>(true, alpha_one, Q_prev, q_col, beta_zero, r_col_tmp);

      // Add correction to R
      axpy<T_O, T_O, T_COMPUTE>(alpha_one, r_col_tmp, r_col);
   }
   
   // Compute norm
   T_O t;
   norm2<T_O, T_O, T_COMPUTE>(q_col, t);
   
   // Check if column became too small after orthogonalization
   if (t < select_tol) {
      skip[i] = 1;
   } else {
      // save diagonal element
      r_diag[i] = t;
      
      // normalize column
      T_COMPUTE one_over_t;
      if constexpr (std::is_same_v<T_O, __half>) {
         if constexpr (std::is_same_v<T_COMPUTE, float>) {
            one_over_t = 1.0f / __half2float(t);
         }
         else {
            one_over_t = __hdiv(get_one<T_COMPUTE>(), t);
         }
      } else {
         if constexpr (std::is_same_v<T_COMPUTE, T_O>) {
            one_over_t = get_one<T_COMPUTE>() / t;
         }
         else {
            one_over_t = get_one<T_COMPUTE>() / static_cast<T_COMPUTE>(t);
         }
      }
      
      // scale q_col to 1/t (in-place operation)
      scale<T_O, T_O, T_COMPUTE>(one_over_t, q_col, q_col);
   }

   return skip[i];
}

template std::vector<int> mgs<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&,
   double, double, double);

template std::vector<int> mgs<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&,
   float, float, float);

template std::vector<int> mgs<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&,
   __half, __half, __half);

template int mgs_step<double, double>(
   Matrix<double>&, Matrix<double>&, std::vector<int>&, size_t, double, double, double);

template int mgs_step<float, float>(
   Matrix<float>&, Matrix<float>&, std::vector<int>&, size_t, float, float, float);

template int mgs_step<__half, float>(
   Matrix<__half>&, Matrix<__half>&, std::vector<int>&, size_t, __half, __half, __half);

template std::vector<int> mgs_v2<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&, double, double, double);

template std::vector<int> mgs_v2<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&, float, float, float);

template std::vector<int> mgs_v2<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&, __half, __half, __half);

template std::vector<int> cgs<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&, double, double, double);

template std::vector<int> cgs<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&, float, float, float);

template std::vector<int> cgs<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&, __half, __half, __half);

template int cgs_step<double, double>(
   Matrix<double>&, Matrix<double>&, Vector<double>&, std::vector<int>&, size_t, double);

template int cgs_step<float, float>(
   Matrix<float>&, Matrix<float>&, Vector<float>&, std::vector<int>&, size_t, float);

template int cgs_step<__half, float>(
   Matrix<__half>&, Matrix<__half>&, Vector<__half>&, std::vector<int>&, size_t, __half);

template std::vector<int> cgs2<double, double, double>(
   const Matrix<double>&, Matrix<double>&, Matrix<double>&, double, double, double);

template std::vector<int> cgs2<float, float, float>(
   const Matrix<float>&, Matrix<float>&, Matrix<float>&, float, float, float);

template std::vector<int> cgs2<__half, __half, float>(
   const Matrix<__half>&, Matrix<__half>&, Matrix<__half>&, __half, __half, __half);

template int cgs2_step<double, double>(
   Matrix<double>&, Matrix<double>&, Vector<double>&, Vector<double>&, std::vector<int>&, size_t, double);

template int cgs2_step<float, float>(
   Matrix<float>&, Matrix<float>&, Vector<float>&, Vector<float>&, std::vector<int>&, size_t, float);

template int cgs2_step<__half, float>(
   Matrix<__half>&, Matrix<__half>&, Vector<__half>&, Vector<__half>&, std::vector<int>&, size_t, __half);

} // namespace msvd
