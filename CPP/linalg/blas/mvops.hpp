#pragma once
#include "../../containers/vector.hpp"
#include "../../containers/matrix.hpp"
#include "../../core/utils/error_handling.hpp"

namespace msvd {

/*

template <typename T>
bool check_nan(const Vector<T>& V, int n, int k);

template <typename T>
void scalecols(Vector<T>& V, int n, int k);

*/

/**
 * @brief Compute the dot product between two vectors
 * @param [in] x First vector
 * @param [in] y Second vector
 * @param [out] result Result of dot product
 * @details Unified CPU/GPU interface for dot product calculation \n
 *          Supports different input, output and compute precision types \n
 *          Automatically detects whether vectors are on host or device \n
 *          Uses the full length of the shorter vector
 * @return Status code indicating success or error
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus dot(const Vector<T_I>& x, const Vector<T_I>& y, T_O& result);

/**
 * @brief Compute the 2-norm (Euclidean norm) of a vector
 * @param [in] x Input vector
 * @param [out] result Result of 2-norm calculation
 * @details Unified CPU/GPU interface for 2-norm calculation \n
 *          Supports different input, output and compute precision types \n
 *          On CPU: supports all FP32, all FP64 \n
 *          On GPU: supports all FP32, all FP64, and FP16 I/O with FP32 compute \n
 *          Automatically detects whether vector is on host or device
 * @return Status code indicating success or error
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus norm2(const Vector<T_I>& x, T_O& result);

/**
 * @brief Performs general matrix-vector multiplication: y = alpha * op(A) * x + beta * y
 * @param [in] trans Specifies whether to transpose matrix A (true for transpose, false for no transpose)
 * @param [in] alpha Scalar multiplier for the matrix-vector product
 * @param [in] A Matrix in column-major format
 * @param [in] x Input vector
 * @param [in] beta Scalar multiplier for vector y
 * @param [in,out] y Output vector (overwritten with the result)
 * @details Unified CPU/GPU interface for matrix-vector multiplication \n
 *          Supports both single and double precision (FP32 and FP64) \n
 *          Automatically detects whether vectors and matrix are on host or device
 * @return Status code indicating success or error
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus gemv(bool trans, 
                const T_COMPUTE& alpha, const Matrix<T_I>& A, const Vector<T_I>& x, 
                const T_COMPUTE& beta, Vector<T_O>& y);

/**
 * @brief Performs general matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C
 * @param [in] transA Specifies whether to transpose matrix A (true for transpose, false for no transpose)
 * @param [in] transB Specifies whether to transpose matrix B (true for transpose, false for no transpose)
 * @param [in] alpha Scalar multiplier for the matrix-matrix product
 * @param [in] A First matrix in column-major format
 * @param [in] B Second matrix in column-major format
 * @param [in] beta Scalar multiplier for matrix C
 * @param [in,out] C Output matrix (overwritten with the result)
 * @details Unified CPU/GPU interface for matrix-matrix multiplication \n
 *          Host supports single and double precision (FP32 and FP64) \n
 *          Device supports FP64, FP32, and FP16 (with both FP32 and FP16 compute) \n
 *          Automatically detects whether matrices are on host or device
 * @return Status code indicating success or error
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus gemm(bool transA, bool transB,
                const T_COMPUTE& alpha, const Matrix<T_I>& A, const Matrix<T_I>& B,
                const T_COMPUTE& beta, Matrix<T_O>& C);

/**
 * @brief Finds the index of the element with maximum absolute value in a vector
 * @note Warning: result are 1-based, not 0-based. Result_val will not be written back 
 *       if location is kDEVICE and result is kHOST and T and T_O are double or float
 * @param [in] x Input vector
 * @param [out] result Index of the element with maximum absolute value (0-based)
 * @param [out] result_val Value of the element with maximum absolute value
 * @param [in] location Location of the result (default is kHOST, result should be provided on the given location)
 * @details Unified CPU/GPU interface for finding the index of max abs value \n
 *          Host supports single and double precision (FP32 and FP64) \n
 *          Device supports FP64, FP32, and FP16 \n
 *          Automatically detects whether vector is on host or device
 * @return Status code indicating success or error
 */
template<typename T, typename T_O = T>
MSVDStatus iamax(const Vector<T>& x, int& result, T_O& result_val, Location location = Location::kHOST);

/**
 * @brief Finds indices of the maximum absolute value elements in each column of a matrix
 * @param [in] A Input matrix
 * @param [out] results Array of indices, one for each column (0-based)
 * @param [in] location Location of results array (default is kHOST)
 * @details Unified CPU/GPU interface for finding indices of max abs values per column \n
 *          Host supports single and double precision (FP32 and FP64) \n
 *          Device supports FP64, FP32, and FP16 \n
 *          Automatically detects whether matrix is on host or device \n
 *          The results array must be pre-allocated with size at least equal to the number of columns in A
 * @return Status code indicating success or error
 */
template<typename T>
MSVDStatus matrix_iamax(const Matrix<T>& A, int* results, Location location = Location::kHOST);

/**
 * @brief Performs the operation y = alpha * x + y
 * @param [in] alpha Scalar multiplier
 * @param [in] x Input vector
 * @param [in,out] y Input/Output vector, overwritten with result
 * @details Unified CPU/GPU interface for AXPY operation \n
 *          Supports different input, output and compute precision types \n
 *          Host supports single and double precision (FP32 and FP64) \n
 *          Device supports FP64, FP32, and FP16 (with FP32 compute for FP16) \n
 *          Automatically detects whether vectors are on host or device \n
 *          Uses the full length of the shorter vector
 * @return Status code indicating success or error
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus axpy(const T_COMPUTE& alpha, const Vector<T_I>& x, Vector<T_O>& y);

/**
 * @brief Performs the operation y = alpha * x
 * @param [in] alpha Scalar multiplier
 * @param [in] x Input vector
 * @param [out] y Output vector, overwritten with result
 * @details Unified CPU/GPU interface for scaling operation \n
 *          Supports different input, output and compute precision types \n
 *          Host supports single and double precision (FP32 and FP64) \n
 *          Device supports FP64, FP32, and FP16 (with FP32 compute for FP16) \n
 *          Automatically detects whether vectors are on host or device \n
 *          Uses the full length of the shorter vector
 * @return Status code indicating success or error
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus scale(const T_COMPUTE& alpha, const Vector<T_I>& x, Vector<T_O>& y);

/**
 * @brief Check for NaN or Inf values in a vector
 * @param [in] x Input vector to check
 * @details Checks every element in the vector for NaN or Inf values \n
 *          Works for both CPU and GPU vectors \n
 *          Supports FP32 and FP64 precision for CPU and GPU, FP16 for GPU only
 * @return Status code indicating success or error type
 *         MSVDStatus::kSuccess - No NaN or Inf values detected
 *         MSVDStatus::kErrorNaN - NaN value detected
 *         MSVDStatus::kErrorInf - Inf value detected
 */
template<typename T>
MSVDStatus check_special_values(const Vector<T>& x);

/**
 * @brief Check for NaN or Inf values in a matrix
 * @param [in] A Input matrix to check
 * @details Checks every element in the matrix for NaN or Inf values \n
 *          Works for both CPU and GPU matrices \n
 *          Supports FP32 and FP64 precision for CPU and GPU, FP16 for GPU only
 * @return Status code indicating success or error type
 *         MSVDStatus::kSuccess - No NaN or Inf values detected
 *         MSVDStatus::kErrorNaN - NaN value detected
 *         MSVDStatus::kErrorInf - Inf value detected
 */
template<typename T>
MSVDStatus check_special_values(const Matrix<T>& A);

/**
 * @brief Compute the eigenvalues and eigenvectors of a symmetric matrix stored in upper triangular form
 * @note Due to the nature of the problem, only FP64 and FP32 are supported, and everything must be on host
 * @param [in, out] A Input symmetric matrix, on return the eigenvectors
 * @param [out] W Eigenvalues
 * @param [in, out] work Workspace array for the computation
 * @param [in] lwork Size of the workspace array
 */
template<typename T>
MSVDStatus syev( Matrix<T>& A, Vector<T>& W, T* work, int lwork);

/**
 * @brief Compute the eigenvalues and eigenvectors of a symmetric matrix stored in upper triangular form
 * @note Due to the nature of the problem, only FP64 and FP32 are supported, and everything must be on host
 * @param [in, out] A Input symmetric matrix, on return the eigenvectors
 * @param [in, out] B Input symmetric matrix, on return the Cholesky factor of B
 * @param [out] W Eigenvalues
 * @param [in, out] work Workspace array for the computation
 * @param [in] lwork Size of the workspace array
 */
template<typename T>
MSVDStatus sygv( Matrix<T>& A, Matrix<T>& B, Vector<T>& W, T* work, int lwork);

} // namespace msvd 