#pragma once
#include "../../containers/matrix.hpp"
#include "../../core/utils/type_utils.hpp"
#include <vector>
#include <type_traits>

namespace msvd {

/**
 * @brief Modified Gram-Schmidt QR factorization
 * @details Column-oriented implementation for matrices stored in column-major format
 * 
 * @param [in] A Input matrix (column-major) Must be m x n and on device
 * @param [out] Q Output orthogonal matrix. Must be m x n and on device
 * @param [out] R Output upper triangular matrix. Must be n x n and on host
 * @param [in] orth_tol Tolerance for orthogonalization (default is machine epsilon)
 * @param [in] reorth_tol Tolerance for reorthogonalization (default is 1/sqrt(2))
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> mgs(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

/**
 * @brief Modified Gram-Schmidt QR factorization without reorthogonalization
 * @details Uses GEMM for batch orthogonalization and applies MGS once for efficiency
 * 
 * @param [in] A Input matrix (column-major) Must be m x n and on device
 * @param [out] Q Output orthogonal matrix. Must be m x n and on device
 * @param [out] R Output upper triangular matrix. Must be n x n and on host
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @param [in] orth_tol Placeholder, not used
 * @param [in] reorth_tol Placeholder, not used
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> mgs_v2(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

/**
 * @brief Modified Gram-Schmidt step
 * @details Performs a single step of the Modified Gram-Schmidt process
 * 
 * @param [in] A Input matrix (column-major) Must be m x n and on device
 * @param [out] Q Output orthogonal matrix. Must be m x n and on device
 * @param [out] R Output upper triangular matrix. Must be n x n and on host
 * @param [in,out] skip Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 * @param [in] i Column index to process, orth against columns 0, 1, ..., i-1
 * @param [in] orth_tol Tolerance for orthogonalization (default is machine epsilon)
 * @param [in] reorth_tol Tolerance for reorthogonalization (default is 1/sqrt(2))
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @return int 1 if column was skipped, 0 otherwise
 */
template<typename T_O, typename T_COMPUTE>
int mgs_step(
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   std::vector<int>& skip,
   size_t i,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

/**
 * @brief Classical Gram-Schmidt QR factorization
 * @details Uses GEMM for batch orthogonalization and applies CGS once dd
 * 
 * @param [in] A Input matrix (column-major) Must be m x n and on device
 * @param [out] Q Output orthogonal matrix. Must be m x n and on device
 * @param [out] R Output upper triangular matrix. Must be n x n and on host
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @param [in] orth_tol Placeholder, not used
 * @param [in] reorth_tol Placeholder, not used
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> cgs(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

template<typename T_O, typename T_COMPUTE>
int cgs_step(
   Matrix<T_O>& Q, 
   Matrix<T_O>& R_device,
   Vector<T_O>& r_diag,
   std::vector<int>& skip,
   size_t i,
   T_O select_tol = get_eps<T_O>()
);


/**
 * @brief Classical Gram-Schmidt QR factorization with two iterations
 * @details Uses GEMM for batch orthogonalization and applies CGS twice for numerical stability
 * 
 * @param [in] A Input matrix (column-major) Must be m x n and on device
 * @param [out] Q Output orthogonal matrix. Must be m x n and on device
 * @param [out] R Output upper triangular matrix. Must be n x n and on host
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @param [in] orth_tol Placeholder, not used
 * @param [in] reorth_tol Placeholder, not used
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> cgs2(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

template<typename T_O, typename T_COMPUTE>
int cgs2_step(
   Matrix<T_O>& Q, 
   Matrix<T_O>& R_device,
   Vector<T_O>& r_correction,
   Vector<T_O>& r_diag,
   std::vector<int>& skip,
   size_t i,
   T_O select_tol = get_eps<T_O>()
);


} // namespace msvd 