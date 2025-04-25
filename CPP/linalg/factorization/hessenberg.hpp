#pragma once
#include "../../containers/matrix.hpp"
#include "../../core/utils/type_utils.hpp"
#include <vector>
#include <type_traits>

namespace msvd {

/**
 * @brief Hessenberg process
 * @details Hessenberg process, which selects the element with maximum
 *          absolute value in each column instead of using 2-norm scaling.
 * @note CUBLAS-based version
 * @param [in] A Input matrix (column-major). Must be m x n and on device
 * @param [out] Q Output matrix with linearly independent columns. Must be m x n and on device
 * @param [out] R Output coefficient matrix. Must be n x n and on host
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @param [in] orth_tol Placeholder, not used
 * @param [in] reorth_tol Placeholder, not used
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> hessenberg(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

/**
 * @brief Hessenberg process
 * @details Hessenberg process, which selects the element with maximum
 *          absolute value in each column instead of using 2-norm scaling.
 * @note Customized kernel version
 * 
 * @param [in] A Input matrix (column-major). Must be m x n and on device
 * @param [out] Q Output matrix with linearly independent columns. Must be m x n and on device
 * @param [out] R Output coefficient matrix. Must be n x n and on host
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @param [in] orth_tol Placeholder, not used
 * @param [in] reorth_tol Placeholder, not used
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> hessenberg_v2(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q, 
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

/**
 * @brief Hessenberg process
 * @details Hessenberg process, which selects the element with maximum
 *          absolute value in each column instead of using 2-norm scaling.
 * @note MGS-like version, following strategy exactly as in standard MGS (eliminating with columns in the front)
 * 
 * @param [in] A Input matrix (column-major). Must be m x n and on device
 * @param [out] Q Output matrix with linearly independent columns. Must be m x n and on device
 * @param [out] R Output coefficient matrix. Must be n x n and on host
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @param [in] orth_tol Placeholder, not used
 * @param [in] reorth_tol Placeholder, not used
 * @return std::vector<int> Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
std::vector<int> hessenberg_v3(
   const Matrix<T_I>& A, 
   Matrix<T_O>& Q,
   Matrix<T_O>& R,
   T_O select_tol = get_eps<T_O>(),
   T_O orth_tol = get_eps<T_O>(),
   T_O reorth_tol = get_one_over_sqrt_two<T_O>()
);

/**
 * @brief Hessenberg process
 * @details Hessenberg process, which selects the element with maximum
 *          absolute value in each column instead of using 2-norm scaling.
 * @note MGS-like version, following strategy exactly as in standard MGS (eliminating with columns in the front)
 * 
 * @param [in] Q Input matrix (column-major). Must be m x n and on device
 * @param [out] R Output coefficient matrix. Must be n x n and on host
 * @param [in,out] skip_h One element pinned host number
 * @param [in,out] skip_d Vector indicating which columns were skipped (1 if skipped, 0 otherwise)
 * @param [in] i Column index to process, orth against columns 0, 1, ..., i-1
 * @param [in] perm_indices_d Permutation indices on device
 * @param [in] scale_factor_matrix Scale factor matrix
 * @param [in] constant_perm_val_pointer Constant permutation value pointer
 * @param [in] select_tol Tolerance for selecting columns (default is machine epsilon)
 * @return int 1 if column was skipped, 0 otherwise
 */
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
   T_O select_tol = get_eps<T_O>()
);

} // namespace msvd 