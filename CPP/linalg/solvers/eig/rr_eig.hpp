#pragma once

#include "../../../containers/matrix.hpp"
#include "../../../containers/vector.hpp"
#include "../../../core/memory/location.hpp"
#include "../../../core/utils/error_handling.hpp"
#include "../../blas/mvops.hpp"

namespace msvd {

/**
 * @brief Rayleigh-Ritz method for eigenvalue problem
 * @details Computes eigenvalues and eigenvectors of a matrix using the Rayleigh-Ritz method
 * @param[in] A Matrix whose eigenvalues are to be computed
 * @param[in,out] V Initial approximate eigenvectors, overwritten with the computed eigenvectors
 * @param[out] D Diagonal matrix of eigenvalues
 * @param[in] use_generalized Whether to use generalized eigenvalue problem (A*x = lambda*B*x)
 * @param[in] select_tol Tolerance parameter for selecting vectors in the generalized problem
 * @return MSVDStatus Status code
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus rr_eig(const Matrix<T_I>& A, Matrix<T_O>& V, Matrix<T_O>& D, bool use_generalized = false, T_COMPUTE select_tol = 1e-10);

/**
 * @brief Orthogonalized frequency domain Rayleigh-Ritz method (OFRR) for generalized eigenvalue problem
 * @details Computes eigenvalues and eigenvectors of a generalized eigenvalue problem (A*x = lambda*B*x)
 * @param[in] A Matrix A in the generalized eigenvalue problem
 * @param[in] B Matrix B in the generalized eigenvalue problem
 * @param[out] V Computed eigenvectors
 * @param[out] D Diagonal matrix of eigenvalues
 * @param[in] select_tol Tolerance parameter for the stability of the generalized eigenvalue problem
 * @return MSVDStatus Status code
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus ofrr_eig(const Matrix<T_I>& A, const Matrix<T_I>& B, Matrix<T_O>& V, Matrix<T_O>& D, T_COMPUTE select_tol = 1e-10);

} // namespace msvd 