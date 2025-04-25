#pragma once

#include "../../../containers/matrix.hpp"
#include "../../../containers/vector.hpp"
#include "../../../core/memory/location.hpp"
#include "../../../core/utils/error_handling.hpp"
#include "../../blas/mvops.hpp"
#include "../../factorization/qr.hpp"
#include "rr_eig.hpp"

namespace msvd {

/**
 * @brief Arnoldi method for eigenvalue problem
 * @details Computes eigenvalues and eigenvectors of a matrix using the Arnoldi method
 * @param[in] A Matrix whose eigenvalues are to be computed
 * @param[in,out] V Initial vector, overwritten with the computed eigenvectors
 * @param[out] D Diagonal matrix of eigenvalues
 * @param[in] max_iter Maximum number of iterations (maximum Krylov subspace dimension)
 * @param[in] tol Convergence tolerance
 * @param[in] use_generalized Whether to use generalized eigenvalue problem in the Rayleigh-Ritz procedure
 * @param[in] select_tol Tolerance parameter for selecting vectors in the generalized problem
 * @return MSVDStatus Status code
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus arnoldi_eig(const Matrix<T_I>& A, Matrix<T_O>& V, Matrix<T_O>& D, 
                  int max_iter = 100, T_COMPUTE tol = 1e-10, 
                  bool use_generalized = false, T_COMPUTE select_tol = 1e-10);

} // namespace msvd 