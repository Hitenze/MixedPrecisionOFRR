#pragma once

#include "../../../containers/matrix.hpp"
#include "../../../containers/vector.hpp"
#include "../../../core/memory/location.hpp"
#include "../../../core/utils/error_handling.hpp"
#include "../../blas/mvops.hpp"

namespace msvd {

/**
 * @brief Rayleigh-Ritz method for singular value decomposition
 * @details Computes singular values and vectors of a matrix using the Rayleigh-Ritz method
 * @param[in] A Matrix whose SVD is to be computed
 * @param[in,out] U Initial left singular vectors, overwritten with the computed vectors
 * @param[out] S Diagonal matrix of singular values
 * @param[in,out] V Initial right singular vectors, overwritten with the computed vectors
 * @param[in] use_generalized Whether to use generalized eigenvalue problem
 * @param[in] select_tol Tolerance parameter for the stability of the generalized eigenvalue problem
 * @return MSVDStatus Status code
 */
template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus rr_svd(const Matrix<T_I>& A, Matrix<T_O>& U, Matrix<T_O>& S, Matrix<T_O>& V, 
             bool use_generalized = false, T_COMPUTE select_tol = 1e-10);

} // namespace msvd 