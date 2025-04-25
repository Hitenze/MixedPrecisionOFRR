#include "arnoldi_eig.hpp"
#include "rr_eig.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

namespace msvd {

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus arnoldi_eig(const Matrix<T_I>& A, Matrix<T_O>& V, Matrix<T_O>& D, 
                   int max_iter, T_COMPUTE tol, 
                   bool use_generalized, T_COMPUTE select_tol) {
   // TODO: Implementation of Arnoldi algorithm for eigenvalue problem
   
   // For now, just return success
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus arnoldi_eig<double, double, double>(const Matrix<double>& A, Matrix<double>& V, Matrix<double>& D, 
                                                     int max_iter, double tol, 
                                                     bool use_generalized, double select_tol);
template MSVDStatus arnoldi_eig<float, float, float>(const Matrix<float>& A, Matrix<float>& V, Matrix<float>& D, 
                                                   int max_iter, float tol, 
                                                   bool use_generalized, float select_tol);
template MSVDStatus arnoldi_eig<double, float, float>(const Matrix<double>& A, Matrix<float>& V, Matrix<float>& D, 
                                                    int max_iter, float tol, 
                                                    bool use_generalized, float select_tol);
template MSVDStatus arnoldi_eig<float, double, double>(const Matrix<float>& A, Matrix<double>& V, Matrix<double>& D, 
                                                     int max_iter, double tol, 
                                                     bool use_generalized, double select_tol);
template MSVDStatus arnoldi_eig<__half, __half, float>(const Matrix<__half>& A, Matrix<__half>& V, Matrix<__half>& D, 
                                                     int max_iter, float tol, 
                                                     bool use_generalized, float select_tol);
template MSVDStatus arnoldi_eig<__half, float, float>(const Matrix<__half>& A, Matrix<float>& V, Matrix<float>& D, 
                                                    int max_iter, float tol, 
                                                    bool use_generalized, float select_tol);

} // namespace msvd 