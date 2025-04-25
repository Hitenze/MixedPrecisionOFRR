#include "arnoldi_svd.hpp"
#include "rr_svd.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

namespace msvd {

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus arnoldi_svd(const Matrix<T_I>& A, Matrix<T_O>& U, Matrix<T_O>& S, Matrix<T_O>& V, 
                        int max_iter, T_COMPUTE tol, 
                        bool use_generalized, T_COMPUTE select_tol) {
   // TODO: Implementation of Arnoldi algorithm for SVD problem
   
   // For now, just return success
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus arnoldi_svd<double, double, double>(const Matrix<double>& A, Matrix<double>& U, Matrix<double>& S, Matrix<double>& V, 
                                                    int max_iter, double tol, 
                                                    bool use_generalized, double select_tol);
template MSVDStatus arnoldi_svd<float, float, float>(const Matrix<float>& A, Matrix<float>& U, Matrix<float>& S, Matrix<float>& V, 
                                                 int max_iter, float tol, 
                                                 bool use_generalized, float select_tol);
template MSVDStatus arnoldi_svd<double, float, float>(const Matrix<double>& A, Matrix<float>& U, Matrix<float>& S, Matrix<float>& V, 
                                                  int max_iter, float tol, 
                                                  bool use_generalized, float select_tol);
template MSVDStatus arnoldi_svd<float, double, double>(const Matrix<float>& A, Matrix<double>& U, Matrix<double>& S, Matrix<double>& V, 
                                                   int max_iter, double tol, 
                                                   bool use_generalized, double select_tol);
template MSVDStatus arnoldi_svd<__half, __half, float>(const Matrix<__half>& A, Matrix<__half>& U, Matrix<__half>& S, Matrix<__half>& V, 
                                                   int max_iter, float tol, 
                                                   bool use_generalized, float select_tol);
template MSVDStatus arnoldi_svd<__half, float, float>(const Matrix<__half>& A, Matrix<float>& U, Matrix<float>& S, Matrix<float>& V, 
                                                  int max_iter, float tol, 
                                                  bool use_generalized, float select_tol);

} // namespace msvd 