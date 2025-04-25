#include "rr_eig.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

namespace msvd {

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus rr_eig(const Matrix<T_I>& A, Matrix<T_O>& V, Matrix<T_O>& D, bool use_generalized, T_COMPUTE select_tol) {
   
   return MSVDStatus::kSuccess;
}

template<typename T_I, typename T_O, typename T_COMPUTE>
MSVDStatus ofrr_eig(const Matrix<T_I>& A, const Matrix<T_I>& B, Matrix<T_O>& V, Matrix<T_O>& D, T_COMPUTE select_tol) {
   
   return MSVDStatus::kSuccess;
}

// Explicit instantiation for supported types
template MSVDStatus rr_eig<double, double, double>(const Matrix<double>& A, Matrix<double>& V, Matrix<double>& D, bool use_generalized, double select_tol);
template MSVDStatus rr_eig<float, float, float>(const Matrix<float>& A, Matrix<float>& V, Matrix<float>& D, bool use_generalized, float select_tol);
template MSVDStatus rr_eig<double, float, float>(const Matrix<double>& A, Matrix<float>& V, Matrix<float>& D, bool use_generalized, float select_tol);
template MSVDStatus rr_eig<float, double, double>(const Matrix<float>& A, Matrix<double>& V, Matrix<double>& D, bool use_generalized, double select_tol);
template MSVDStatus rr_eig<__half, __half, float>(const Matrix<__half>& A, Matrix<__half>& V, Matrix<__half>& D, bool use_generalized, float select_tol);
template MSVDStatus rr_eig<__half, float, float>(const Matrix<__half>& A, Matrix<float>& V, Matrix<float>& D, bool use_generalized, float select_tol);

template MSVDStatus ofrr_eig<double, double, double>(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& V, Matrix<double>& D, double select_tol);
template MSVDStatus ofrr_eig<float, float, float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& V, Matrix<float>& D, float select_tol);
template MSVDStatus ofrr_eig<double, float, float>(const Matrix<double>& A, const Matrix<double>& B, Matrix<float>& V, Matrix<float>& D, float select_tol);
template MSVDStatus ofrr_eig<float, double, double>(const Matrix<float>& A, const Matrix<float>& B, Matrix<double>& V, Matrix<double>& D, double select_tol);
template MSVDStatus ofrr_eig<__half, __half, float>(const Matrix<__half>& A, const Matrix<__half>& B, Matrix<__half>& V, Matrix<__half>& D, float select_tol);
template MSVDStatus ofrr_eig<__half, float, float>(const Matrix<__half>& A, const Matrix<__half>& B, Matrix<float>& V, Matrix<float>& D, float select_tol);

} // namespace msvd 