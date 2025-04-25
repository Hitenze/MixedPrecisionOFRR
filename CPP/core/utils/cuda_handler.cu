#include "cuda_handler.hpp"
#include <stdexcept>

namespace msvd {

// Singleton instance initialization
CUDAHandler* CUDAHandler::instance = nullptr;

CUDAHandler* CUDAHandler::get_instance() {
   if (instance == nullptr) {
      instance = new CUDAHandler();
   }
   return instance;
}

void CUDAHandler::init() {
   CUDAHandler* handler = get_instance();
   
   CUBLAS_CHECK(cublasCreate(&handler->cublas_handle));
   CUSPARSE_CHECK(cusparseCreate(&handler->cusparse_handle));
   CUSOLVER_CHECK(cusolverDnCreate(&handler->cusolver_handle));
}

void CUDAHandler::finalize() {
   if (instance != nullptr) {
      CUBLAS_CHECK(cublasDestroy(instance->cublas_handle));
      CUSPARSE_CHECK(cusparseDestroy(instance->cusparse_handle));
      CUSOLVER_CHECK(cusolverDnDestroy(instance->cusolver_handle));
      
      delete instance;
      instance = nullptr;
   }
}

cublasHandle_t CUDAHandler::cublas() {
   return get_instance()->cublas_handle;
}

cusparseHandle_t CUDAHandler::cusparse() {
   return get_instance()->cusparse_handle;
}

cusolverDnHandle_t CUDAHandler::cusolver() {
   return get_instance()->cusolver_handle;
}

} // namespace msvd 