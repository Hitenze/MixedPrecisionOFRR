#pragma once
#include "error_handling.hpp"
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

namespace msvd {

/**
 * @brief Global handler for CUDA libraries
 * @details Singleton class to manage CUDA library handles \n
 *          Provides centralized access to cuBLAS, cuSPARSE and cuSOLVER handles \n
 *          Ensures proper initialization and cleanup of CUDA resources
 */
class CUDAHandler {
private:
   // Singleton instance
   static CUDAHandler* instance;
   
   // CUDA library handles
   cublasHandle_t cublas_handle;
   cusparseHandle_t cusparse_handle;
   cusolverDnHandle_t cusolver_handle;

   // Private constructor for singleton
   CUDAHandler() = default;
   
public:
   // Delete copy constructor and assignment
   CUDAHandler(const CUDAHandler&) = delete;
   CUDAHandler& operator=(const CUDAHandler&) = delete;

   /**
    * @brief Get singleton instance
    * @details Returns a pointer to the singleton instance of CUDAHandler \n
    *          If the instance doesn't exist, it will be created \n
    *          This method is thread-safe as long as it's not called during static initialization
    * @return Pointer to the singleton instance
    */
   static CUDAHandler* get_instance();

   /**
    * @brief Initialize all CUDA library handles
    * @details Creates and initializes handles for cuBLAS, cuSPARSE and cuSOLVER \n
    *          This method should be called once at the beginning of the program \n
    *          Will throw exceptions if any handle creation fails
    */
   static void init();

   /**
    * @brief Destroy all CUDA library handles
    * @details Properly releases all CUDA resources and destroys handles \n
    *          This method should be called before program termination \n
    *          It's safe to call this method multiple times
    */
   static void finalize();

   /**
    * @brief Get the cuBLAS handle
    * @details Provides access to the cuBLAS library handle for BLAS operations \n
    *          The handle is already initialized and ready to use
    * @return Initialized cuBLAS handle
    */
   static cublasHandle_t cublas();

   /**
    * @brief Get the cuSPARSE handle
    * @details Provides access to the cuSPARSE library handle for sparse matrix operations \n
    *          The handle is already initialized and ready to use
    * @return Initialized cuSPARSE handle
    */
   static cusparseHandle_t cusparse();

   /**
    * @brief Get the cuSOLVER handle
    * @details Provides access to the cuSOLVER library handle for linear algebra solvers \n
    *          The handle is already initialized and ready to use
    * @return Initialized cuSOLVER handle
    */
   static cusolverDnHandle_t cusolver();
};

} // namespace msvd 