#pragma once

#include "../core/utils/error_handling.hpp"

#include <cuda_runtime_api.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace msvd::examples {

/**
 * @brief Reusable CUDA event timer for work submitted to one stream.
 *
 * The stop event is synchronized before a duration is returned, so asynchronous
 * launch failures are reported at the benchmark boundary. Event handles are
 * owned by the timer and released during stack unwinding.
 */
class CudaEventTimer {
public:
   explicit CudaEventTimer(cudaStream_t stream = nullptr) : _stream(stream) {
      CUDA_CHECK(cudaEventCreate(&_start));
      try {
         CUDA_CHECK(cudaEventCreate(&_stop));
      } catch (...) {
         cudaEventDestroy(_start);
         _start = nullptr;
         throw;
      }
   }

   ~CudaEventTimer() noexcept {
      if (_stop != nullptr) {
         cudaEventDestroy(_stop);
      }
      if (_start != nullptr) {
         cudaEventDestroy(_start);
      }
   }

   CudaEventTimer(const CudaEventTimer&) = delete;
   CudaEventTimer& operator=(const CudaEventTimer&) = delete;
   CudaEventTimer(CudaEventTimer&&) = delete;
   CudaEventTimer& operator=(CudaEventTimer&&) = delete;

   template<typename Function>
   double measure_ms(Function&& function) {
      CUDA_CHECK(cudaEventRecord(_start, _stream));
      std::invoke(std::forward<Function>(function));
      CUDA_CHECK(cudaEventRecord(_stop, _stream));
      CUDA_CHECK(cudaEventSynchronize(_stop));

      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, _start, _stop));
      return static_cast<double>(elapsed_ms);
   }

   static void synchronize_device() {
      CUDA_CHECK(cudaDeviceSynchronize());
   }

private:
   cudaStream_t _stream = nullptr;
   cudaEvent_t _start = nullptr;
   cudaEvent_t _stop = nullptr;
};

inline void throw_if_error(MSVDStatus status, const char* operation) {
   if (status != MSVDStatus::kSuccess) {
      throw std::runtime_error(
         std::string(operation) + " failed: " + GetStatusString(status)
      );
   }
}

} // namespace msvd::examples
