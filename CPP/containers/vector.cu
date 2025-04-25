#include "vector.hpp"
#include "../core/utils/type_utils.hpp"
#include "../core/utils/error_handling.hpp"
#include <curand.h>
#include <random>
#include <cassert>

namespace msvd {

// Explicit instantiations
template class Vector<double>;
template class Vector<float>;
template class Vector<__half>;

// CUDA kernel for float to half conversion
__global__ void float_to_half_kernel(float* input, __half* output, size_t length) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < length) {
      output[idx] = __float2half(input[idx]);
   }
}

// CUDA kernel for scaling random values from [0,1] to [-1,1]
template <typename T>
__global__ void scale_random_kernel(T* data, size_t length) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < length) {
      // Convert from [0,1] to [-1,1]
      data[idx] = static_cast<T>(2.0 * static_cast<double>(data[idx]) - 1.0);
   }
}

// CUDA kernel for filling a vector with a constant value
template <typename T>
__global__ void fill_kernel(T* data, T value, size_t length) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < length) {
      data[idx] = value;
   }
}

// CUDA kernel for type casting
template <typename T, typename U>
__global__ void cast_kernel(T* input, U* output, size_t length) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < length) {
      if constexpr (std::is_same_v<T, double> && std::is_same_v<U, float>) {
         output[idx] = static_cast<float>(input[idx]);
      } else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, __half>) {
         output[idx] = __float2half(static_cast<float>(input[idx]));
      } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, double>) {
         output[idx] = static_cast<double>(input[idx]);
      } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half>) {
         output[idx] = __float2half(input[idx]);
      } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, float>) {
         output[idx] = __half2float(input[idx]);
      } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, double>) {
         output[idx] = static_cast<double>(__half2float(input[idx]));
      } else {
         // Same type, just copy
         output[idx] = static_cast<U>(input[idx]);
      }
   }
}

// CUDA kernel for scaling elements
template <typename T>
__global__ void scale_kernel(T* data, T alpha, size_t length) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < length) {
      data[idx] *= alpha;
   }
}

// Template specializations
template<typename T>
Vector<T>::Vector(size_t size, Location loc)
   : _data_ptr(nullptr), _length(size), _location(loc), _owns_data(true) {
   if (_location == Location::kHOST) {
      _data_ptr = new T[size];
   } else {
      CUDA_CHECK(cudaMalloc(&_data_ptr, size * sizeof(T)));
   }
}

// Constructor for referencing external data without ownership
template<typename T>
Vector<T>::Vector(T* external_data, size_t size, Location loc)
   : _data_ptr(external_data), _length(size), _location(loc), _owns_data(false) {
}

template<typename T>
Vector<T>::~Vector() {
   Free();
}

template<typename T>
void Vector<T>::Free() {
   if (_data_ptr != nullptr && _owns_data) {
      if (_location == Location::kHOST) {
         delete[] _data_ptr;
      } else {
         CUDA_CHECK(cudaFree(_data_ptr));
      }
   }
   _data_ptr = nullptr;
}

template<typename T>
void Vector<T>::to_device() {
   if (_location == Location::kDEVICE) {
      return;
   }

   // If we don't own the data, create a copy that we own
   if (!_owns_data) {
      T* device_ptr;
      CUDA_CHECK(cudaMalloc(&device_ptr, _length * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(device_ptr, _data_ptr, _length * sizeof(T), cudaMemcpyHostToDevice));
      _data_ptr = device_ptr;
      _owns_data = true;
   } else {
      // Regular host to device transfer for owned data
      T* device_ptr;
      CUDA_CHECK(cudaMalloc(&device_ptr, _length * sizeof(T)));
      CUDA_CHECK(cudaMemcpy(device_ptr, _data_ptr, _length * sizeof(T), cudaMemcpyHostToDevice));
      delete[] _data_ptr;
      _data_ptr = device_ptr;
   }
   _location = Location::kDEVICE;
}

template<typename T>
void Vector<T>::to_host() {
   if (_location == Location::kHOST) {
      return;
   }

   // If we don't own the data, create a copy that we own
   if (!_owns_data) {
      T* host_ptr = new T[_length];
      CUDA_CHECK(cudaMemcpy(host_ptr, _data_ptr, _length * sizeof(T), cudaMemcpyDeviceToHost));
      _data_ptr = host_ptr;
      _owns_data = true;
   } else {
      // Regular device to host transfer for owned data
      T* host_ptr = new T[_length];
      CUDA_CHECK(cudaMemcpy(host_ptr, _data_ptr, _length * sizeof(T), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(_data_ptr));
      _data_ptr = host_ptr;
   }
   _location = Location::kHOST;
}

template<typename T>
void Vector<T>::fill(T val) {
   if (_location == Location::kHOST) {
      for (size_t i = 0; i < _length; i++) {
         _data_ptr[i] = val;
      }
   } else {
      // Use CUDA kernel for device fill
      constexpr int BLOCK_SIZE = 256;
      int grid_size = (_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
      
      fill_kernel<<<grid_size, BLOCK_SIZE>>>(_data_ptr, val, _length);
      CUDA_CHECK(cudaDeviceSynchronize());
   }
}

template<typename T>
void Vector<T>::fill_random(unsigned long long seed) {
   if (_location == Location::kHOST) {
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<double> dist(-1.0, 1.0);
      
      for (size_t i = 0; i < _length; i++) {
         if constexpr (std::is_same_v<T, double>) {
            _data_ptr[i] = dist(rng);
         } else if constexpr (std::is_same_v<T, float>) {
            _data_ptr[i] = static_cast<float>(dist(rng));
         } else if constexpr (std::is_same_v<T, __half>) {
            _data_ptr[i] = __float2half(static_cast<float>(dist(rng)));
         }
      }
   } else {
      // Use cuRAND for device random generation
      curandGenerator_t gen;
      CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
      
      constexpr int BLOCK_SIZE = 256;
      int grid_size = (_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
      
      if constexpr (std::is_same_v<T, double>) {
         // Generate uniform [0,1)
         CURAND_CHECK(curandGenerateUniformDouble(gen, _data_ptr, _length));
         
         // Scale to [-1,1)
         scale_random_kernel<<<grid_size, BLOCK_SIZE>>>(_data_ptr, _length);
      } else if constexpr (std::is_same_v<T, float>) {
         // Generate uniform [0,1)
         CURAND_CHECK(curandGenerateUniform(gen, _data_ptr, _length));
         
         // Scale to [-1,1)
         scale_random_kernel<<<grid_size, BLOCK_SIZE>>>(_data_ptr, _length);
      } else if constexpr (std::is_same_v<T, __half>) {
         // For half precision, generate to float first, then convert
         float* temp_ptr;
         CUDA_CHECK(cudaMalloc(&temp_ptr, _length * sizeof(float)));
         
         // Generate uniform [0,1)
         CURAND_CHECK(curandGenerateUniform(gen, temp_ptr, _length));
         
         // Scale to [-1,1)
         scale_random_kernel<<<grid_size, BLOCK_SIZE>>>(temp_ptr, _length);
         
         // Convert float to half
         float_to_half_kernel<<<grid_size, BLOCK_SIZE>>>(temp_ptr, _data_ptr, _length);
         CUDA_CHECK(cudaDeviceSynchronize());
         
         CUDA_CHECK(cudaFree(temp_ptr));
      }
      
      CURAND_CHECK(curandDestroyGenerator(gen));
   }
}

template<typename T>
void Vector<T>::scale(T alpha) {
   if (_location == Location::kHOST) {
      // For host vectors, directly scale each element
      for (size_t i = 0; i < _length; i++) {
         _data_ptr[i] *= alpha;
      }
   } else {
      // Use CUDA kernel for device scaling
      constexpr int BLOCK_SIZE = 256;
      int grid_size = (_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
      
      scale_kernel<<<grid_size, BLOCK_SIZE>>>(_data_ptr, alpha, _length);
      CUDA_CHECK(cudaDeviceSynchronize());
   }
}

// Template specialization for casting
template<typename T>
template<typename U>
Vector<U> Vector<T>::cast() {
   Vector<U> result(_length, _location);
   
   if (_location == Location::kHOST) {
      for (size_t i = 0; i < _length; i++) {
         if constexpr (std::is_same_v<T, double> && std::is_same_v<U, float>) {
            result.data()[i] = static_cast<float>(_data_ptr[i]);
         } else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, __half>) {
            result.data()[i] = __float2half(static_cast<float>(_data_ptr[i]));
         } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, double>) {
            result.data()[i] = static_cast<double>(_data_ptr[i]);
         } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half>) {
            result.data()[i] = __float2half(_data_ptr[i]);
         } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, float>) {
            result.data()[i] = __half2float(_data_ptr[i]);
         } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, double>) {
            result.data()[i] = static_cast<double>(__half2float(_data_ptr[i]));
         } else {
            // Same type, just copy
            result.data()[i] = static_cast<U>(_data_ptr[i]);
         }
      }
   } else {
      // Use CUDA kernel for device casting
      constexpr int BLOCK_SIZE = 256;
      int grid_size = (_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
      
      cast_kernel<<<grid_size, BLOCK_SIZE>>>(_data_ptr, result.data(), _length);
      CUDA_CHECK(cudaDeviceSynchronize());
   }
   
   return result;
}

// Explicit instantiations for cast function
template Vector<double> Vector<double>::cast<double>();
template Vector<float> Vector<double>::cast<float>();
template Vector<__half> Vector<double>::cast<__half>();

template Vector<double> Vector<float>::cast<double>();
template Vector<float> Vector<float>::cast<float>();
template Vector<__half> Vector<float>::cast<__half>();

template Vector<double> Vector<__half>::cast<double>();
template Vector<float> Vector<__half>::cast<float>();
template Vector<__half> Vector<__half>::cast<__half>();

template<typename T>
Vector<T>::Vector(const Vector& other)
   : _data_ptr(nullptr), _length(other._length), _location(other._location), _owns_data(true) {
   // Always create a copy with ownership for copy constructor
   // Allocate memory
   if (_location == Location::kHOST) {
      _data_ptr = new T[_length];
      // Copy data
      for (size_t i = 0; i < _length; i++) {
         _data_ptr[i] = other._data_ptr[i];
      }
   } else {
      // Allocate device memory
      CUDA_CHECK(cudaMalloc(&_data_ptr, _length * sizeof(T)));
      // Copy data
      CUDA_CHECK(cudaMemcpy(_data_ptr, other._data_ptr, _length * sizeof(T), cudaMemcpyDeviceToDevice));
   }
}

template<typename T>
Vector<T>::Vector(Vector&& other) noexcept
   : _data_ptr(other._data_ptr), _length(other._length), _location(other._location), _owns_data(other._owns_data) {
   // Take ownership and nullify the source to prevent double free
   other._data_ptr = nullptr;
   other._length = 0;
   // Keep the location and _owns_data values intact to avoid corrupting enum
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector& other) {
   if (this != &other) {
      // Free existing resources
      Free();
      
      // Copy from other
      _length = other._length;
      _location = other._location;
      _owns_data = true; // Always take ownership in copy assignment
      
      // Allocate and copy memory based on location
      if (_location == Location::kHOST) {
         _data_ptr = new T[_length];
         for (size_t i = 0; i < _length; i++) {
            _data_ptr[i] = other._data_ptr[i];
         }
      } else {
         CUDA_CHECK(cudaMalloc(&_data_ptr, _length * sizeof(T)));
         CUDA_CHECK(cudaMemcpy(_data_ptr, other._data_ptr, _length * sizeof(T), cudaMemcpyDeviceToDevice));
      }
   }
   return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator=(Vector&& other) noexcept {
   if (this != &other) {
      // Free existing resources
      Free();
      
      // Move resources from other
      _data_ptr = other._data_ptr;
      _length = other._length;
      _location = other._location;
      _owns_data = other._owns_data;
      
      // Nullify the source
      other._data_ptr = nullptr;
      other._length = 0;
      // Keep the location and _owns_data values intact to avoid corrupting enum
   }
   return *this;
}

} // namespace msvd