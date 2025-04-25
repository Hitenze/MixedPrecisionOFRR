#include "matrix.hpp"
#include <cassert>

namespace msvd {

// Explicit instantiations
template class Matrix<double>;
template class Matrix<float>;
template class Matrix<__half>;

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, Location loc)
    : _data(rows * cols, loc),
      _num_rows(rows),
      _num_cols(cols),
      _ld(rows),
      _owns_data(true) {
}

template<typename T>
Matrix<T>::Matrix(T* external_data, size_t rows, size_t cols, size_t ld, Location loc)
    : _data(external_data, ld * cols, loc),  // Initialize with reference to external data
      _num_rows(rows),
      _num_cols(cols),
      _ld(ld),
      _owns_data(false) {
}

template<typename T>
Matrix<T>::Matrix(const Matrix& other)
    : _data(other._data),  // Uses Vector's copy constructor
      _num_rows(other._num_rows),
      _num_cols(other._num_cols),
      _ld(other._ld),
      _owns_data(other._owns_data) {
}

template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : _data(std::move(other._data)),  // Uses Vector's move constructor
      _num_rows(other._num_rows),
      _num_cols(other._num_cols),
      _ld(other._ld),
      _owns_data(other._owns_data) {
    // Clear out the moved-from object
    other._num_rows = 0;
    other._num_cols = 0;
    other._ld = 0;
    other._owns_data = false;
    // Note: We don't need to reset _data's location as Vector's move constructor handles it correctly
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this != &other) {
        _data = other._data;  // Uses Vector's copy assignment
        _num_rows = other._num_rows;
        _num_cols = other._num_cols;
        _ld = other._ld;
        _owns_data = other._owns_data;
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        _data = std::move(other._data);  // Uses Vector's move assignment
        _num_rows = other._num_rows;
        _num_cols = other._num_cols;
        _ld = other._ld;
        _owns_data = other._owns_data;
        
        // Clear out the moved-from object
        other._num_rows = 0;
        other._num_cols = 0;
        other._ld = 0;
        other._owns_data = false;
        // Note: We don't need to reset _data's location as Vector's move assignment handles it correctly
    }
    return *this;
}

template<typename T>
Matrix<T>::~Matrix() {
    // Vector destructor will handle memory cleanup
}

template<typename T>
T& Matrix<T>::operator()(size_t i, size_t j) {
    assert(i < _num_rows && j < _num_cols && "Matrix indices out of bounds");
    return _data.data()[i + j * _ld];
}

template<typename T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    assert(i < _num_rows && j < _num_cols && "Matrix indices out of bounds");
    return _data.data()[i + j * _ld];
}

template<typename T>
Matrix<T> Matrix<T>::submatrix(size_t start_row, size_t start_col, size_t num_rows, size_t num_cols) {
    assert(start_row + num_rows <= _num_rows && "Submatrix rows exceed matrix dimensions");
    assert(start_col + num_cols <= _num_cols && "Submatrix columns exceed matrix dimensions");
    
    // For now, we'll create a new matrix and copy the data
    // Later, we'll implement a view-based approach
    Matrix<T> result(num_rows, num_cols, _data.location());
    
    if (_data.location() == Location::kHOST) {
        for (size_t j = 0; j < num_cols; ++j) {
            for (size_t i = 0; i < num_rows; ++i) {
                result(i, j) = (*this)(start_row + i, start_col + j);
            }
        }
    } else {
        // For device data, we would need CUDA kernels
        // This is a placeholder - will be implemented later
        assert(false && "Device submatrix not yet implemented");
    }
    
    return result;
}

template<typename T>
void Matrix<T>::to_host() {
    _data.to_host();
}

template<typename T>
void Matrix<T>::to_device() {
    _data.to_device();
}

template<typename T>
void Matrix<T>::fill(T val) {
    _data.fill(val);
}

template<typename T>
void Matrix<T>::fill_random(unsigned long long seed) {
    _data.fill_random(seed);
}

template<typename T>
void Matrix<T>::scale(T alpha) {
    // Use Vector's scale function which handles both host and device
    _data.scale(alpha);
}

template<typename T>
template<typename U>
Matrix<U> Matrix<T>::cast() {
    Matrix<U> result(_num_rows, _num_cols, _data.location());
    
    if (_data.location() == Location::kHOST) {
        // For host memory, directly cast each element
        for (size_t j = 0; j < _num_cols; ++j) {
            for (size_t i = 0; i < _num_rows; ++i) {
                if constexpr (std::is_same_v<T, double> && std::is_same_v<U, float>) {
                    result(i, j) = static_cast<float>((*this)(i, j));
                } else if constexpr (std::is_same_v<T, double> && std::is_same_v<U, __half>) {
                    result(i, j) = __float2half(static_cast<float>((*this)(i, j)));
                } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, double>) {
                    result(i, j) = static_cast<double>((*this)(i, j));
                } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half>) {
                    result(i, j) = __float2half((*this)(i, j));
                } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, float>) {
                    result(i, j) = __half2float((*this)(i, j));
                } else if constexpr (std::is_same_v<T, __half> && std::is_same_v<U, double>) {
                    result(i, j) = static_cast<double>(__half2float((*this)(i, j)));
                } else {
                    // Same type, just copy
                    result(i, j) = static_cast<U>((*this)(i, j));
                }
            }
        }
    } else {
        // For device memory, we'll move to host, cast, then move back
        // This is inefficient, but good enough for now
        // TODO: Implement device kernel for casting
        Matrix<T> host_copy = *this;
        host_copy.to_host();
        Matrix<U> host_result = host_copy.cast<U>();
        host_result.to_device();
        return host_result;
    }
    
    return result;
}

template<typename T>
T* Matrix<T>::data() {
    return _data.data();
}

template<typename T>
const T* Matrix<T>::data() const {
    return _data.data();
}

template<typename T>
Location Matrix<T>::location() const {
    return _data.location();
}

// Explicit instantiations for cast function
template Matrix<double> Matrix<double>::cast<double>();
template Matrix<float> Matrix<double>::cast<float>();
template Matrix<__half> Matrix<double>::cast<__half>();

template Matrix<double> Matrix<float>::cast<double>();
template Matrix<float> Matrix<float>::cast<float>();
template Matrix<__half> Matrix<float>::cast<__half>();

template Matrix<double> Matrix<__half>::cast<double>();
template Matrix<float> Matrix<__half>::cast<float>();
template Matrix<__half> Matrix<__half>::cast<__half>();

} // namespace msvd 