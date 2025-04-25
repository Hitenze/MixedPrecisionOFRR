#pragma once
#include "vector.hpp"
#include "../core/memory/location.hpp"
#include <cassert>

namespace msvd {

/**
 * @brief Matrix class
 * @details A column-major matrix class that can either own its data or reference external data.
 *          Supports operations like submatrix extraction, data transfer between host and device,
 *          and type conversion.
 */
template<typename T>
class Matrix {
private:
    /**
     * @brief Vector that stores the matrix data
     */
    Vector<T> _data;
    
    /**
     * @brief Number of rows in the matrix
     */
    size_t _num_rows;
    
    /**
     * @brief Number of columns in the matrix
     */
    size_t _num_cols;
    
    /**
     * @brief Leading dimension (stride between columns)
     */
    size_t _ld;
    
    /**
     * @brief Indicates whether this matrix owns its data
     */
    bool _owns_data;

public:
    /**
     * @brief Constructor for a matrix that owns its data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param loc Location of the matrix (host or device)
     */
    Matrix(size_t rows, size_t cols, Location loc = Location::kDEVICE);
    
    /**
     * @brief Constructor for a matrix that references external data
     * @param external_data Pointer to external data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param ld Leading dimension (stride between columns)
     * @param loc Location of the data (host or device)
     */
    Matrix(T* external_data, size_t rows, size_t cols, size_t ld, Location loc);
    
    /**
     * @brief Copy constructor
     * @param other Matrix to copy from
     */
    Matrix(const Matrix& other);
    
    /**
     * @brief Move constructor
     * @param other Matrix to move from
     */
    Matrix(Matrix&& other) noexcept;
    
    /**
     * @brief Copy assignment operator
     * @param other Matrix to copy from
     * @return Reference to this matrix
     */
    Matrix& operator=(const Matrix& other);
    
    /**
     * @brief Move assignment operator
     * @param other Matrix to move from
     * @return Reference to this matrix
     */
    Matrix& operator=(Matrix&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~Matrix();
    
    /**
     * @brief Access element at position (i,j)
     * @param i Row index
     * @param j Column index
     * @return Reference to the element
     */
    T& operator()(size_t i, size_t j);
    
    /**
     * @brief Access element at position (i,j) (const version)
     * @param i Row index
     * @param j Column index
     * @return Const reference to the element
     */
    const T& operator()(size_t i, size_t j) const;
    
    /**
     * @brief Get a submatrix view
     * @param start_row Starting row index
     * @param start_col Starting column index
     * @param num_rows Number of rows in the submatrix
     * @param num_cols Number of columns in the submatrix
     * @return A matrix that references a portion of this matrix
     */
    Matrix<T> submatrix(size_t start_row, size_t start_col, size_t num_rows, size_t num_cols);
    
    /**
     * @brief Move data to the host
     */
    void to_host();
    
    /**
     * @brief Move data to the device
     */
    void to_device();
    
    /**
     * @brief Fill the matrix with a constant value
     * @param val Value to fill the matrix with
     */
    void fill(T val);
    
    /**
     * @brief Fill the matrix with random values
     * @param seed Seed for the random number generator
     */
    void fill_random(unsigned long long seed = 815);
    
    /**
     * @brief Scale all elements in the matrix
     * @param alpha Scaling factor
     * @details Multiplies all elements in the matrix by the given scalar value
     */
    void scale(T alpha);
    
    /**
     * @brief Cast the matrix to a different data type
     * @return A new matrix with the converted data type
     */
    template<typename U>
    Matrix<U> cast();
    
    /**
     * @brief Get pointer to the matrix data
     * @return Pointer to the data
     */
    T* data();
    
    /**
     * @brief Get const pointer to the matrix data
     * @return Const pointer to the data
     */
    const T* data() const;
    
    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    size_t rows() const { return _num_rows; }
    
    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    size_t cols() const { return _num_cols; }
    
    /**
     * @brief Get leading dimension
     * @return Leading dimension
     */
    size_t ld() const { return _ld; }
    
    /**
     * @brief Get the location of the matrix data
     * @return Location (host or device)
     */
    Location location() const;
};

} // namespace msvd 