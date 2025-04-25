#pragma once
#include "../core/memory/location.hpp"
#include "../core/utils/error_handling.hpp"
#include <cassert>

namespace msvd {

// Forward declaration for friendship
template<typename U>
class Matrix;

/**
 * @brief Vector class
 * @details This class is used to create a vector of type T \n 
 *          It supports multiple precision types (double, float, __half) and \n
 *          can be located on either host or device memory. \n
 *          The class provides memory management, data manipulation, \n
 *          and type conversion functionality.
 */
template<typename T>
class Vector {
private:
   /**
    * @brief Pointer to the data in the vector
    * @details Raw pointer to the allocated memory that stores the vector elements. \n
    *          The memory is allocated on either host or device depending on the location.
    */
   T* _data_ptr;

   /**
    * @brief Location of the vector
    * @details Indicates whether the vector data resides on host or device memory
    */
   Location _location;
   
   /**
    * @brief Length of the vector
    * @details Number of elements in the vector
    */
   size_t _length;
   
   /**
    * @brief Indicates whether this vector owns its data
    * @details If true, the vector is responsible for freeing the memory when destroyed
    */
   bool _owns_data;

public:
   /**
    * @brief Constructor
    * @param [in] size Size of the vector
    * @param [in] loc Location of the vector
    * @details Allocates memory for the vector on either host or device based on the location
    */
   Vector(size_t size, Location loc = Location::kDEVICE);

   /**
    * @brief Constructor for a vector that references external data
    * @param [in] external_data Pointer to external data
    * @param [in] size Size of the vector
    * @param [in] loc Location of the data (host or device)
    * @details Creates a vector that references external data without taking ownership
    */
   Vector(T* external_data, size_t size, Location loc);

   /**
    * @brief Copy constructor
    * @param [in] other Vector to copy from
    * @details Creates a deep copy of the data from the other vector
    */
   Vector(const Vector& other);

   /**
    * @brief Move constructor
    * @param [in] other Vector to move from
    * @details Takes ownership of the data from the other vector
    */
   Vector(Vector&& other) noexcept;

   /**
    * @brief Copy assignment operator
    * @param [in] other Vector to copy from
    * @details Copies data from the other vector, freeing any existing data
    * @return Reference to this vector
    */
   Vector& operator=(const Vector& other);

   /**
    * @brief Move assignment operator
    * @param [in] other Vector to move from
    * @details Takes ownership of the data from the other vector, freeing any existing data
    * @return Reference to this vector
    */
   Vector& operator=(Vector&& other) noexcept;

   /**
    * @brief Destructor
    * @details Automatically frees allocated memory when the vector goes out of scope
    */
   ~Vector();
   
   /**
    * @brief Move to device
    * @details Transfers the vector data from host to device memory. \n
    *          If already on device, this operation does nothing. \n
    *          If vector doesn't own data and is on host, it will create a new copy that it owns.
    */
   void to_device();

   /**
    * @brief Move to host
    * @details Transfers the vector data from device to host memory. \n
    *          If already on host, this operation does nothing. \n
    *          If vector doesn't own data and is on device, it will create a new copy that it owns.
    */
   void to_host();

   /**
    * @brief Fill the vector with a constant value
    * @param [in] val Value to fill the vector with
    * @details Sets all elements in the vector to the specified value. \n
    *          Implementation differs between host and device memory.
    */
   void fill(T val);

   /**
    * @brief Fill with random values
    * @param [in] seed Seed for the random number generator
    * @details Fills the vector with random values in the range [-1, 1] \n
    *          Uses std::mt19937_64 for host and cuRAND for device.
    */
   void fill_random(unsigned long long seed = 815);

   /**
    * @brief Scale all elements in the vector
    * @param [in] alpha Scaling factor
    * @details Multiplies all elements in the vector by the given scalar value. \n
    *          Implementation differs between host and device memory.
    */
   void scale(T alpha);

   /**
    * @brief Cast the vector to a new type
    * @details Creates a new vector with the same contents but different precision
    * @return A new vector of type U with converted values
    */
   template<typename U>
   Vector<U> cast();

   /**
    * @brief Free the vector
    * @details Releases allocated memory and resets the data pointer
    */
   void Free();
   
   /**
    * @brief Get the data pointer
    * @details Provides access to the raw data pointer for read/write operations
    * @return Pointer to the vector data
    */
   T* data() const {
      return _data_ptr;
   }
   
   /**
    * @brief Get reference to element at specified index
    * @param [in] idx Index of the element
    * @details Direct access to vector elements (only works for host memory)
    * @return Reference to the element
    */
   T& operator[](size_t idx) {
      assert(idx < _length && "Index out of bounds");
      return _data_ptr[idx];
   }
   
   /**
    * @brief Get const reference to element at specified index
    * @param [in] idx Index of the element
    * @details Direct access to vector elements (only works for host memory)
    * @return Const reference to the element
    */
   const T& operator[](size_t idx) const {
      assert(idx < _length && "Index out of bounds");
      return _data_ptr[idx];
   }
   
   /**
    * @brief Get the location of the vector
    * @return Location of the vector (HOST or DEVICE)
    */
   Location location() const {
      return _location;
   }
   
   /**
    * @brief Get the length of the vector
    * @return Number of elements in the vector
    */
   size_t length() const {
      return _length;
   }
   
   /**
    * @brief Check if the vector owns its data
    * @return True if the vector owns its data, false otherwise
    */
   bool owns_data() const {
      return _owns_data;
   }
   
   // Make Matrix a friend class to access private members
   template<typename U>
   friend class Matrix;
};

} // namespace msvd 