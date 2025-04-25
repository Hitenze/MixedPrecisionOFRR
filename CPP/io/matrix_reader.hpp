#pragma once
#include <vector>
#include <string>

namespace msvd {

class CSRMatrix {
public:
   // CSR format arrays
   std::vector<int> row_ptr;
   std::vector<int> col_ind;
   std::vector<double> values;
   int rows;
   int cols;

   // Constructor
   CSRMatrix() : rows(0), cols(0) {}
};

class MatrixReader {
public:
   /**
    * @brief Read matrix from binary files in CSR format
    * @param prefix Filename prefix (e.g., "Maragal_7")
    * @return CSRMatrix object containing the matrix in CSR format
    */
   static CSRMatrix read_csr_matrix(const std::string& prefix);

   /**
    * @brief Read singular values from binary file
    * @param prefix Filename prefix (e.g., "Maragal_7")
    * @return Vector containing singular values
    */
   static std::vector<double> read_singular_values(const std::string& prefix);

private:
   /**
    * @brief Read binary file into a vector
    * @param filename Name of the binary file
    * @param count Number of elements to read (0 for all)
    * @return Vector containing the data
    */
   template<typename T>
   static std::vector<T> read_binary_file(const std::string& filename, size_t count = 0);
};

} // namespace msvd 