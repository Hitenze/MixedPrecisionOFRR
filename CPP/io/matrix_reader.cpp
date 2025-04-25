#include "matrix_reader.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace msvd {

CSRMatrix MatrixReader::read_csr_matrix(const std::string& prefix) {
   CSRMatrix matrix;
   
   // Read row pointers
   std::string row_ptr_file = prefix + "_row_ptr.bin";
   matrix.row_ptr = read_binary_file<int>(row_ptr_file);
   
   if (matrix.row_ptr.empty()) {
      throw std::runtime_error("Failed to read row pointers from " + row_ptr_file);
   }
   
   // Number of rows is (row_ptr.size() - 1)
   matrix.rows = matrix.row_ptr.size() - 1;
   
   // Read column indices
   std::string col_ind_file = prefix + "_col_ind.bin";
   matrix.col_ind = read_binary_file<int>(col_ind_file);
   
   if (matrix.col_ind.empty()) {
      throw std::runtime_error("Failed to read column indices from " + col_ind_file);
   }
   
   // Read values
   std::string values_file = prefix + "_values.bin";
   matrix.values = read_binary_file<double>(values_file);
   
   if (matrix.values.empty()) {
      throw std::runtime_error("Failed to read values from " + values_file);
   }
   
   // Determine number of columns by finding the maximum column index
   matrix.cols = 0;
   for (const auto& idx : matrix.col_ind) {
      matrix.cols = std::max(matrix.cols, idx + 1);
   }
   
   return matrix;
}

std::vector<double> MatrixReader::read_singular_values(const std::string& prefix) {
   std::string singular_values_file = prefix + "_singular_values.bin";
   return read_binary_file<double>(singular_values_file);
}

template<typename T>
std::vector<T> MatrixReader::read_binary_file(const std::string& filename, size_t count) {
   std::ifstream file(filename, std::ios::binary);
   
   if (!file) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return {};
   }
   
   // Get file size
   file.seekg(0, std::ios::end);
   size_t file_size = file.tellg();
   file.seekg(0, std::ios::beg);
   
   // Calculate element count
   size_t element_size = sizeof(T);
   size_t element_count = file_size / element_size;
   
   // If count is specified, use the smaller of the two
   if (count > 0) {
      element_count = std::min(element_count, count);
   }
   
   // Read data
   std::vector<T> data(element_count);
   file.read(reinterpret_cast<char*>(data.data()), element_count * element_size);
   
   if (!file) {
      std::cerr << "Error: Failed to read " << element_count << " elements from " << filename << std::endl;
      return {};
   }
   
   return data;
}

// Explicit instantiations
template std::vector<int> MatrixReader::read_binary_file<int>(const std::string&, size_t);
template std::vector<double> MatrixReader::read_binary_file<double>(const std::string&, size_t);

} // namespace msvd 