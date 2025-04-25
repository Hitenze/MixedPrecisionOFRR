#include <gtest/gtest.h>
#include "../containers/matrix.hpp"
#include <cmath>

namespace msvd {

/**
 * @brief Test fixture for Matrix class tests
 */
class MatrixTest : public ::testing::Test {
protected:
   void SetUp() override {
      // Common setup for all tests
   }

   void TearDown() override {
      // Common cleanup for all tests
   }
};

/**
 * @brief Test Matrix construction and basic properties
 */
TEST_F(MatrixTest, Construction) {
   // Test host matrix construction
   Matrix<float> host_mat(5, 3, Location::kHOST);
   EXPECT_EQ(host_mat.rows(), 5);
   EXPECT_EQ(host_mat.cols(), 3);
   EXPECT_EQ(host_mat.ld(), 5);  // Leading dimension should equal rows by default
   EXPECT_EQ(host_mat.location(), Location::kHOST);
   EXPECT_NE(host_mat.data(), nullptr);

   // Test device matrix construction
   Matrix<double> device_mat(4, 2, Location::kDEVICE);
   EXPECT_EQ(device_mat.rows(), 4);
   EXPECT_EQ(device_mat.cols(), 2);
   EXPECT_EQ(device_mat.ld(), 4);  // Leading dimension should equal rows by default
   EXPECT_EQ(device_mat.location(), Location::kDEVICE);
   EXPECT_NE(device_mat.data(), nullptr);

   // Test default location (should be device)
   Matrix<float> default_mat(3, 3);
   EXPECT_EQ(default_mat.location(), Location::kDEVICE);
}

/**
 * @brief Test element access through operator()
 */
TEST_F(MatrixTest, ElementAccess) {
   Matrix<float> host_mat(3, 2, Location::kHOST);
   
   // Fill the matrix with known values
   float val = 0.0f;
   for (size_t j = 0; j < host_mat.cols(); ++j) {
      for (size_t i = 0; i < host_mat.rows(); ++i) {
         host_mat(i, j) = val;
         val += 1.0f;
      }
   }
   
   // Verify the values
   val = 0.0f;
   for (size_t j = 0; j < host_mat.cols(); ++j) {
      for (size_t i = 0; i < host_mat.rows(); ++i) {
         EXPECT_FLOAT_EQ(host_mat(i, j), val);
         val += 1.0f;
      }
   }
}

/**
 * @brief Test fill functionality
 */
TEST_F(MatrixTest, Fill) {
   // Test host matrix fill
   Matrix<float> host_mat(3, 3, Location::kHOST);
   host_mat.fill(3.14f);
   
   for (size_t j = 0; j < host_mat.cols(); ++j) {
      for (size_t i = 0; i < host_mat.rows(); ++i) {
         EXPECT_FLOAT_EQ(host_mat(i, j), 3.14f);
      }
   }

   // Test device matrix fill
   Matrix<double> device_mat(4, 2, Location::kDEVICE);
   device_mat.fill(2.718);
   
   // Move to host to verify
   device_mat.to_host();
   for (size_t j = 0; j < device_mat.cols(); ++j) {
      for (size_t i = 0; i < device_mat.rows(); ++i) {
         EXPECT_DOUBLE_EQ(device_mat(i, j), 2.718);
      }
   }

   // Test __half matrix fill
   Matrix<__half> half_mat(3, 3, Location::kDEVICE);
   half_mat.fill(__float2half(1.0f));
   half_mat.to_host();
   for (size_t j = 0; j < half_mat.cols(); ++j) {
      for (size_t i = 0; i < half_mat.rows(); ++i) {
         EXPECT_NEAR(__half2float(half_mat(i, j)), 1.0f, 0.01);
      }
   }
}

/**
 * @brief Test data transfer between host and device
 */
TEST_F(MatrixTest, DataTransfer) {
   // Create a host matrix and fill it with a pattern
   Matrix<float> mat(3, 2, Location::kHOST);
   float val = 0.0f;
   for (size_t j = 0; j < mat.cols(); ++j) {
      for (size_t i = 0; i < mat.rows(); ++i) {
         mat(i, j) = val;
         val += 1.0f;
      }
   }
   
   // Transfer to device and back
   mat.to_device();
   EXPECT_EQ(mat.location(), Location::kDEVICE);
   mat.to_host();
   EXPECT_EQ(mat.location(), Location::kHOST);
   
   // Verify data is preserved
   val = 0.0f;
   for (size_t j = 0; j < mat.cols(); ++j) {
      for (size_t i = 0; i < mat.rows(); ++i) {
         EXPECT_FLOAT_EQ(mat(i, j), val);
         val += 1.0f;
      }
   }
}

/**
 * @brief Test scaling a matrix on host and device
 */
TEST_F(MatrixTest, Scale) {
   const size_t rows = 3;
   const size_t cols = 2;
   
   // Test host matrix
   Matrix<double> host_mat(rows, cols, Location::kHOST);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         host_mat(i, j) = i + j * rows + 1.0;  // 1, 2, 3, 4, 5, 6
      }
   }
   
   // Scale by 2.0
   host_mat.scale(2.0);
   
   // Verify scaling
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(host_mat(i, j), (i + j * rows + 1.0) * 2.0);  // 2, 4, 6, 8, 10, 12
      }
   }
   
   // Test device matrix
   Matrix<double> device_mat(rows, cols, Location::kDEVICE);
   device_mat.to_host();
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         device_mat(i, j) = i + j * rows + 1.0;  // 1, 2, 3, 4, 5, 6
      }
   }
   device_mat.to_device();
   
   // Scale by 3.0
   device_mat.scale(3.0);
   
   // Transfer back to host for verification
   device_mat.to_host();
   
   // Verify scaling
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(device_mat(i, j), (i + j * rows + 1.0) * 3.0);  // 3, 6, 9, 12, 15, 18
      }
   }
   
   // Test float matrix
   Matrix<float> float_mat(rows, cols, Location::kHOST);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         float_mat(i, j) = static_cast<float>(i + j * rows + 1.0);
      }
   }
   
   // Scale by 0.5
   float_mat.scale(0.5f);
   
   // Verify scaling
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_FLOAT_EQ(float_mat(i, j), (i + j * rows + 1.0f) * 0.5f);
      }
   }
}

/**
 * @brief Test Matrix copy and move semantics
 */
TEST_F(MatrixTest, CopyMoveSemantics) {
   const size_t rows = 3;
   const size_t cols = 2;
   
   // Create source matrices
   Matrix<double> host_mat(rows, cols, Location::kHOST);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         host_mat(i, j) = i + j * rows + 1.0;  // 1, 2, 3, 4, 5, 6
      }
   }
   
   Matrix<double> device_mat(rows, cols, Location::kDEVICE);
   device_mat.to_host();
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         device_mat(i, j) = i + j * rows + 10.0;  // 10, 11, 12, 13, 14, 15
      }
   }
   device_mat.to_device();
   
   // Test copy constructor (host)
   Matrix<double> copied_host_mat(host_mat);
   EXPECT_EQ(copied_host_mat.rows(), rows);
   EXPECT_EQ(copied_host_mat.cols(), cols);
   EXPECT_EQ(copied_host_mat.location(), Location::kHOST);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(copied_host_mat(i, j), i + j * rows + 1.0);
      }
   }
   
   // Test copy constructor (device)
   Matrix<double> copied_device_mat(device_mat);
   EXPECT_EQ(copied_device_mat.rows(), rows);
   EXPECT_EQ(copied_device_mat.cols(), cols);
   EXPECT_EQ(copied_device_mat.location(), Location::kDEVICE);
   copied_device_mat.to_host();
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(copied_device_mat(i, j), i + j * rows + 10.0);
      }
   }
   
   // Test copy assignment (host)
   Matrix<double> assign_host_mat(1, 1, Location::kHOST);
   assign_host_mat = host_mat;
   EXPECT_EQ(assign_host_mat.rows(), rows);
   EXPECT_EQ(assign_host_mat.cols(), cols);
   EXPECT_EQ(assign_host_mat.location(), Location::kHOST);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(assign_host_mat(i, j), i + j * rows + 1.0);
      }
   }
   
   // Test copy assignment (device)
   Matrix<double> assign_device_mat(1, 1, Location::kDEVICE);
   assign_device_mat = device_mat;
   EXPECT_EQ(assign_device_mat.rows(), rows);
   EXPECT_EQ(assign_device_mat.cols(), cols);
   EXPECT_EQ(assign_device_mat.location(), Location::kDEVICE);
   assign_device_mat.to_host();
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(assign_device_mat(i, j), i + j * rows + 10.0);
      }
   }
   
   // Test move constructor (host)
   Matrix<double> move_host_mat(std::move(copied_host_mat));
   EXPECT_EQ(move_host_mat.rows(), rows);
   EXPECT_EQ(move_host_mat.cols(), cols);
   EXPECT_EQ(move_host_mat.location(), Location::kHOST);
   EXPECT_EQ(copied_host_mat.rows(), 0);  // Source matrix should be empty
   EXPECT_EQ(copied_host_mat.cols(), 0);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(move_host_mat(i, j), i + j * rows + 1.0);
      }
   }
   
   // Test move constructor (device)
   Matrix<double> move_device_mat(std::move(copied_device_mat));
   EXPECT_EQ(move_device_mat.rows(), rows);
   EXPECT_EQ(move_device_mat.cols(), cols);
   // After moving a device matrix, we don't check location
   EXPECT_EQ(copied_device_mat.rows(), 0);  // Source matrix should be empty
   EXPECT_EQ(copied_device_mat.cols(), 0);
   move_device_mat.to_host();
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(move_device_mat(i, j), i + j * rows + 10.0);
      }
   }
   
   // Test move assignment (host)
   Matrix<double> move_assign_host_mat(5, 5, Location::kHOST);
   move_assign_host_mat = std::move(move_host_mat);
   EXPECT_EQ(move_assign_host_mat.rows(), rows);
   EXPECT_EQ(move_assign_host_mat.cols(), cols);
   EXPECT_EQ(move_assign_host_mat.location(), Location::kHOST);
   EXPECT_EQ(move_host_mat.rows(), 0);  // Source matrix should be empty
   EXPECT_EQ(move_host_mat.cols(), 0);
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(move_assign_host_mat(i, j), i + j * rows + 1.0);
      }
   }
   
   // Test move assignment (device)
   Matrix<double> move_assign_device_mat(5, 5, Location::kDEVICE);
   move_assign_device_mat = std::move(move_device_mat);
   EXPECT_EQ(move_assign_device_mat.rows(), rows);
   EXPECT_EQ(move_assign_device_mat.cols(), cols);
   // After moving a device matrix, we don't check location
   EXPECT_EQ(move_device_mat.rows(), 0);  // Source matrix should be empty
   EXPECT_EQ(move_device_mat.cols(), 0);
   move_assign_device_mat.to_host();
   for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
         EXPECT_DOUBLE_EQ(move_assign_device_mat(i, j), i + j * rows + 10.0);
      }
   }
}

} // namespace msvd

int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
} 