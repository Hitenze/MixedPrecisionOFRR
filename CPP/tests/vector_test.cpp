#include <gtest/gtest.h>
#include "../containers/vector.hpp"
#include <cmath>
#include <limits>

namespace msvd {

/**
 * @brief Test fixture for Vector class tests
 */
class VectorTest : public ::testing::Test {
protected:
   void SetUp() override {
      // Common setup for all tests
   }

   void TearDown() override {
      // Common cleanup for all tests
   }
};

/**
 * @brief Test Vector construction and basic properties
 */
TEST_F(VectorTest, Construction) {
   // Test host vector construction
   Vector<float> host_vec(10, Location::kHOST);
   EXPECT_EQ(host_vec.length(), 10);
   EXPECT_EQ(host_vec.location(), Location::kHOST);
   EXPECT_NE(host_vec.data(), nullptr);

   // Test device vector construction
   Vector<double> device_vec(5, Location::kDEVICE);
   EXPECT_EQ(device_vec.length(), 5);
   EXPECT_EQ(device_vec.location(), Location::kDEVICE);
   EXPECT_NE(device_vec.data(), nullptr);

   // Test default location (should be device)
   Vector<float> default_vec(3);
   EXPECT_EQ(default_vec.location(), Location::kDEVICE);
}

/**
 * @brief Test fill functionality
 */
TEST_F(VectorTest, Fill) {
   // Test host vector fill
   Vector<float> host_vec(5, Location::kHOST);
   host_vec.fill(3.14f);
   
   for (size_t i = 0; i < host_vec.length(); i++) {
      EXPECT_FLOAT_EQ(host_vec[i], 3.14f);
   }

   // Test device vector fill
   Vector<double> device_vec(10, Location::kDEVICE);
   device_vec.fill(2.718);
   
   // Move to host to verify
   device_vec.to_host();
   for (size_t i = 0; i < device_vec.length(); i++) {
      EXPECT_DOUBLE_EQ(device_vec[i], 2.718);
   }

   // Test half precision fill
   Vector<__half> half_vec(5, Location::kDEVICE);
   half_vec.fill(__float2half(1.0f));
   half_vec.to_host();
   for (size_t i = 0; i < half_vec.length(); i++) {
      EXPECT_NEAR(__half2float(half_vec[i]), 1.0f, 0.01);
   }
}

/**
 * @brief Test memory location transfers
 */
TEST_F(VectorTest, MemoryTransfer) {
   // Test host to device
   Vector<float> vec(5, Location::kHOST);
   vec.fill(1.5f);
   vec.to_device();
   
   EXPECT_EQ(vec.location(), Location::kDEVICE);
   
   // Test device to host
   vec.to_host();
   EXPECT_EQ(vec.location(), Location::kHOST);
   
   // Verify data was preserved
   for (size_t i = 0; i < vec.length(); i++) {
      EXPECT_FLOAT_EQ(vec[i], 1.5f);
   }
   
   // Test no-op transfers
   Vector<__half> host_vec(3, Location::kHOST);
   host_vec.to_host(); // Should be a no-op
   EXPECT_EQ(host_vec.location(), Location::kHOST);
   
   Vector<__half> device_vec(3, Location::kDEVICE);
   device_vec.to_device(); // Should be a no-op
   EXPECT_EQ(device_vec.location(), Location::kDEVICE);
}

/**
 * @brief Test random filling
 */
TEST_F(VectorTest, FillRandom) {
   // Test host random fill
   Vector<float> host_vec(100, Location::kHOST);
   host_vec.fill_random(42); // Use fixed seed for reproducibility
   
   // Check range [-1, 1]
   for (size_t i = 0; i < host_vec.length(); i++) {
      EXPECT_GE(host_vec[i], -1.0f);
      EXPECT_LE(host_vec[i], 1.0f);
   }
   
   // Check not all values are the same
   bool all_same = true;
   float first_val = host_vec[0];
   for (size_t i = 1; i < host_vec.length(); i++) {
      if (host_vec[i] != first_val) {
         all_same = false;
         break;
      }
   }
   EXPECT_FALSE(all_same);
   
   // Test device random fill
   Vector<double> device_vec(100, Location::kDEVICE);
   device_vec.fill_random(42);
   device_vec.to_host();
   
   // Check range [-1, 1] for device random values
   for (size_t i = 0; i < device_vec.length(); i++) {
      EXPECT_GE(device_vec[i], -1.0);
      EXPECT_LE(device_vec[i], 1.0);
   }
}

/**
 * @brief Test type casting
 */
TEST_F(VectorTest, TypeCasting) {
   // Test double -> float on host
   Vector<double> double_vec(3, Location::kHOST);
   double_vec.fill(1.23456789);
   
   Vector<float> float_vec = double_vec.cast<float>();
   EXPECT_EQ(float_vec.length(), double_vec.length());
   EXPECT_EQ(float_vec.location(), double_vec.location());
   
   for (size_t i = 0; i < float_vec.length(); i++) {
      EXPECT_NEAR(float_vec[i], static_cast<float>(double_vec[i]), 1e-5);
   }
   
   // Test float -> half -> float round trip on device
   Vector<float> device_float(5, Location::kDEVICE);
   device_float.fill(3.14159f);
   
   Vector<__half> device_half = device_float.cast<__half>();
   Vector<float> round_trip = device_half.cast<float>();
   
   device_float.to_host();
   round_trip.to_host();
   
   for (size_t i = 0; i < device_float.length(); i++) {
      // Half precision loses some accuracy, so use a larger epsilon
      EXPECT_NEAR(round_trip[i], device_float[i], 1e-2);
   }
}

/**
 * @brief Test operator[] access
 */
TEST_F(VectorTest, IndexOperator) {
   Vector<float> host_vec(5, Location::kHOST);
   
   // Test writing through operator[]
   for (size_t i = 0; i < host_vec.length(); i++) {
      host_vec[i] = static_cast<float>(i * 10);
   }
   
   // Test reading through operator[]
   for (size_t i = 0; i < host_vec.length(); i++) {
      EXPECT_FLOAT_EQ(host_vec[i], static_cast<float>(i * 10));
   }
}

/**
 * @brief Test all supported precision types
 */
TEST_F(VectorTest, PrecisionTypes) {
   // Test double precision
   Vector<double> double_vec(2, Location::kHOST);
   double_vec[0] = 1.0;
   double_vec[1] = 2.0;
   EXPECT_DOUBLE_EQ(double_vec[0], 1.0);
   EXPECT_DOUBLE_EQ(double_vec[1], 2.0);
   
   // Test single precision
   Vector<float> float_vec(2, Location::kHOST);
   float_vec[0] = 1.0f;
   float_vec[1] = 2.0f;
   EXPECT_FLOAT_EQ(float_vec[0], 1.0f);
   EXPECT_FLOAT_EQ(float_vec[1], 2.0f);
   
   // Test half precision
   Vector<__half> half_vec(2, Location::kHOST);
   half_vec[0] = __float2half(1.0f);
   half_vec[1] = __float2half(2.0f);
   EXPECT_NEAR(__half2float(half_vec[0]), 1.0f, 0.01);
   EXPECT_NEAR(__half2float(half_vec[1]), 2.0f, 0.01);
}

/**
 * @brief Test scaling a vector on host and device
 */
TEST_F(VectorTest, Scale) {
   // Test host vector
   Vector<double> host_vec(5, Location::kHOST);
   for (int i = 0; i < 5; i++) {
      host_vec[i] = i + 1.0;  // 1, 2, 3, 4, 5
   }
   
   // Scale by 2.0
   host_vec.scale(2.0);
   
   // Verify scaling
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(host_vec[i], (i + 1.0) * 2.0);  // 2, 4, 6, 8, 10
   }
   
   // Test device vector
   Vector<double> device_vec(5, Location::kHOST);
   for (int i = 0; i < 5; i++) {
      device_vec[i] = i + 1.0;  // 1, 2, 3, 4, 5
   }
   device_vec.to_device();
   
   // Scale by 3.0
   device_vec.scale(3.0);
   
   // Transfer back to host for verification
   device_vec.to_host();
   
   // Verify scaling
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(device_vec[i], (i + 1.0) * 3.0);  // 3, 6, 9, 12, 15
   }
   
   // Test float vector
   Vector<float> float_vec(5, Location::kHOST);
   for (int i = 0; i < 5; i++) {
      float_vec[i] = static_cast<float>(i + 1.0);  // 1, 2, 3, 4, 5
   }
   
   // Scale by 0.5
   float_vec.scale(0.5f);
   
   // Verify scaling
   for (int i = 0; i < 5; i++) {
      EXPECT_FLOAT_EQ(float_vec[i], (i + 1.0f) * 0.5f);  // 0.5, 1.0, 1.5, 2.0, 2.5
   }
   
   // Test __half vector (only on device)
   Vector<__half> half_vec(5, Location::kHOST);
   for (int i = 0; i < 5; i++) {
      half_vec[i] = __float2half(static_cast<float>(i + 1.0));  // 1, 2, 3, 4, 5
   }
   half_vec.to_device();
   
   // Scale by 2.0
   half_vec.scale(__float2half(2.0f));
   
   // Transfer back to host for verification
   half_vec.to_host();
   
   // Verify scaling (with tolerance for half precision)
   for (int i = 0; i < 5; i++) {
      EXPECT_NEAR(__half2float(half_vec[i]), (i + 1.0f) * 2.0f, 0.01f);
   }
}

/**
 * @brief Test Vector copy and move semantics
 */
TEST_F(VectorTest, CopyMoveSemantics) {
   // Create source vectors
   Vector<double> host_vec(5, Location::kHOST);
   for (int i = 0; i < 5; i++) {
      host_vec[i] = i + 1.0;  // 1, 2, 3, 4, 5
   }
   
   Vector<double> device_vec(5, Location::kHOST);
   for (int i = 0; i < 5; i++) {
      device_vec[i] = i + 6.0;  // 6, 7, 8, 9, 10
   }
   device_vec.to_device();
   
   // Test copy constructor (host)
   Vector<double> copied_host_vec(host_vec);
   EXPECT_EQ(copied_host_vec.location(), Location::kHOST);
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(copied_host_vec[i], i + 1.0);
   }
   
   // Test copy constructor (device)
   Vector<double> copied_device_vec(device_vec);
   EXPECT_EQ(copied_device_vec.location(), Location::kDEVICE);
   copied_device_vec.to_host();
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(copied_device_vec[i], i + 6.0);
   }
   
   // Test copy assignment (host)
   Vector<double> assign_host_vec(3, Location::kHOST);
   assign_host_vec = host_vec;
   EXPECT_EQ(assign_host_vec.length(), 5);
   EXPECT_EQ(assign_host_vec.location(), Location::kHOST);
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(assign_host_vec[i], i + 1.0);
   }
   
   // Test copy assignment (device)
   Vector<double> assign_device_vec(3, Location::kDEVICE);
   assign_device_vec = device_vec;
   EXPECT_EQ(assign_device_vec.length(), 5);
   EXPECT_EQ(assign_device_vec.location(), Location::kDEVICE);
   assign_device_vec.to_host();
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(assign_device_vec[i], i + 6.0);
   }
   
   // Test move constructor (host)
   Vector<double> move_host_vec(std::move(copied_host_vec));
   EXPECT_EQ(move_host_vec.location(), Location::kHOST);
   EXPECT_EQ(move_host_vec.length(), 5);
   EXPECT_EQ(copied_host_vec.length(), 0);  // Source vector should be empty
   EXPECT_EQ(copied_host_vec.data(), nullptr);  // Source data should be null
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(move_host_vec[i], i + 1.0);
   }
   
   // Test move constructor (device)
   Vector<double> move_device_vec(std::move(copied_device_vec));
   // After moving a device vector, check only data was moved
   EXPECT_EQ(move_device_vec.length(), 5);
   EXPECT_EQ(copied_device_vec.length(), 0);  // Source vector should be empty
   EXPECT_EQ(copied_device_vec.data(), nullptr);  // Source data should be null
   // We can only test data after moving to host
   move_device_vec.to_host();
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(move_device_vec[i], i + 6.0);
   }
   
   // Test move assignment (host)
   Vector<double> move_assign_host_vec(10, Location::kHOST);
   move_assign_host_vec = std::move(move_host_vec);
   EXPECT_EQ(move_assign_host_vec.length(), 5);
   EXPECT_EQ(move_assign_host_vec.location(), Location::kHOST);
   EXPECT_EQ(move_host_vec.length(), 0);  // Source vector should be empty
   EXPECT_EQ(move_host_vec.data(), nullptr);  // Source data should be null
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(move_assign_host_vec[i], i + 1.0);
   }
   
   // Test move assignment (device)
   Vector<double> move_assign_device_vec(10, Location::kDEVICE);
   move_assign_device_vec = std::move(move_device_vec);
   EXPECT_EQ(move_assign_device_vec.length(), 5);
   // After moving a device vector, we only check data was moved
   EXPECT_EQ(move_device_vec.length(), 0);  // Source vector should be empty
   EXPECT_EQ(move_device_vec.data(), nullptr);  // Source data should be null
   move_assign_device_vec.to_host();
   for (int i = 0; i < 5; i++) {
      EXPECT_DOUBLE_EQ(move_assign_device_vec[i], i + 6.0);
   }
}

/**
 * @brief Test referencing external data
 */
TEST_F(VectorTest, ExternalDataReference) {
   // Create host data
   float* host_data = new float[5];
   for (int i = 0; i < 5; i++) {
      host_data[i] = static_cast<float>(i);
   }
   
   // Create vector referencing the host data
   Vector<float> host_vec(host_data, 5, Location::kHOST);
   
   // Verify properties
   EXPECT_EQ(host_vec.length(), 5);
   EXPECT_EQ(host_vec.location(), Location::kHOST);
   EXPECT_EQ(host_vec.data(), host_data);
   EXPECT_FALSE(host_vec.owns_data());
   
   // Verify data access
   for (int i = 0; i < 5; i++) {
      EXPECT_FLOAT_EQ(host_vec[i], static_cast<float>(i));
   }
   
   // Test modification through vector (should modify original data)
   host_vec[2] = 99.0f;
   EXPECT_FLOAT_EQ(host_data[2], 99.0f);
   
   // Test ownership change when moving to device
   host_vec.to_device();
   EXPECT_EQ(host_vec.location(), Location::kDEVICE);
   EXPECT_TRUE(host_vec.owns_data());
   EXPECT_NE(host_vec.data(), host_data);
   
   // Move back to host and verify data
   host_vec.to_host();
   for (int i = 0; i < 5; i++) {
      if (i == 2) {
         EXPECT_FLOAT_EQ(host_vec[i], 99.0f);
      } else {
         EXPECT_FLOAT_EQ(host_vec[i], static_cast<float>(i));
      }
   }
   
   // Test fill operation (shouldn't change ownership)
   host_vec = Vector<float>(host_data, 5, Location::kHOST); // Reset to non-owning
   EXPECT_FALSE(host_vec.owns_data());
   host_vec.fill(7.0f);
   EXPECT_FALSE(host_vec.owns_data());
   for (int i = 0; i < 5; i++) {
      EXPECT_FLOAT_EQ(host_vec[i], 7.0f);
   }
   
   // Test scale operation (shouldn't change ownership)
   host_vec = Vector<float>(host_data, 5, Location::kHOST); // Reset to non-owning
   EXPECT_FALSE(host_vec.owns_data());
   host_vec.scale(2.0f);
   EXPECT_FALSE(host_vec.owns_data());
   
   // Test with device data
   float* device_data;
   cudaMalloc(&device_data, 5 * sizeof(float));
   float init_values[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
   cudaMemcpy(device_data, init_values, 5 * sizeof(float), cudaMemcpyHostToDevice);
   
   // Create vector referencing device data
   Vector<float> device_vec(device_data, 5, Location::kDEVICE);
   EXPECT_EQ(device_vec.location(), Location::kDEVICE);
   EXPECT_EQ(device_vec.data(), device_data);
   EXPECT_FALSE(device_vec.owns_data());
   
   // Test to_host (should take ownership)
   device_vec.to_host();
   EXPECT_EQ(device_vec.location(), Location::kHOST);
   EXPECT_TRUE(device_vec.owns_data());
   for (int i = 0; i < 5; i++) {
      EXPECT_FLOAT_EQ(device_vec[i], static_cast<float>(i + 1));
   }
   
   // Clean up
   delete[] host_data;
   cudaFree(device_data);
}

} // namespace msvd

int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
} 