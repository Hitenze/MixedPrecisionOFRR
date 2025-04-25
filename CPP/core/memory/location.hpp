#pragma once

namespace msvd {

/**
 * @brief Enum for data location
 * @details This enum is used to specify the location of the data in the vector \n
 *          We have HOST (kHOST) and DEVICE (kDEVICE) locations
 */
enum class Location {
   kHOST = 0,
   kDEVICE
};

} // namespace msvd 