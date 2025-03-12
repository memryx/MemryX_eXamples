cmake_minimum_required(VERSION 3.10)

# Get the name of the application from the current source directory
get_filename_component(app_name face_emotion_classification NAME)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to Libtorch
set(LIBTORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

# Check if libtorch exists
if(NOT EXISTS ${LIBTORCH_PATH})
    message(WARNING "Skipping ${app_name} application: libtorch is not found. Please ensure the required file is available. Refer to the README for more details.")
else()
    # Add libtorch to CMAKE_PREFIX_PATH to find Torch properly
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Torch package
find_package(Torch REQUIRED)

# Include directories for OpenCV, Torch, and MemryX API (assuming MX_API_DIR is defined elsewhere)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${MX_API_DIR}/include  # Assuming this is defined in the environment or before
)

# Find all source files in the current directory
file(GLOB local_src
    "*.cpp"
    "*.h"
)

# Add executable for the application
add_executable(${app_name} ${local_src})

# Link the necessary libraries (OpenCV, Torch, MemryX)
target_link_libraries(${app_name}
    mx_accl
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# Set additional C++ flags (TORCH_CXX_FLAGS should already be set)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Copy the assets directory to the binary directory
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../../models/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/models)
