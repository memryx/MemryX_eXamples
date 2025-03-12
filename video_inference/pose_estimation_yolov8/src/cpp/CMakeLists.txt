cmake_minimum_required(VERSION 3.10)

# Get the name of the application from the current source directory
get_filename_component(app_name poseEstimation NAME)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for OpenCV and MemryX API
include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${MX_API_DIR}/include             
                    )

# Gather all source files in the current directory                    
file(GLOB local_src
    "*.cpp"
	)

# Create the executable for the application
add_executable(${app_name} ${local_src})

# Link the mx_accl library and OpenCV libraries
target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} ) 

# Copy DFP file to the binary directory during the build
file(COPY "../../models/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})