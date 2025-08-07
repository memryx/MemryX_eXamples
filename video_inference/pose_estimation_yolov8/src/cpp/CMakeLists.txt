cmake_minimum_required(VERSION 3.10)
 
# Get the name of the application from the current source directory
get_filename_component(app_name poseEstimation NAME)
project(poseEstimation)
 
# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
# Gather all source files in the current directory                    
file(GLOB local_src
    "*.cpp"
	)
 
# Create the executable for the application
add_executable(${app_name} ${local_src})
 
if(UNIX)
    message(STATUS "configuring for linux")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS} 
                        ${MX_API_DIR}/include             
                        )
    target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} ) 
    
elseif(WIN32)

    message(STATUS "configuring for windows")
    set(OpenCV_DIR "C:/OpenCV/opencv/build/x64/vc16/lib/")  # <- this must point to where OpenCVConfig.cmake is
    set(MEMRYX_INCLUDE_DIR "C:/Program Files/MemryX/include")
    set(MEMRYX_LIB_DIR "C:/Program Files/MemryX/lib")
    set(ONNXRUNTIME_DIR "C:/onnxruntime")
 
    # === Find OpenCV ===
    find_package(OpenCV REQUIRED)
 
    # === Include directories ===
    include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "${MEMRYX_INCLUDE_DIR}"
        "${ONNXRUNTIME_DIR}/include"
    )
 
    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
    endif()
 
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_DIR}/lib"
        NO_DEFAULT_PATH
    )
 
    if (NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "Cannot find onnxruntime.lib")
    endif()
 
    find_library(MXACCL_DYNAMIC
        NAMES memx-accl
        PATHS
            "C:/Program Files/MemryX/lib"
        NO_DEFAULT_PATH
    )

    if (NOT MXACCL_DYNAMIC)
        message(FATAL_ERROR "Cannot find memx-accl.lib")
    endif()
 
    find_library(MEMX_LIB
        NAMES udriver
        PATHS
            "C:/Program Files/MemryX/lib"
        NO_DEFAULT_PATH
    )

    if (NOT MEMX_LIB)
        message(FATAL_ERROR "Cannot find udriver.lib")
    endif()

    # Link the mx_accl library and OpenCV libraries
    target_link_libraries(${app_name}  
        ${MXACCL_DYNAMIC}  
        ${MEMX_LIB}  
        ${OpenCV_LIBS} 
        ${ONNXRUNTIME_LIB} 
    )
endif()
 
 
# Copy DFP file to the binary directory during the build
file(COPY "../../models/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
 