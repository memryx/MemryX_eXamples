cmake_minimum_required(VERSION 3.10)
project(depthEstimation)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gather all source files in the current directory
file(GLOB local_src
    "*.c"
    "*.cpp"
)

# Create the executable for the application
add_executable(${PROJECT_NAME} ${local_src})


# Include directories for OpenCV and MemryX API
if(WIN32)
    message(STATUS "Configuring for Windows")

    # MemryX SDK include path for Windows
    set(MEMRYX_INCLUDE_DIR "C:/Program Files/MemryX/include")

    # Find OpenCV package
    set(OpenCV_DIR  "C:/OpenCV/opencv/build/x64/vc16/lib")
    find_package(OpenCV REQUIRED)

    include_directories(
        "${MEMRYX_INCLUDE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
    )

    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
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
    target_link_libraries(${PROJECT_NAME}
        ${MXACCL_DYNAMIC}
        ${MEMX_LIB}
        ${OpenCV_LIBS}
    )


elseif(UNIX)
    message(STATUS "Configuring for Linux")

    find_package(OpenCV REQUIRED)

    set(MEMRYX_INCLUDE_DIR "/usr/local/include")

    include_directories(
        "${MEMRYX_INCLUDE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
    )

    # Link the mx_accl library and OpenCV libraries
    target_link_libraries(${PROJECT_NAME} mx_accl ${OpenCV_LIBS})

endif()


# Copy DFP file to the binary directory during the build
file(COPY "../../models/midas_v2_small.dfp" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
