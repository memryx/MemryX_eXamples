cmake_minimum_required(VERSION 3.10)

project(OptimizedYOLOv8)

# Get the name of the application from the current source directory
get_filename_component(app_name yolov8_pcb_defect_detection NAME)

# Gather all source files in the current directory
file(GLOB local_src
    "src/cpp/*.cpp"
    "src/cpp/utils/*.cpp"
	)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable maximum optimization CXXFLAGS
set(
  CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} -Wall -O3 -ffast-math -march=native -flto=auto -fopenmp"
)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -O3 -ffast-math -march=native -flto=auto -fopenmp"
)

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

# FFmpeg
FIND_PATH(FFMPEG_INCLUDE_DIR_AVUTIL NAMES libavutil/avutil.h)
FIND_PATH(FFMPEG_INCLUDE_DIR_AVCODEC NAMES libavcodec/avcodec.h)
FIND_PATH(FFMPEG_INCLUDE_DIR_AVFORMAT NAMES libavformat/avformat.h)
FIND_PATH(FFMPEG_INCLUDE_DIR_SWSCALE NAMES libswscale/swscale.h)

FIND_LIBRARY(FFMPEG_AVUTIL_LIBRARY NAMES avutil)
FIND_LIBRARY(FFMPEG_AVCODEC_LIBRARY NAMES avcodec)
FIND_LIBRARY(FFMPEG_AVFORMAT_LIBRARY NAMES avformat)
FIND_LIBRARY(FFMPEG_SWSCALE_LIBRARY NAMES swscale)


# Include directories for OpenCV and MemryX API
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${MX_API_DIR}/include
                    )


# Create the executable for the application
add_executable(${app_name} ${local_src})

target_link_libraries(${app_name}  mx_accl ${OpenCV_LIBS})
target_link_libraries(${app_name} Qt5::Widgets Qt5::Core Qt5::Gui Threads::Threads)
target_link_libraries(${app_name} ${FFMPEG_AVUTIL_LIBRARY} ${FFMPEG_AVCODEC_LIBRARY} ${FFMPEG_AVFORMAT_LIBRARY} ${FFMPEG_SWSCALE_LIBRARY})
target_include_directories(${app_name} PUBLIC ${FFMPEG_INCLUDE_DIR_AVUTIL} ${FFMPEG_INCLUDE_DIR_AVCODEC} ${FFMPEG_INCLUDE_DIR_AVFORMAT} ${FFMPEG_INCLUDE_DIR_SWSCALE})

# Link files to the binary directory during the build
execute_process(COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/models)
execute_process(COMMAND ln -fs ../models/ ${CMAKE_CURRENT_BINARY_DIR}/models)
execute_process(COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/assets)
execute_process(COMMAND ln -fs ../assets/ ${CMAKE_CURRENT_BINARY_DIR}/assets)
