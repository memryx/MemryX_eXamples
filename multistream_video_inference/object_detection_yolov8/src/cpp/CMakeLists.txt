cmake_minimum_required(VERSION 3.10)

get_filename_component(app_name multistream_objectdetection_yolov8s NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    )

file(GLOB local_src
    "*.c"
    "*.cpp"
	)

add_executable(${app_name} ${local_src})

target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} mxutils_gui)
target_link_libraries(${app_name} Qt5::Widgets Qt5::Core Qt5::Gui)

file(COPY "../../models/tflite/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
