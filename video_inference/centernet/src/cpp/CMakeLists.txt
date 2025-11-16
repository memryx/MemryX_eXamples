cmake_minimum_required(VERSION 3.10)

project(CenterNet)

get_filename_component(app_name CenterNet NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${MX_API_DIR}/include
                    )

file(GLOB local_src
    "*.c"
    "*.cpp"
	)

add_executable(${app_name} ${local_src})

target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} mxutils_gui)
target_link_libraries(${app_name} Qt5::Widgets Qt5::Core Qt5::Gui)

file(COPY "../../assets/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# Copy DFP files to the binary directory during the build
file(COPY "../../models/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
