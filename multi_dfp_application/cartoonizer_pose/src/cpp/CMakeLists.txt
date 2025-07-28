cmake_minimum_required(VERSION 3.10)

# project(cartoon_pose_app)
get_filename_component(app_name cartoon_pose_app NAME)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Paths ===
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${MX_API_DIR}/include             
                    )


# === Source Files ===
file(GLOB local_src
    "*.cpp"
	)

# === Executable ===
add_executable(${app_name} ${local_src})

# === Link Libraries ===
target_link_libraries(${app_name}  mx_accl  ${OpenCV_LIBS} mxutils_gui)
target_link_libraries(${app_name} Qt5::Widgets Qt5::Core Qt5::Gui) 

# === Copy Models (optional, if needed at runtime) ===
file(COPY "../../models/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
