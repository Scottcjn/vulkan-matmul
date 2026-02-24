cmake_minimum_required(VERSION 3.12)
project(vulkan_matmul_server C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# Vulkan headers: try system first, then local download (for Ubuntu 20.04)
find_package(Vulkan QUIET)
if (NOT Vulkan_FOUND)
    message(STATUS "System Vulkan not found, using ~/vulkan-1.3/include")
    set(VULKAN_INCLUDE_DIR "$ENV{HOME}/vulkan-1.3/include")
    set(VULKAN_LIBRARIES "vulkan")
else()
    set(VULKAN_INCLUDE_DIR ${Vulkan_INCLUDE_DIRS})
    set(VULKAN_LIBRARIES ${Vulkan_LIBRARIES})
endif()

message(STATUS "Vulkan include: ${VULKAN_INCLUDE_DIR}")
message(STATUS "Vulkan libs:    ${VULKAN_LIBRARIES}")

# POWER8 optimizations - each flag as separate list item
set(POWER8_FLAGS
    -mcpu=power8
    -mvsx
    -maltivec
    -O3
    -mtune=power8
    -funroll-loops
)

add_executable(vulkan_matmul_server server.cpp)

target_compile_options(vulkan_matmul_server PRIVATE
    ${POWER8_FLAGS}
)

target_include_directories(vulkan_matmul_server PRIVATE
    ${VULKAN_INCLUDE_DIR}
)

target_link_libraries(vulkan_matmul_server PRIVATE
    ${VULKAN_LIBRARIES}
)

# Install to current directory for easy use
install(TARGETS vulkan_matmul_server DESTINATION .)
