# Download BuildExternalProject Module

file(DOWNLOAD
  https://raw.githubusercontent.com/Sbte/BuildExternalProject/master/BuildExternalProject.cmake
  ${CMAKE_BINARY_DIR}/external/cmake/BuildExternalProject.cmake TIMEOUT 5)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}/external/cmake" PARENT_SCOPE)
