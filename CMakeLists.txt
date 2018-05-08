cmake_minimum_required(VERSION 2.8)
project(RAILS C CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using GNU compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -std=c++11 -Wall")
  endif()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native -std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DTESTING -DDEBUGGING -std=c++11 -Wall")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -mkl -DTESTING -std=c++11 -Wall")
  endif()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mkl -ffast-math -march=native -std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mkl -DTESTING -DDEBUGGING -std=c++11 -Wall")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
endif()

set(SOURCES
  src/LapackWrapper
  src/SlicotWrapper
  src/StlVector
  src/StlWrapper
  src/Timer
  )

set(TEST_SOURCES
  test/GenericDenseMatrixWrapper_test
  test/GenericMultiVectorWrapper_test
  test/GenericOperatorWrapper_test
  test/LyapunovSolver_test
  test/ScalarWrapper_test
  test/SlicotWrapper_test
  )

# Manually add headers without source file
set(HEADERS ${HEADERS}
  src/BlasWrapper.hpp
  src/LapackWrapper.hpp
  src/LyapunovSolver.hpp
  src/LyapunovSolverDecl.hpp
  src/MatrixOrMultiVectorWrapper.hpp
  src/ScalarWrapper.hpp
  src/SlicotWrapper.hpp
  src/StlTools.hpp
  src/StlVector.hpp
  src/StlWrapper.hpp
  src/Timer.hpp
  )

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_package(Trilinos QUIET)

if (${Trilinos_FOUND})
  list(APPEND SOURCES
    src/Epetra_MultiVectorWrapper.cpp
    src/Epetra_OperatorWrapper.cpp
    src/Epetra_SerialDenseMatrixWrapper.cpp
    src/SchurOperator.cpp
    )

  list(APPEND TEST_SOURCES
    test/Epetra_MultiVectorWrapper_test
    test/Epetra_OperatorWrapper_test
    test/LyapunovSolverEpetra_test
    )

  list(APPEND HEADERS
    src/Epetra_MultiVectorWrapper.hpp
    src/Epetra_OperatorWrapper.hpp
    src/Epetra_SerialDenseMatrixWrapper.hpp
    src/SchurOperator.hpp
    )

  include_directories(${Trilinos_INCLUDE_DIRS})
  include_directories(${Trilinos_TPL_INCLUDE_DIRS})

  link_directories(${Trilinos_LIBRARY_DIRS})
  link_directories(${Trilinos_TPL_LIBRARY_DIRS})

  add_definitions(-DENABLE_TRILINOS)

  find_package(MPI)

  include_directories(${MPI_CXX_INCLUDE_DIRS})
  link_directories(${MPI_CXX_LIBRARY_DIRS})
endif()

include_directories(${PROJECT_SOURCE_DIR})

set(BUILD_SHARED_LIBS ON)

add_library(rails ${SOURCES})
target_link_libraries(rails "slicot")
target_link_libraries(rails ${BLAS_LIBRARIES})
target_link_libraries(rails ${LAPACK_LIBRARIES})

if (${Trilinos_FOUND})
  target_link_libraries(rails ${Trilinos_LIBRARIES})
  target_link_libraries(rails ${Trilinos_TPL_LIBRARIES})
  target_link_libraries(rails ${MPI_CXX_LIBRARIES})

  add_executable(main src/main.cpp)
  target_link_libraries(main rails)
endif()

########## Install ##########

set(INCLUDE_INSTALL_DIR include/RAILS)
set(LIB_INSTALL_DIR lib)
set(BIN_INSTALL_DIR bin)

# Install libraries
set(library_list)

install(TARGETS rails EXPORT RAILSTargets LIBRARY DESTINATION ${LIB_INSTALL_DIR})
list(APPEND library_list rails)

# Install headers
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/RAILS/RAILSConfigVersion.cmake"
  VERSION 0 COMPATIBILITY SameMajorVersion
  )

set(ConfigPackageLocation lib/cmake/RAILS)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/RAILSConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/RAILSConfig.cmake" INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS INCLUDE_INSTALL_DIR)

install(EXPORT RAILSTargets FILE RAILSTargets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES "${PROJECT_BINARY_DIR}/RAILSConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/RAILS/RAILSConfigVersion.cmake"
  DESTINATION ${ConfigPackageLocation}
  )

########## Tests ############

INCLUDE(ExternalProject)

SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIRS ${source_dir}/googletest/include)

ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/googlemock/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_MAIN_LIBRARY_PATH ${binary_dir}/googlemock/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)
set(GTEST_LIBRARY gtest)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_BOTH_LIBRARIES gtest gtest_main)

add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION
  ${GTEST_LIBRARY_PATH})
add_dependencies(${GTEST_LIBRARY} googletest)

add_library(${GTEST_MAIN_LIBRARY} UNKNOWN IMPORTED)
set_property(TARGET ${GTEST_MAIN_LIBRARY} PROPERTY IMPORTED_LOCATION
  ${GTEST_MAIN_LIBRARY_PATH})
add_dependencies(${GTEST_MAIN_LIBRARY} googletest)

enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})

set(test_list)

find_package(Threads REQUIRED)
foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} rails)
  target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(${test_name} ${CMAKE_THREAD_LIBS_INIT})

  add_test(${test_name} ${test_name})
  list(APPEND test_list ${test_name})
endforeach()

set(test_targets ${test_list})

ExternalProject_Add(
  moxunit
  GIT_REPOSITORY https://github.com/Sbte/MOxUnit.git
  CONFIGURE_COMMAND ""
  PREFIX ""
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON)

find_package(Matlab COMPONENTS MAIN_PROGRAM)

find_program(OCTAVE octave)

ExternalProject_Get_Property(moxunit source_dir)
set(MATLAB_TEST_COMMAND "addpath('${source_dir}/MOxUnit'); \
moxunit_set_path; \
cd ${PROJECT_SOURCE_DIR}/matlab; \
success = moxunit_runtests('test', '-verbose'); \
exit(~success);")

if (Matlab_FOUND)
  add_test(NAME matlab_test COMMAND ${Matlab_MAIN_PROGRAM} "\
-nojvm -r \"${MATLAB_TEST_COMMAND}\"")
  list(APPEND test_list matlab_test)
endif()

if (OCTAVE)
  add_test(NAME octave_test COMMAND ${OCTAVE} --no-gui --eval "\
addpath('${CMAKE_CURRENT_BINARY_DIR}'); \
mex ${PROJECT_SOURCE_DIR}/matlab/mex/lyap.c -lslicot; \
${MATLAB_TEST_COMMAND}")
  list(APPEND test_list octave_test)
endif()

set_tests_properties(${test_list} PROPERTIES TIMEOUT 60)

# Use make check to both build tests and test
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --verbose
  DEPENDS ${test_targets})
