cmake_minimum_required(VERSION 2.8)
project(Lyapunov C CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GNU compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -Wall")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -std=c++11 -Wall")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -Wall")
  endif()

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -ffast-math -march=native -std=c++11")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g -DTESTING -DDEBUGGING")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DTESTING -DDEBUGGING -std=c++11 -Wall")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -DTESTING -DDEBUGGING")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -mkl -DTESTING")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -mkl -DTESTING -std=c++11 -Wall")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -mkl -DTESTING")
  endif()

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -mkl -ffast-math -march=native -std=c++11")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -mkl -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g -mkl -DTESTING -DDEBUGGING")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -mkl -DTESTING -DDEBUGGING -std=c++11 -Wall")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -mkl -DTESTING -DDEBUGGING")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")

  add_definitions(-DUSE_MKL)
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
  src/ScalarWrapper.hpp
  src/SlicotWrapper.hpp
  src/StlVector.hpp
  src/StlWrapper.hpp
  )

if (DEFINED ENV{TRILINOS_HOME})
  set(Trilinos_DIR "$ENV{TRILINOS_HOME}/lib/cmake/Trilinos")
endif()
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
endif()

include_directories(${PROJECT_SOURCE_DIR})

add_library(lyapunov ${SOURCES})
target_link_libraries(lyapunov "slicot")

if (${Trilinos_FOUND})
  target_link_libraries(lyapunov ${Trilinos_LIBRARIES})
  target_link_libraries(lyapunov ${Trilinos_TPL_LIBRARIES})

  add_executable(main src/main.cpp)
  target_link_libraries(main lyapunov)
endif()

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(Threads REQUIRED)
foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} lyapunov)
  target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(${test_name} ${CMAKE_THREAD_LIBS_INIT})
  
  add_test(${test_name} ${test_name})
endforeach()
