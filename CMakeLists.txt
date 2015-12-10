cmake_minimum_required(VERSION 2.8)
project(Lyapunov CXX)

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
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DTESTING -DDEBUGGING -std=c++11")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -DTESTING -DDEBUGGING")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")
elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -mkl -DTESTING")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -mkl -DTESTING")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -mkl -DTESTING")
  endif()

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -mkl -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g -mkl -DTESTING -DDEBUGGING")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -mkl -DTESTING -DDEBUGGING")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -mkl -DTESTING -DDEBUGGING")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")

  add_definitions(-DUSE_MKL)
endif()

set(SOURCES
  src/SlicotWrapper.cpp
  src/Timer.cpp
  )

set(TEST_SOURCES
  test/ScalarWrapper_test.cpp
  test/SlicotWrapper_test.cpp
  test/LyapunovSolver_test.cpp
  )

# Manually add headers without source file
set(HEADERS ${HEADERS}
  src/LyapunovSolver.hpp
  src/LyapunovSolverDecl.hpp
  src/SlicotWrapper.hpp
  src/ScalarWrapper.hpp
  )

if (DEFINED ENV{TRILINOS_HOME})
  set(Trilinos_DIR "$ENV{TRILINOS_HOME}/lib/cmake/Trilinos")
endif()
find_package(Trilinos)

if (${Trilinos_FOUND})
  list(APPEND SOURCES
    src/EpetraWrapper.cpp
    )

  list(APPEND TEST_SOURCES
    test/EpetraWrapper_test.cpp
    test/LyapunovSolverEpetra_test.cpp
    )

  list(APPEND HEADERS
    src/EpetraWrapper.hpp
    )

  include_directories(${Trilinos_INCLUDE_DIRS})
  include_directories(${Trilinos_TPL_INCLUDE_DIRS})

  link_directories(${Trilinos_LIBRARY_DIRS})
  link_directories(${Trilinos_TPL_LIBRARY_DIRS})
endif()

include_directories(${PROJECT_SOURCE_DIR})

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(tests ${TEST_SOURCES} ${SOURCES})
target_link_libraries(tests ${GTEST_BOTH_LIBRARIES})
target_link_libraries(tests "slicot")

if (${Trilinos_FOUND})
  target_link_libraries(tests ${Trilinos_LIBRARIES})
  target_link_libraries(tests ${Trilinos_TPL_LIBRARIES})

  add_test(EpetraWrapperTest tests)
  add_test(LyapunovSolverEpetraTest tests)

  add_executable(main ${SOURCES} src/main.cpp)
  target_link_libraries(main "slicot")
  target_link_libraries(main ${Trilinos_LIBRARIES})
  target_link_libraries(main ${Trilinos_TPL_LIBRARIES})
endif()

add_test(LyapunovSolverTest tests)
add_test(ScalarWrapperTest tests)
add_test(SlicotWrapperTest tests)
