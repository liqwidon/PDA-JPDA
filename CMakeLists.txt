cmake_minimum_required(VERSION 3.5)

project(PDA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN3_INCLUDE_DIR "C:/programming/eigen/eigen-3.4.0/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_PREFIX_PATH "C:\\local\\boost_installed")
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(SOURCES
    main.cpp)
set(HEADERS
    mark.h
    PDA.h)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

install(TARGETS PDA
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
