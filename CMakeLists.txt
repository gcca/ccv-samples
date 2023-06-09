cmake_minimum_required(VERSION 3.25)
project(ccv-samples VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)

add_subdirectory(8)
