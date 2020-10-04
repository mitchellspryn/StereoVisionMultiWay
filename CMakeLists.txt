cmake_minimum_required(VERSION 2.8.3)
project(StereoVisionMultiWay LANGUAGES CXX CUDA)

add_compile_options(-std=c++14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(GenerateDisparityVisualization 
    src/GenerateDisparityVisualization.cpp
    src/CudaFunctions.cu
    src/CudaDisparityMapGenerator.cpp
    src/DisparityMapGeneratorFactory.cpp
    src/OpenMpThreadedDisparityMapGenerator.cpp
    src/SingleThreadedDisparityMapGenerator.cpp)

target_link_libraries(GenerateDisparityVisualization
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARY_DIRS}
)
