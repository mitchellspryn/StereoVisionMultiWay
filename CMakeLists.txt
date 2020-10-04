cmake_minimum_required(VERSION 2.8.3)
project(StereoVisionMultiWay)

add_compile_options(-std=c++14)
add_compile_options(-O3)
#add_compile_options(-O0)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(GenerateDisparityVisualization 
    src/GenerateDisparityVisualization.cpp
    src/CommonArgumentParser.cpp
    src/DisparityMapGeneratorFactory.cpp
    src/OpenMpThreadedDisparityMapGenerator.cpp
    src/SingleThreadedDisparityMapGenerator.cpp)

target_link_libraries(GenerateDisparityVisualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  OpenMP::OpenMP_CXX
)
