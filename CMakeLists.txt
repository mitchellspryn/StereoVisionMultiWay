cmake_minimum_required(VERSION 2.8.3)
project(StereoVisionMultiWay)

add_compile_options(-std=c++14)

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(GenerateDisparityVisualization 
    src/GenerateDisparityVisualization.cpp
    src/CommonArgumentParser.cpp
    src/DisparityMapGeneratorFactory.cpp
    src/SingleThreadedDisparityMapGenerator.cpp)

target_link_libraries(GenerateDisparityVisualization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
