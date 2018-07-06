cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(main)

FIND_PACKAGE( OpenCV REQUIRED PATHS /usr/local/opencv320/share/OpenCV NO_DEFAULT_PATH)



add_executable (main main.cpp)
target_link_libraries (main ${OpenCV_LIBS})
