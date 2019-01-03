cmake_minimum_required(VERSION 2.8)
project(main)

set(QMAKE_CXXFLAGS "-std=c++11")
include_directories(C:/Users/HP/Downloads/OpenBLAS-v0.2.8-x86-Win/OpenBLAS-v0.2.8-x86-Win/include)
link_directories(C:/Users/HP/Downloads/OpenBLAS-v0.2.8-x86-Win/OpenBLAS-v0.2.8-x86-Win/lib)

find_package( OpenCV REQUIRED)

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${DIR_SRCS})
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( main openblas)
target_link_libraries( main pthread )
