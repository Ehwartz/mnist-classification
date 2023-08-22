cmake_minimum_required(VERSION 3.21)
project(nn)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "D:\\opencv\\mingw-build")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(nn main.cpp nn.cpp nn.h Variable.h Variable.cpp Module.cpp Module.h optim.h optim.cpp dataset.h dataset.cpp train.h train.cpp stat.h stat.cpp)
link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(nn ${OpenCV_LIBS})

