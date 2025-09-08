cmake_minimum_required(VERSION 3.16)
project(yolo_optim CXX)

add_library(yolo_preprocess STATIC preprocess.cpp)
target_include_directories(yolo_preprocess PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_library(yolo_nms STATIC nms.cpp)
target_include_directories(yolo_nms PUBLIC ${CMAKE_CURRENT_LIST_DIR})
