cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov8_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/include)
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so)
include_directories(${RKNN_API_PATH})

# opencv
# find_package(OpenCV REQUIRED)    # 自动设置了OpenCV_INCLUDE_DIRS（包含OpenCV头文件的目录）和 OpenCV_LIBS（OpenCV库文件的路径）
# include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenCV_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv)
set(OpenCV_LIBS 
  ${OpenCV_PATH}/lib/libopencv_core.so
  ${OpenCV_PATH}/lib/libopencv_highgui.so
  ${OpenCV_PATH}/lib/libopencv_imgproc.so
  ${OpenCV_PATH}/lib/libopencv_videoio.so
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so
  ${OpenCV_PATH}/lib/libopencv_core.so.407
  ${OpenCV_PATH}/lib/libopencv_highgui.so.407
  ${OpenCV_PATH}/lib/libopencv_imgproc.so.407
  ${OpenCV_PATH}/lib/libopencv_videoio.so.407
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.407
  ${OpenCV_PATH}/lib/libopencv_core.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_highgui.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_imgproc.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_videoio.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.4.7.0
)
include_directories(${OpenCV_PATH}/include/opencv4)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(LIB_ARCH aarch64)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories(${RGA_PATH}/include)
set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov8_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov8_demo
        src/main.cc
        src/postprocess.cc
        src/yolov8s.cc
)

target_link_libraries(rknn_yolov8_demo
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)


# install target and libraries
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8_demo${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov8_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
