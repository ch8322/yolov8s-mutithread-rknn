# Install script for directory: /home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/install/rknn_yolov8_demo_Linux")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE EXECUTABLE FILES "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/build/rknn_yolov8_demo")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_demo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_demo")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/./rknn_yolov8_demo")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/librknnrt.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE PROGRAM FILES
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_core.so"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_highgui.so"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_imgproc.so"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_videoio.so"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_imgcodecs.so"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_core.so.407"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_highgui.so.407"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_imgproc.so.407"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_videoio.so.407"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_imgcodecs.so.407"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_core.so.4.7.0"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_highgui.so.4.7.0"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_imgproc.so.4.7.0"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_videoio.so.4.7.0"
    "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/include/3rdparty/opencv/lib/libopencv_imgcodecs.so.4.7.0"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/./" TYPE DIRECTORY FILES "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/model")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/abc/Desktop/test/11_rknn-cpp-Multithreading-1.5.0/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
