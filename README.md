# yolov8s模型（RK官方）
* 按照RKNN官方的方法，使用rk的yolov8代码，导出onnx模型，然后再转成RKNN模型；
* RK官方yolov8源码[yolov8s](https://github.com/airockchip/ultralytics_yolov8)
* [多线程](https://github.com/leafqycc/rknn-cpp-Multithreading/tree/main)


# 演示说明
* 模型部署的板端是rk3588
* 系统需安装有**OpenCV**
* 可切换至root用户运行performance.sh定频提高性能和稳定性
* 运行./build-linux_RK3588.sh进行编译
* 编译完成后，将install文件夹通过adb push命令推送到板端
* 进入install/rknn_yolov8_demo文件夹后，运行命令：./rknn_yolov8_demo 模型路径 视频路径/摄像头序号




# Acknowledgements
* https://github.com/airockchip/rknn-toolkit2
* https://github.com/senlinzhan/dpool
* https://github.com/airockchip/ultralytics_yolov8
* https://github.com/airockchip/rknn_model_zoo
