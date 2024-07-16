# yolov8pose模型（CSDN博主）
* yolov8pose的模型参考csdn博主的方法，将yolov8pose.pt模型导出为onnx模型；
* 然后按照瑞芯微模型转换的方法，将onnx模型转为RKNN模型；
* 参考链接方法：https://blog.csdn.net/zhangqian_1/article/details/133267470
* https://github.com/cqu20160901/yolov8pose_rknn_Cplusplus


# rtmpose模型
* 配置mmpose的环境；
* 按照mmpose模型导出的方法，导出onnx模型文件，再按照瑞芯微模型转换的方法，将onnx模型转换为RKNN模型；
* https://mmpose.readthedocs.io/zh-cn/dev-1.x/user_guides/how_to_deploy.html


# yolov8s模型（CSDN博主）
* yolov8s的模型参考csdn博主的方法，将yolov8s.pt模型导出为onnx模型；
* 然后按照瑞芯微官方的方法，将onnx模型转为RKNN模型；
* 参考链接方法：https://blog.csdn.net/zhangqian_1/article/details/128918268
* https://github.com/cqu20160901/yolov8n_onnx_tensorRT_rknn_horizon



# yolov8s模型（RK官方）
* 按照RKNN官方的方法，使用rk的yolov8代码，导出onnx模型，然后再转成RKNN模型；
* RK的yolov8源码链接：https://github.com/airockchip/ultralytics_yolov8

# 简介
* 此仓库为c++实现, 大体改自[rknpu2](https://github.com/rockchip-linux/rknpu2), python快速部署见于[rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)
* 使用[线程池](https://github.com/senlinzhan/dpool)异步操作rknn模型, 提高rk3588/rk3588s的NPU使用率, 进而提高推理帧数
* [yolov5s](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)使用relu激活函数进行优化,提高推理帧率

# 更新说明
* 修复了cmake找不到pthread的问题
* 新建nosigmoid分支,使用[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models)下的模型以达到极限性能提升
* 将RK3588 NPU SDK 更新至官方主线1.5.0, [yolov5s-silu](https://github.com/rockchip-linux/rknn-toolkit2/tree/v1.4.0/examples/onnx/yolov5)将沿用1.4.0的旧版本模型, [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)更新至1.5.0版本, 弃用nosigmoid分支。

# 使用说明
### 演示
  * 系统需安装有**OpenCV**
  * 下载Releases中的测试视频于项目根目录,运行build-linux_RK3588.sh
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov8_demo **模型所在路径** **视频所在路径/摄像头序号**



# Acknowledgements
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo
* https://github.com/rockchip-linux/rknn-toolkit2
