#ifndef _YOLOV8S_H_
#define _YOLOV8S_H_

#include "rknn_api.h"
#include "opencv2/core/core.hpp"

using cv::Mat;

class rkyolov8s
{
public:
    rknn_context rkModel;
    unsigned char *model_data;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    int channel = 3;
    int width = 0;
    int height = 0;
    bool is_quant;
    
    int ret;
    Mat ori_img;
    int interf();
    rkyolov8s(char *dst, int n);
    ~rkyolov8s();
};

#endif   // _YOLOV8S_H_