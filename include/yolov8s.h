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

    rknn_tensor_mem* input_mems[1];
    rknn_tensor_mem* output_mems[9];

    rknn_input inputs[1];
    rknn_output outputs[9];

    int channel;
    int width;
    int height;
    bool is_quant;
    
    int ret;
    Mat ori_img;
    int interf();
    rkyolov8s(char *dst, int n);
    ~rkyolov8s();
};

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);

#endif   // _YOLOV8S_H_