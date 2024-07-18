#include <queue>
#include <vector>
#include <iostream>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "postprocess.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "yolov8s.h"
using cv::Mat;
using std::queue;
using std::vector;

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);



rkyolov8s::rkyolov8s(char *model_name, int n)
{
    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    // 读取模型文件数据
    model_data = load_model(model_name, &model_data_size);
    // 通过模型文件初始化rknn类
    ret = rknn_init(&rkModel, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    // 
    rknn_core_mask core_mask;
    if (n == 0)
        core_mask = RKNN_NPU_CORE_0;
    else if(n == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;
    int ret = rknn_set_core_mask(rkModel, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }

    // 初始化rknn类的版本
    ret = rknn_query(rkModel, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 获取模型的输入参数
    ret = rknn_query(rkModel, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入数组
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    // 设置输出数组
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rkModel, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // 设置输入参数
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    // inputs[0].pass_through = 0;
}

rkyolov8s::~rkyolov8s()
{
    ret = rknn_destroy(rkModel);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
}

int rkyolov8s::interf()
{
    cv::Mat img;
    // 获取图像宽高
    cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;
    // printf("yolov8_width = %d, yolov8_height = %d\n", img_width, img_height);

    ///////// letterbox //////////////////
    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    cv::Size target_size(width, height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;

    if (img_width != width || img_height != height)
    {
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);
        
        inputs[0].buf = resized_img.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    // 设置rknn的输入数据
    rknn_inputs_set( rkModel,  io_num.n_input,  inputs);
    // 设置输出
    rknn_output outputs[ io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i <  io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!is_quant);    // =! 这是一种取反的赋值操作，如果is_quant是false，则outputs[i].want_float将被设置为true
        // outputs[i].want_float = 0;    /////
    }

    // 调用npu进行推演
    ret = rknn_run( rkModel, NULL);
    // 获取npu的推演输出结果
    ret = rknn_outputs_get( rkModel,  io_num.n_output, outputs, NULL);
    
    // post process
    // width是模型需要的输入宽度, img_width是图片的实际宽度
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    detect_result_group_t od_results;
    Postprocess(this, outputs, pads, box_conf_threshold, nms_threshold, scale_w, scale_h, &od_results);
    
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        detect_result *det_result = &(od_results.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        rectangle(ori_img, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
        putText(ori_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }
    
    ret = rknn_outputs_release( rkModel,  io_num.n_output, outputs);
    
    return 0;
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}


static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}