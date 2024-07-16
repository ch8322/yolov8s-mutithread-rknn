#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "rknn_api.h"
#include "yolov8s.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25
// #define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

int loadLabelName(const char *locationFilename, char *label[]);
char *readLine(FILE *fp, char *buffer, int *len);
int readLines(const char *fileName, char *lines[], int max_line);

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result;

typedef struct
{
    int id;
    int count;
    detect_result results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

int Postprocess(rkyolov8s* rknn_app, rknn_output *outputs, BOX_RECT pads, float conf_threshold, float nms_threshold, float scale_w, float scale_h, detect_result_group_t *od_results);

void deinitPostProcess();



#endif //_POSTPROCESS_H_
