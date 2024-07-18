#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "opencv2/imgproc.hpp"

#include <set>
#include <vector>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

int loadLabelName(const char *locationFilename, char *label[])
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

char *readLine(FILE *fp, char *buffer, int *len)
{
  int ch;
  int i = 0;
  size_t buff_len = 0;

  buffer = (char *)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF)
  {
    buff_len++;
    void *tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL)
    {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char *)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp)))
  {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char *fileName, char *lines[], int max_line)
{
  FILE *file = fopen(fileName, "r");
  char *s;
  int i = 0;
  int n = 0;

  if (file == NULL)
  {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL)
  {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for(int i = 0; i < validCount; ++i)
    {
        if(order[i] == -1 || classIds[i] != filterId)
        {
            continue;
        }
        int n = order[i];
        for(int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if(m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if(iou > threshold)   //如果IoU超过nms阈值，则将检测框标记为 -1（即移除该检测框）
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}


static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if(left < right)
    {
        key_index = indices[left];
        key = input[left];
        while(low < high)
        {
            while(low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while(low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

//这个函数的目的是将一个浮点数 f32 进行量化（Quantization）转换为一个8位整数（int8_t）
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);   //将数值裁剪到8位整数的范围内（-128 到 127）
    return res;
}

// 这个函数的作用是将一个8位整数（int8_t）进行反量化（Dequantization）转换为浮点数。
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) 
{ 
    return ((float)qnt - (float)zp) * scale; 
}

//目的是将神经网络输出的归一化值转化为最终的目标框坐标
//计算离散边界框的实际值
void compute_dfl(float* tensor, int dfl_len, float* box)
{
    for(int b=0; b<4; b++)
    {
        // 定义一个数组用于存储指数函数的值
        float exp_t[dfl_len];
        // 存储所有指数函数的和
        float exp_sum=0;
        // 存储加权平均值的累积和
        float acc_sum=0;

        // 计算指数函数并累加到 exp_sum 中
        for(int i=0; i< dfl_len; i++)
        {
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }

        // 计算加权平均值的累积和
        for(int i=0; i< dfl_len; i++)
        {
            acc_sum += exp_t[i]/exp_sum *i;
        }
        // 将加权平均值存储到结果数组 box 中
        box[b] = acc_sum;
    }
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold)
{
    // 初始化有效目标计数
    int validCount = 0;
    // 计算网格总数
    int grid_len = grid_h * grid_w;
    // 将浮点数转换为8位整数
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);   //置信度阈值
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
    
    //遍历每个网格
    for(int i = 0; i < grid_h; i++)
    {
        for(int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;   //当前网格单元在特征图中的偏移量
            int max_class_id = -1;    //用于记录当前网格单元中的最大类别ID,初始化为-1

            if(score_sum_tensor != nullptr)   
            {
                if(score_sum_tensor[offset] < score_sum_thres_i8)   // 通过score_sum_tensor进行快速过滤，如果类别分数和小于阈值，则跳过该网格
                {
                    continue;
                }
            }

            //计算最大类别分数和ID
            int8_t max_score = -score_zp;   //初始化 max_score 为最低可能的量化分数
            for(int c= 0; c< OBJ_CLASS_NUM; c++)   //遍历所有类别，找到最大类别分数和对应的类别ID
            {
                if((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))   //如果当前类别分数大于阈值并且大于当前最大分数，则更新最大分数和类别ID
                {
                    max_score = score_tensor[offset];   //当前类别的量化分数
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            //解码边界框
            if(max_score> score_thres_i8)   //如果最大类别分数大于阈值，则进行边界框的解码
            {
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];   //存储解码前的离散边界框数据
                for(int k=0; k< dfl_len*4; k++)
                {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);  //将一个8位整数（int8_t）进行反量化（Dequantization）转换为浮点数
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);   //通过计算离散边界框的加权平均值，得到边界框的实际值

                //将特征图尺寸中的坐标映射到模型尺寸上
                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int> &classId, 
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for(int i = 0; i < grid_h; i++)
    {
        for(int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if(score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for(int c= 0; c< OBJ_CLASS_NUM; c++){
                if((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if(max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

int Postprocess(rkyolov8s* rknn_app, rknn_output *outputs, BOX_RECT pads, float conf_threshold, float nms_threshold, float scale_w, float scale_h, detect_result_group_t *od_results)
{
    //初始化标签
    static int init = -1;
    if (init == -1)
    {
        int ret = 0;
        ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);  //从文件中读取标签并存储在 labels 数组中
        if (ret < 0)
        {
            return -1;
        }
        init = 0;
    }
    // 创建用于存储目标框信息的向量
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    // 记录有效目标数量
    int validCount = 0;
    // 用于存储目标检测网络的输出特征图的步长和尺寸信息
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    
    // 获取模型输入的宽度和高度信息
    int model_in_w = rknn_app->width;
    int model_in_h = rknn_app->height;
    
    // 将目标检测结果结构体的内存区域全部设置为零
    memset(od_results, 0, sizeof(detect_result_group_t));

    int dfl_len = rknn_app->output_attrs[0].dims[1] / 4;    // 64/4=16  特征图每个通道的长度
    int output_per_branch = rknn_app->io_num.n_output / 3;   // 9/3=3   每个分支的输出数
    
    // 遍历三个分支
    for(int i = 0; i < 3; i++)
    {
        // 初始化指向 score_sum 的指针为 nullptr
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        // 如果每个分支的输出数为3，则获取 score_sum 的相关信息
        if(output_per_branch == 3)
        {
            score_sum = outputs[i*output_per_branch + 2].buf;
            score_sum_zp = rknn_app->output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = rknn_app->output_attrs[i*output_per_branch + 2].scale;
        }
        // 获取当前分支的 box 和 score 的索引
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;
        // 获取当前分支的特征图尺寸和步长信息  nchw 0 1 2 3
        grid_h = rknn_app->output_attrs[box_idx].dims[2];
        grid_w = rknn_app->output_attrs[box_idx].dims[3];
        stride = model_in_h / grid_h;
    
        // 根据量化情况，选择调用相应的处理函数

        // validCount += process_i8((int8_t *)outputs[box_idx].buf, rknn_app->output_attrs[box_idx].zp, rknn_app->output_attrs[box_idx].scale,
        //                              (int8_t *)outputs[score_idx].buf, rknn_app->output_attrs[score_idx].zp, rknn_app->output_attrs[score_idx].scale,
        //                              (int8_t *)score_sum, score_sum_zp, score_sum_scale,
        //                              grid_h, grid_w, stride, dfl_len, 
        //                              filterBoxes, objProbs, classId, conf_threshold);
        
        if(rknn_app->is_quant)
        {
            // 处理量化数据
            validCount += process_i8((int8_t *)outputs[box_idx].buf, rknn_app->output_attrs[box_idx].zp, rknn_app->output_attrs[box_idx].scale,
                                     (int8_t *)outputs[score_idx].buf, rknn_app->output_attrs[score_idx].zp, rknn_app->output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            // 处理浮点数数据
            validCount += process_fp32((float *)outputs[box_idx].buf, (float *)outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       filterBoxes, objProbs, classId, conf_threshold);
        }
    }
    
    // no object detect 如果没有检测到目标，直接返回
    if(validCount <= 0)
    {
        return 0;
    }
    
    // 创建用于存储排序后索引的向量
    std::vector<int> indexArray;   
    // 将有效目标的索引加入到 indexArray 中
    for(int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    // 对置信度进行降序排序，并存储到 indexArray 中
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    
    // 创建 class_set 用于存储所有检测到的类别ID的集合
    std::set<int> class_set(std::begin(classId), std::end(classId));
    
    // 针对每个类别应用非极大值抑制（NMS），过滤掉重叠过多的检测框
    for(auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }
    
    // 初始化最终目标检测结果的数量
    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */ 
    // 遍历排序后的索引
    //将存储的模型尺寸中的坐标映射到原始输入图像中
    for(int i = 0; i < validCount; ++i)
    {
        // 如果索引为-1或者目标数量已经达到上限，继续下一轮循环
        if(indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        // 获取当前目标的索引
        int n = indexArray[i];
        // 根据索引获取目标框的坐标信息,类别和置信度（这也是还在模型尺寸上）
        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        // 根据 letterbox 的信息进行坐标调整
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);   //clamp 用于将坐标值限制在模型输入的宽度和高度范围内
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);    //将坐标从模型尺寸映射回原始输入图像
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        od_results->results[last_count].prop = obj_conf;
        char* label = labels[id];
        strncpy(od_results->results[last_count].name, label, OBJ_NAME_MAX_SIZE);   //将源字符串label复制最多OBJ_NAME_MAX_SIZE个字符到目标字符串name

        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

void deinitPostProcess()
{
    for(int i = 0; i < OBJ_CLASS_NUM; i++) 
    {
        if(labels[i] != nullptr) 
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}

void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color)
{
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), scale, scale);
    int pad_width = target_size.width - resized_image.cols;
    int pad_height = target_size.height - resized_image.rows;
    
    pads.left = pad_width / 2;
    pads.right = pad_width - pads.left;
    pads.top = pad_height / 2;
    pads.bottom = pad_height - pads.top;
    cv::copyMakeBorder(resized_image, padded_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, pad_color);
    
}