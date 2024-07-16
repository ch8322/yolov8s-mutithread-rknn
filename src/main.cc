/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <queue>
#include <vector>
#define _BASETSD_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "yolov8s.h"
#include "ThreadPool.hpp"
using std::queue;
using std::time;
using std::time_t;
using std::vector;
int main(int argc, char **argv)
{
  char *model_name = NULL;
  if (argc != 3)
  {
    printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    return -1;
  }
  model_name = (char *)argv[1]; // 参数二，模型所在路径
  char *image_name = argv[2];   // 参数三, 视频/摄像头
  printf("模型名称:\t%s\n", model_name);

  cv::VideoCapture capture;
  cv::namedWindow("Camera FPS");
  if (strlen(image_name) == 1)
    capture.open((int)(image_name[0] - '0'));
  else
    capture.open(image_name);

  // 设置线程数
  int n = 3, frames = 0;
  printf("线程数:\t%d\n", n);
  // 类似于多个rk模型的集合?
  vector<rkyolov8s *> rkpool;
  // 线程池
  dpool::ThreadPool pool(n);
  // 线程队列
  queue<std::future<int>> futs;

  //初始化
  for (int i = 0; i < n; i++)
  {
    rkyolov8s *ptr = new rkyolov8s(model_name, i % 3);
    rkpool.push_back(ptr);
    capture >> ptr->ori_img;
    futs.push(pool.submit(&rkyolov8s::interf, &(*ptr)));
  }

  struct timeval time;
  gettimeofday(&time, nullptr);
  auto initTime = time.tv_sec * 1000 + time.tv_usec / 1000;

  gettimeofday(&time, nullptr);
  long tmpTime, lopTime = time.tv_sec * 1000 + time.tv_usec / 1000;

  while (capture.isOpened())
  {
    if (futs.front().get() != 0)
      break;
    futs.pop();
    cv::imshow("Camera FPS", rkpool[frames % n]->ori_img);
    if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出
      break;
    if(!capture.read(rkpool[frames % n]->ori_img))
      break;
    futs.push(pool.submit(&rkyolov8s::interf, &(*rkpool[frames++ % n])));

    if(frames % 60 == 0){
        gettimeofday(&time, nullptr);
        tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        printf("60帧平均帧率:\t%f帧\n", 60000.0 / (float)(tmpTime - lopTime));
        lopTime = tmpTime;
    }
  }

  gettimeofday(&time, nullptr);
  printf("\n平均帧率:\t%f帧\n", float(frames) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);

  // 释放剩下的资源
  while (!futs.empty())
  {
    if (futs.front().get())
      break;
    futs.pop();
  }
  for (int i = 0; i < n; i++)
    delete rkpool[i];
  capture.release();
  cv::destroyAllWindows();
  return 0;
}
