#pragma once
#include <vector>
#include "caffe/common.hpp"
#include <opencv2/opencv.hpp>
class CAFFE_DLL_EXPORT ImageRegresionNN
{
public:
  ImageRegresionNN();
  ~ImageRegresionNN(void);
  void LoadNet(const std::string& netconfig, const std::string& trainedLayers = "");
  /*
   * 0 : CPU, 1 : GPU
   */
  void SetMode(int mode);
  void SetDevice(int device);
  //std::vector<float> Regression(const cv::Mat& src, const std::vector<float>& values);
  std::vector<float> Regression(const cv::Mat& src, float scale = 1 / 255.0f, const std::string& blob_name = "");

  static void Train(const std::string& netconfig, const std::string& netresult = NULL, bool finetune_net = false);
  void InitNet(const std::string& input);
  void LoadTrainedLayer(const std::string& trainedfile);
private:
  void* m_Net;
};

