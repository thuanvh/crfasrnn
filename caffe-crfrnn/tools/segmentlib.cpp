#include "caffe\ImageRegresionNN.h"
#include "caffe\segmentlib.h"
#include <opencv2/opencv.hpp>
#include <list>
//#include "caffe\caffe.hpp"
#include <windows.h>
#define _TIME_LOG_
namespace {
  cv::Mat MatScaleExtend(const cv::Mat& mat, const cv::Size& size, int border_type, const cv::Scalar& value, cv::Rect& rect_region)
  {
    cv::Mat resize_mat;
    float scale_x = size.width / (float)mat.cols;
    float scale_y = size.height / (float)mat.rows;
    float scale = (std::min)(scale_x, scale_y);
    cv::Size mat_size(mat.cols * scale, mat.rows * scale);
    cv::resize(mat, resize_mat, mat_size);
    int left = (size.width - mat_size.width) / 2;
    int top = (size.height - mat_size.height) / 2;
    int right = size.width - left - mat_size.width;
    int bot = size.height - top - mat_size.height;
    cv::copyMakeBorder(resize_mat, resize_mat, top, bot, left, right, border_type, value);
    rect_region = cv::Rect(left, top, size.width - right - left, size.height - top - bot);
    return resize_mat;
  }


  void Blob2Mat(const float* blob, int channels, int height, int width, cv::Mat & mat)
  {
    //int size = (int)sqrt(blob_size / 3.0f);
    int area = height * width;
    std::vector<cv::Mat> mat_vec(channels);
    for (int i = 0; i < channels; ++i)
    {
      const float* ptr = blob + i * area;
      mat_vec[i] = cv::Mat(height, width, CV_32FC1);
      float* m_ptr = mat_vec[i].ptr<float>();
      memcpy(m_ptr, ptr, area*sizeof(float));
      //caffe::caffe_copy(area, ptr, m_ptr);
      //cv::Mat display = mat_vec[i];
      //display.convertTo(display, CV_8UC1, 128, 128);
      //imwrite("output" + std::to_string(i) + ".jpg", display);
    }
    cv::merge(mat_vec, mat);
  }
}


void SemanticSegment::Initialize(const std::string& model, const std::string& trained)
{
  regress_.LoadNet(model, trained);
}
void SemanticSegment::SetInputSize(const cv::Size& size)
{
  input_size_ = size;
}
void SemanticSegment::SetClassNumber(int class_num)
{
  class_num_ = class_num;
}
cv::Scalar SemanticSegment::GetBgColor()
{
  cv::RNG rng;
  cv::Scalar bg_color_val(rng.next() % 255, rng.next() % 255, rng.next() % 255);
  return bg_color_val;
}
cv::Mat SemanticSegment::ScaleImage(const cv::Mat& src, cv::Rect& rect_region)
{
  cv::Scalar bg_color_val = GetBgColor();
  cv::Mat image_scale = MatScaleExtend(src, input_size_, cv::BORDER_CONSTANT, bg_color_val, rect_region);

  cv::Mat input;
  std::string scale = "substract";
  bool use_scale = scale == "scale";
  if (use_scale)
  {
    image_scale.convertTo(input, CV_32FC3, 2 / 255.0f, -1);
  }
  else {
    image_scale.convertTo(input, CV_32FC3);
  }
  bool subtract = !use_scale;
  if (subtract)
    input -= cv::Scalar(103.939, 116.779, 123.68);
  return input;
}
cv::Mat SemanticSegment::Segment(const cv::Mat& src)
{
  cv::Rect rect_region;
  cv::Mat input = ScaleImage(src, rect_region);
  std::vector<float> val = regress_.Regression(input);

  cv::Mat prob_mat;
  Blob2Mat(&(val[0]), class_num_, input.rows, input.cols, prob_mat);
  int pix_count = input.rows * input.cols;

  float* pval = prob_mat.ptr<float>();
  cv::Mat img_segment(input.size(), CV_16SC1);
  for (int y = 0; y < input.rows; y++)
  {
    for (int x = 0; x < input.cols; ++x)
    {
      int idx = std::max_element(pval, pval + class_num_) - pval;
      img_segment.at<short>(y, x) = idx;
      pval += class_num_;
    }
  }

  cv::Mat img_seg_crop = img_segment(rect_region);
  cv::Mat img_seg_output;
  cv::resize(img_seg_crop, img_seg_output, src.size());
  return img_seg_output;
}
