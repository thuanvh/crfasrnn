#include "caffe\ImageRegresionNN.h"
#include <opencv2/opencv.hpp>
//#include "caffe\caffe.hpp"

cv::Mat MatScaleExtend(const cv::Mat& mat, const cv::Size& size, int border_type, const cv::Scalar& value, cv::Rect& rect_region)
{
  cv::Mat resize_mat;
  float scale_x = size.width / (float)mat.cols;
  float scale_y = size.height / (float)mat.rows;
  float scale = std::min(scale_x, scale_y);
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
  }
  cv::merge(mat_vec, mat);
}

int main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);

  ImageRegresionNN regress;
  regress.LoadNet(argv[1], argv[2]);

  cv::Mat img = cv::imread(argv[3]);
  int width = atoi(argv[4]);
  int height = atoi(argv[5]);
  int class_num = atoi(argv[6]);

  cv::Size size(width, height);
  cv::Rect rect_region;
  cv::Mat image_scale = MatScaleExtend(img, size, cv::BORDER_CONSTANT, 0, rect_region);

  std::vector<float> val = regress.Regression(image_scale, 1);
  cv::Mat prob_mat;
  Blob2Mat(&(val[0]), class_num, height, width, prob_mat);
  int pix_count = height * width;

  //std::vector<int> class_val(pix_count);
  float* pval = prob_mat.ptr<float>();
  //for (int i = 0; i < pix_count; ++i, pval+=class_num)
  //{
  //  /*std::vector<std::pair<float, int> > pair_value;
  //  for (int k = 0; k < class_num; ++k)
  //    pair_value.push_back(std::pair<float, int>(pval[k], k));*/

  //  class_val[i] = std::max_element(pval, pval + class_num) - pval;
  //}
  cv::Mat img_segment(height, width, CV_16SC1);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; ++x)
    {
      int idx = std::max_element(pval, pval + class_num) - pval;
      img_segment.at<short>(y, x) = idx;
      pval += class_num;
    }
  }
  
  
  cv::Mat img_seg_crop = img_segment(rect_region);
  cv::Mat img_seg_output;
  cv::resize(img_seg_crop, img_seg_output, img.size());

  
  std::vector<cv::Scalar> color_map(class_num);
  cv::RNG rng;
  for (int i = 0; i < class_num; ++i)
  {
    color_map[i] = cv::Scalar(rng.next() % 255, rng.next() % 255, rng.next() % 255);
  }
  cv::Mat img_seg_color(img_seg_output.size(), CV_8UC3);
  for (int y = 0; y < img.rows; ++y)
  {
    for (int x = 0; x < img.cols; ++x)
    {
      int class_id = cvRound(img_seg_output.at<short>(y, x));
      if (class_id >= class_num)
        class_id = class_num - 1;
      else if (class_id < 0)
        class_id = 0;
        
      img_seg_color.at<cv::Vec3b>(y, x) = cv::Vec3b(color_map[class_id][0], color_map[class_id][1], color_map[class_id][2]);
    }
  }
  cv::imwrite("output.jpg", img_seg_color);
  return 0;
}

//segmentation.exe
//train_val.prototxt model.caffemodel a.jpg 500 500 21