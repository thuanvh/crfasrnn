#include "caffe\ImageRegresionNN.h"
#include <opencv2/opencv.hpp>
#include <list>

#ifdef _SEMANTIC_SEGMENT_EXPORT_
#define DllExport   __declspec( dllexport ) 
#else
#define DllExport   
#endif

class DllExport SemanticSegment {
public:
  void Initialize(const std::string& model, const std::string& trained);
  void SetInputSize(const cv::Size& size);
  void SetClassNumber(int class_num);
  cv::Scalar GetBgColor();
  cv::Mat ScaleImage(const cv::Mat& src, cv::Rect& rect_region);
  cv::Mat Segment(const cv::Mat& src);
private:
  ImageRegresionNN regress_;
  cv::Size input_size_;
  int class_num_;
};

