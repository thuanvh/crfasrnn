#include <opencv2/opencv.hpp>
#include <list>

#ifdef _SEMANTIC_SEGMENT_EXPORT_
#define DllExport   __declspec( dllexport ) 
#else
#define DllExport   
#endif
namespace semantic_segment {
  class ImageRegresionNN;


  class DllExport SemanticSegment {
  public:
    void Initialize(const std::string& model, const std::string& trained);
    void SetInputSize(const cv::Size& size);
    void SetClassNumber(int class_num);
    cv::Scalar GetBgColor();
    cv::Mat ScaleImage(const cv::Mat& src, cv::Rect& rect_region);
    cv::Mat Segment(const cv::Mat& src);
  private:
    std::auto_ptr<semantic_segment::ImageRegresionNN> regress_;
    cv::Size input_size_;
    int class_num_;
  };
}
