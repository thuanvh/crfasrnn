#include "caffe\ImageRegresionNN.h"
#include <opencv2/opencv.hpp>
#include <list>
//#include "caffe\caffe.hpp"
#include <windows.h>

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
  }
  cv::merge(mat_vec, mat);
}
struct SearchFile
{
  typedef std::vector<std::string> FileNameArray;
  FileNameArray files;

  FileNameArray::iterator begin()
  {
    return files.begin();
  }

  FileNameArray::iterator end()
  {
    return files.end();
  }

  int count() const
  {
    return (int)files.size();
  }

  std::string operator[](int index)
  {
    return files[index];
  }

  void operator()(const std::string &path, const std::string &pattern)
  {
    WIN32_FIND_DATA wfd;
    HANDLE hf;
    std::string findwhat;
    std::vector<std::string> dir;

    findwhat = path + "\\*";  // directory

    hf = FindFirstFile(findwhat.c_str(), &wfd);
    while (hf != INVALID_HANDLE_VALUE)
    {
      if (wfd.cFileName[0] != '.' && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
      {
        std::string found;

        found = path + "\\" + wfd.cFileName;
        dir.push_back(found);
      }

      if (!FindNextFile(hf, &wfd))
      {
        FindClose(hf);
        hf = INVALID_HANDLE_VALUE;
      }
    }

    findwhat = path + "\\" + pattern;  // files

    hf = FindFirstFile(findwhat.c_str(), &wfd);
    while (hf != INVALID_HANDLE_VALUE)
    {
      if (wfd.cFileName[0] != '.' && !(wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
      {
        std::string found;

        found = path + "\\" + wfd.cFileName;
        files.push_back(found);
      }

      if (!FindNextFile(hf, &wfd))
      {
        FindClose(hf);
        hf = INVALID_HANDLE_VALUE;
      }
    }

    // continue with directories
    for (std::vector<std::string>::iterator it = dir.begin(); it != dir.end(); ++it)
      operator()(*it, pattern);
  }
};

void ScanFile(const std::string& folder, const std::string& pattern, std::list<std::string>& files)
{
  SearchFile sf;
  sf(folder, pattern);
  for (int i = 0; i != sf.count(); ++i)
  {
    files.push_back(sf[i]);
  }
}
void CreateFolderRecursive(const std::string& path)
{
  size_t pos = 0;
  do
  {
    pos = path.find_first_of("\\/", pos + 1);
    CreateDirectory(path.substr(0, pos).c_str(), NULL);
  } while (pos != std::string::npos);
}

bool CreateFolderIfNotExist(const char* folder)
{
  if (CreateDirectory(folder, NULL) ||
    ERROR_ALREADY_EXISTS == GetLastError())
  {
    return true;
  }
  else
  {
    return false;
  }
}
std::string GetFileName(const std::string& fullpath)
{
  size_t found1 = fullpath.find_last_of("/\\");
  return fullpath.substr(found1 + 1);
}
int main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);

  std::string command = argv[1];

  ImageRegresionNN regress;
  regress.LoadNet(argv[2], argv[3]);

  int width = atoi(argv[4]);
  int height = atoi(argv[5]);
  int class_num = atoi(argv[6]);

  std::string input = argv[7];
  std::string output = argv[8];

  CreateFolderIfNotExist(output.c_str());
  std::ofstream ofs(output + "\\colormapping.txt");
  std::vector<cv::Scalar> color_map(class_num);
  cv::RNG rng;
  for (int i = 0; i < class_num; ++i)
  {
    color_map[i] = cv::Scalar(rng.next() % 255, rng.next() % 255, rng.next() % 255);
    ofs << i << ":" << color_map[i][0] << " " << color_map[i][1] << " " << color_map[i][2] << " " << std::endl;
  }
  ofs.close();

  std::list<std::string> input_files;
  if (command == "testfile")
  {
    input_files.push_back(input);
  }
  else if(command == "testfolder")
  {
    ScanFile(input, "*.*", input_files);
  }

  int file_index = 0;
  for (std::list<std::string>::const_iterator it = input_files.begin();
  it != input_files.end();
    ++it)
  {
    ++file_index;
    std::cout << file_index << " - " << *it << std::endl;
    cv::Mat img = cv::imread(*it);

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
    
    std::string file_name = GetFileName(*it);
    {
      std::string color_path = output + "\\color";
      CreateFolderIfNotExist(color_path.c_str());
      std::string color_image = color_path + "\\" + file_name;
      cv::imwrite(color_image, img_seg_color);
    }

    img_seg_output.convertTo(img_seg_output, CV_8UC1);

    {
      std::string seg_path = output + "\\seg";
      CreateFolderIfNotExist(seg_path.c_str());
      std::string seg_image = seg_path + "\\" + file_name;
      cv::imwrite(seg_image, img_seg_output);      
    }
    
    {
      img_seg_output *= 255 / class_num;
      std::string seg_path = output + "\\gray";
      CreateFolderIfNotExist(seg_path.c_str());
      std::string seg_image = seg_path + "\\" + file_name;
      cv::imwrite(seg_image, img_seg_output);
    }
  }
  return 0;
}

//segmentation.exe
//train_val.prototxt model.caffemodel a.jpg 500 500 21