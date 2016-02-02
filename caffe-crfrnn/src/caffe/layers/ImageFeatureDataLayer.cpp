#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"
#include "caffe/ImageFeatureDataLayer.h"


#include <opencv2/opencv.hpp>
namespace {
  int file_count = 0;

  void Blob2Mat(const float* blob, int blob_size, cv::Mat & mat)
  {
    int size = (int)sqrt(blob_size / 3.0f);
    int area = size * size;
    std::vector<cv::Mat> mat_vec(3);
    for (int i = 0; i < 3; ++i)
    {
      const float* ptr = blob + i * area;
      mat_vec[i] = cv::Mat(size, size, CV_32FC1);
      float* m_ptr = mat_vec[i].ptr<float>();
      //memcpy(m_ptr, ptr, size*size*sizeof(float));
      crfasrnn_caffe::caffe_copy(size*size, ptr, m_ptr);
    }
    cv::merge(mat_vec, mat);
  }
  void SaveBlob(const float* blob, int blob_size, const std::string& name_prefix)
  {
    cv::Mat blobMat;
    Blob2Mat(blob, blob_size, blobMat);
    blobMat.convertTo(blobMat, CV_8UC3, 255);
    char fname[255];
    sprintf(fname, "%s_%d.png", name_prefix.c_str(), file_count);
    cv::imwrite(fname, blobMat);
  }
}


namespace crfasrnn_caffe {

//template <typename Dtype>
//ImageFeatureDataLayer<Dtype>::~ImageFeatureDataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::AddImageFeatureData(const std::list<cv::Mat>& src, const std::list<std::vector<float> >& features)
{
  mutex.lock();
  int row = src.size();
  int cols = src.begin()->channels();
  int height = src.begin()->rows;
  int width = src.begin()->cols;
  data_blob_.Reshape(row, cols, height, width);
  int labelsize = features.begin()->size();
  label_blob_.Reshape(row, labelsize, 1, 1);
  
  //num_files = row;
  //current_file_ = 0;

  CHECK_EQ(data_blob_.num(), label_blob_.num());
  //LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::AddImageFeatureData(const std::list<cv::Mat>& src)
{
  mutex.lock();
  int row = src.size();
  int cols = src.begin()->channels();
  int height = src.begin()->rows;
  int width = src.begin()->cols;  
  data_.reset(new Dtype[sizeof(Dtype) * cols * row * height * width]);
  data_blob_.Reshape(row, cols, height, width);
  Dtype* ptr = data_.get();  
  //LOG(INFO) << "Successully loaded " << row << "," << cols << "," << height << "," <<width;
  int blocksize = height * width;
  for (std::list<cv::Mat>::const_iterator it = src.begin(); it != src.end(); it++)
  {
    std::vector<cv::Mat> planes(cols);
    if(cols == 3)
    {
      cv::split(*it, planes);
    }
    else
    {
      planes[0] = *it;
    }
    for (int c = 0; c < cols; c++)
    {
      cv::Mat aplane = planes[c];
      //memcpy((void*)ptr, (void*)aplane.ptr<Dtype>(), blocksize  * sizeof(Dtype));
      caffe_copy(blocksize, aplane.ptr<Dtype>(), ptr);
      ptr += blocksize;
    }
  }
  data_blob_.set_cpu_data(data_.get());

  int feature_size = this->layer_param_.image_feature_data_param().feature_size();
  label_blob_.Reshape(row, feature_size, 1, 1);

  //num_files = row;
  //current_file_ = 0;

  CHECK_EQ(data_blob_.num(), label_blob_.num());
  //LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int feature_size = this->layer_param_.memory_data_param().batch_size();
  int c = this->layer_param_.memory_data_param().channels();
  int h = this->layer_param_.memory_data_param().height();
  int w = this->layer_param_.memory_data_param().width();
  

  /*int c = this->layer_param_.image_feature_data_param().channel();
  int w = this->layer_param_.image_feature_data_param().width();
  int h = this->layer_param_.image_feature_data_param().height();
  int feature_size = this->layer_param_.image_feature_data_param().feature_size();*/

  // Reshape blobs.
  const int batch_size = 1;
  top[0]->Reshape(batch_size, c, h, w);
  top[1]->Reshape(batch_size, feature_size, 1, 1);
  LOG(INFO) << "output data size: " << batch_size << "," << c << "," << h << "," << w;
}

template <typename Dtype>
void ImageFeatureDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = 1;//this->layer_param_.hdf5_data_param().batch_size();
  const int data_count = top[0]->count() / top[0]->num();
  const int label_data_count = top[1]->count() / top[1]->num();  
  int current_row_ = 0;
  //LOG(INFO) << "Loop iter: "<< current_row_ <<"," << data_count << "," << label_data_count;
  for (int i = 0; i < batch_size; ++i, ++current_row_) {    
    /*memcpy(&top[0]->mutable_cpu_data()[i * data_count],
           &data_blob_.cpu_data()[current_row_ * data_count],
           sizeof(Dtype) * data_count);
    memcpy(&top[1]->mutable_cpu_data()[i * label_data_count],
            &label_blob_.cpu_data()[current_row_ * label_data_count],
            sizeof(Dtype) * label_data_count);*/
    caffe_copy(data_count,
      &data_blob_.cpu_data()[current_row_ * data_count], &top[0]->mutable_cpu_data()[i * data_count]);
    caffe_copy(label_data_count,
      &label_blob_.cpu_data()[current_row_ * label_data_count], &top[1]->mutable_cpu_data()[i * label_data_count]);
/*
    if (sizeof(Dtype) == sizeof(float)) {
      int data_dim = top[0]->count() / top[0]->shape(0);
      SaveBlob((float*)top[0]->cpu_data(), data_dim, "ifd_0");      
      ++file_count;
    }
*/
  }
#if 0
  int w = (*top)[0]->width();
  int h = (*top)[0]->height();;
  //LOG(INFO) << "result image size:" << result[iresult]->num() << " " << h << "x"<< w <<" channels: " << result[iresult]->channels();
  Dtype* data = (*top)[0]->mutable_cpu_data();
  cv::Mat img = cv::Mat(h,w,CV_8UC3);
  for(int y = 0; y < h; y++){
    for(int x = 0; x < w; x++){
      int r = (int)(data[ 0 * w * h + y * w + x] * 255);
      int g = (int)(data[ 1 * w * h + y * w + x] * 255);
      int b = (int)(data[ 2 * w * h + y * w + x] * 255);
      //if( r == g && g == b)
      //LOG(INFO) << "gray image";
      r = std::min(std::max(r,0),255);
      g = std::min(std::max(g,0),255);
      b = std::min(std::max(b,0),255);

      img.at<cv::Vec3b>(y,x) = cv::Vec3b(r,g,b);
    }// end x
  }// end y

  char fname[255];
  cv::Mat imorg = img.clone();
  sprintf(fname, "predict_process/img_%d.jpg", 0);
  //cv::imwrite(fname, imorg);
  IplImage i1 = imorg;
  cvSaveImage(fname, &i1);
#endif
  
  mutex.unlock();
}

INSTANTIATE_CLASS(ImageFeatureDataLayer);
REGISTER_LAYER_CLASS(IMAGE_FEATURE_DATA, ImageFeatureDataLayer);
}  // namespace crfasrnn_caffe
