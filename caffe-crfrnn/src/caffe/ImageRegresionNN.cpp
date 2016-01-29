#include "caffe/ImageRegresionNN.h"
#include "caffe/caffe.hpp"
#include "caffe/ImageFeatureDataLayer.h"
//#include "../VirtualMakeover/utility/StringUtils.h"
using namespace caffe;

ImageRegresionNN::ImageRegresionNN():m_Net(NULL)
{
  
}

ImageRegresionNN::~ImageRegresionNN(void)
{
  if(m_Net)
    delete (Net<float>*)m_Net;
}

//std::vector<float> ImageRegresionNN::Regression(const cv::Mat& src, const std::vector<float>& values)
//{
//  std::vector<float> ret;
//  return ret;
//}

std::vector<float> ImageRegresionNN::Regression(const cv::Mat& input, const std::string& blob_name)
{
  std::vector<float> ret;
  Caffe::set_phase(Caffe::TEST);

  Net<float>* net = ((Net<float>*)(m_Net));

  const shared_ptr<ImageFeatureDataLayer<float> > image_data_layer =
    boost::static_pointer_cast<ImageFeatureDataLayer<float> >(
    net->layer_by_name("input"));
  std::list<cv::Mat> images;
  images.push_back(input);
  image_data_layer->AddImageFeatureData(images);

  float loss = 0.0;
  vector<Blob<float>* > dummy_bottom_vec;
  //const vector<Blob<float>*>& result = net->ForwardPrefilled();
  const vector<Blob<float>*>& result = net->Forward(dummy_bottom_vec, &loss);
  const float* data = NULL;
  int pnum = 0;
  if (blob_name.empty())
  {
    int iresult = 1;
    pnum = result[iresult]->width() * result[iresult]->height() * result[iresult]->channels();
    data = result[iresult]->cpu_data();
  }
  else {
    //net->Forward()
    const shared_ptr<Blob<float> > blob = net->blob_by_name(blob_name);
    pnum = blob->width() * blob->height() * blob->channels();
    data = blob->cpu_data();
  }

  ret.resize(pnum);
  /*for (int i = 0; i< pnum; ++i)
  {
    ret[i] = data[i];
  }*/
  caffe_copy<float>(pnum, data, &(ret[0]));

  /*std::ofstream ofs("output.txt");
  for (int i = 0; i < pnum; ++i)
  {
    ofs << ret[i] << " ";
  }
  ofs.close();*/
  return ret;  
}

void ImageRegresionNN::Train( const std::string& netconfig, const std::string& netresult , bool finetune_net )
{
  //::google::InitGoogleLogging(netconfig);

  caffe::SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(netconfig, &solver_param);

  LOG(INFO) << "Starting Optimization";
  caffe::SGDSolver<float> solver(solver_param);
  if (!netresult.empty()) 
  {
    LOG(INFO) << "Resuming from " << netresult;
    if(finetune_net)
    {
      solver.net()->CopyTrainedLayersFrom(netresult);
      solver.Solve();
    }
    else
    {
      solver.Solve(netresult);
    }    
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";
}

void ImageRegresionNN::SetMode( int mode )
{
  Caffe::set_mode((Caffe::Brew)mode);
}

void ImageRegresionNN::SetDevice( int device )
{
  Caffe::SetDevice(device);
}

void ImageRegresionNN::LoadNet( const std::string& netconfig, const std::string& trainedLayers)
{
  if(m_Net)
    delete (Net<float>*)m_Net;
  std::string input = netconfig;
  m_Net = new Net<float>(input);
  if (!trainedLayers.empty())
  {
    Net<float>* pnet = (Net<float>*)(m_Net);
    string trainedfile = trainedLayers;
    if(trainedfile.substr(trainedfile.length() - 4) == ".txt")
    {
      //pnet->CopyTrainedLayersFromTextFile(trainedLayers);
    }
    else
    {
      pnet->CopyTrainedLayersFrom(trainedLayers);
    }    
  }
}

void ImageRegresionNN::InitNet(const std::string& input)
{
  if(m_Net)
    delete (Net<float>*)m_Net;  
  m_Net = new Net<float>(input);
}

void ImageRegresionNN::LoadTrainedLayer(const std::string& trainedfile)
{
  Net<float>* pnet = (Net<float>*)(m_Net);

  if(trainedfile.substr(trainedfile.length() - 4) == ".txt")
  {
    //pnet->CopyTrainedLayersFromTextFile(trainedfile);
  }
  else
  {
    pnet->CopyTrainedLayersFrom(trainedfile);
  }
}