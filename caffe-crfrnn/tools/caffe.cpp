#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"

using crfasrnn_caffe::Blob;
using crfasrnn_caffe::Caffe;
using crfasrnn_caffe::Net;
using crfasrnn_caffe::Layer;
using crfasrnn_caffe::shared_ptr;
using crfasrnn_caffe::Timer;
using crfasrnn_caffe::vector;


DEFINE_int32(gpu, -1,
  "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
  "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
  "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
  "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
  "Optional; the pretrained weights to initialize finetuning. "
  "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
  "The number of iterations to run.");

// A simple registry for crfasrnn_caffe commands.
typedef int(*BrewFunction)();
typedef std::map<crfasrnn_caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const crfasrnn_caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  }
  else {
    LOG(ERROR) << "Available crfasrnn_caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
    it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

cv::Mat upsample_filt(int size)
{
  int factor = (size + 1) / 2;
  float center = 0;
  if (size % 2 == 1)
    center = factor - 1;
  else
    center = factor - 0.5f;
  cv::Mat a(size, 1, CV_32FC1);
  for (int i = 0; i < size; ++i)
    a.at<float>(i, 0) = i;
  a = abs(a - center);
  a *= 1.0f / factor;
  a = 1 - a;
  a = a * a.t();
  return a;
}

void print_mat(const cv::Mat& filt)
{
  for (int y = 0; y < filt.rows; ++y) {
    for (int x = 0; x < filt.cols; ++x)
      std::cout << filt.at<float>(y, x) << ", ";
    std::cout << std::endl;
  }
}
void repeat_data(const float* filt_data, float* data, shared_ptr<Blob<float> >& blob)
{
  int size = blob->height() * blob->width();
  /*for (int i = 0; i < blob->num(); ++i)
    for (int j = 0; j < blob->channels();++j, data += size)
      crfasrnn_caffe::caffe_copy<float>(size, filt_data, data);*/

  for (int i = 0; i < blob->num(); ++i)
  {
    data += (i * blob->channels() + i) * size;
    crfasrnn_caffe::caffe_copy<float>(size, filt_data, data);
  }
}
void copy_mat2blob(const cv::Mat& filt, shared_ptr<Blob<float> >& blob)
{
  float* data = NULL;
  
  {
    data = blob->mutable_cpu_data();
    const float* filt_data = filt.ptr<float>();
    repeat_data(filt_data, data, blob);
  }
  /*if (Caffe::mode() == Caffe::GPU)
  {
    data = blob->mutable_gpu_data();
    const float* filt_data = filt.ptr<float>();
    repeat_data(filt_data, data, blob);
  }*/ 
}

void interp_surgery(shared_ptr<Net<float>>& net)
{
  std::vector<std::string> layernames{ "up","score2","score4" };
  for (auto layer : net->layers())
  {
    std::string layer_name = layer->layer_param().name();
    for (size_t i = 0; i < layernames.size(); i++)
    {
      if (layer_name.substr(0, layernames[i].length()) == layernames[i])
      {
        if (layer->blobs().size() > 0)
        {
          LOG(INFO) << "Initialize deconvolution layer " << layer_name;
          auto blob = layer->blobs()[0];
          CHECK_EQ(blob->num(), blob->channels()) << "input + output channels need to be the same";
          CHECK_EQ(blob->height(), blob->width()) << "filters need to be square";
          cv::Mat filt = upsample_filt(blob->height());
          copy_mat2blob(filt, blob);
        }
      }
    }
  }  
}
// crfasrnn_caffe commands to call by
//     crfasrnn_caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  crfasrnn_caffe::Caffe::SetDevice(FLAGS_gpu);
  crfasrnn_caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);


// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  //std::cout << FLAGS_solver << std::endl;
  crfasrnn_caffe::SolverParameter solver_param;
  crfasrnn_caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  //crfasrnn_caffe::ReadProtoFromTextFileOrDie("solver_gpu.prototxt", &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == crfasrnn_caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<crfasrnn_caffe::Solver<float> >
    solver(crfasrnn_caffe::GetSolver<float>(solver_param));

  interp_surgery(solver->net());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    LOG(INFO) << "Finetuning from " << FLAGS_weights;
    solver->net()->CopyTrainedLayersFrom(FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the crfasrnn_caffe net.
  Caffe::set_phase(Caffe::TEST);
  Net<float> caffe_net(FLAGS_model);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight =
        caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[i]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the crfasrnn_caffe net.
  Caffe::set_phase(Caffe::TRAIN);
  Net<float> caffe_net(FLAGS_model);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      // Although Reshape should be essentially free, we include it here
      // so that we will notice Reshape performance bugs.
      layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const crfasrnn_caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  cv::Mat filt = upsample_filt(1);
  print_mat(filt);
  filt = upsample_filt(2);
  print_mat(filt);
  filt = upsample_filt(3);
  print_mat(filt);
  filt = upsample_filt(4);
  print_mat(filt);
  filt = upsample_filt(5);
  print_mat(filt);
  // Print output to stderr (while still logging).
  //FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: crfasrnn_caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  crfasrnn_caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(crfasrnn_caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/crfasrnn_caffe");
  }
}
