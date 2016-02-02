#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace crfasrnn_caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace crfasrnn_caffe
