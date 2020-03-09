#include <torch/torch.h>

#include <vector>
#include <iostream>

// CUDA forward declarations
at::Tensor nms_cuda_forward(
    at::Tensor dets,
    const float nms_overlap_thresh);

// CPU forward declarations
at::Tensor nms_cpu_forward(
    at::Tensor dets,
    const float nms_overlap_thresh);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

at::Tensor nms_forward(
    at::Tensor dets,
    const float nms_overlap_thresh){

    CHECK_CONTIGUOUS(dets);

    if(dets.type().is_cuda())
    {
        return nms_cuda_forward(dets, nms_overlap_thresh);
    }
    else
    {
        return nms_cpu_forward(dets, nms_overlap_thresh);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nms_forward, "NMS forward");
}