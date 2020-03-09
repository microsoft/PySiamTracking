#include <torch/extension.h>

#include <cmath>
#include <vector>


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")


// CUDA forward declarations
int roi_align_forward_cuda(at::Tensor features,
                           at::Tensor rois,
                           int pooled_height,
                           int pooled_width,
                           float spatial_scale,
                           int sample_num,
                           at::Tensor output);

int roi_align_backward_cuda(at::Tensor top_grad,
                            at::Tensor rois,
                            int pooled_height,
                            int pooled_width,
                            float spatial_scale,
                            int sample_num,
                            at::Tensor bottom_grad);

// CPU forward declarations
int roi_align_forward_cpu(at::Tensor features,
                          at::Tensor rois,
                          int pooled_height,
                          int pooled_width,
                          float spatial_scale,
                          int sample_num,
                          at::Tensor output);

int roi_align_backward_cpu(at::Tensor top_grad,
                           at::Tensor rois,
                           int pooled_height,
                           int pooled_width,
                           float spatial_scale,
                           int sample_num,
                           at::Tensor bottom_grad);


int roi_align_forward(at::Tensor features,
                      at::Tensor rois,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      int sample_num,
                      at::Tensor output)
{
    CHECK_CONTIGUOUS(features);
    CHECK_CONTIGUOUS(rois);
    CHECK_CONTIGUOUS(output);

    if(features.type().is_cuda())
    {
        CHECK_CUDA(rois);
        CHECK_CUDA(output);
        return roi_align_forward_cuda(features, rois, pooled_height, pooled_width,
            spatial_scale, sample_num, output);
    }
    else
    {
        return roi_align_forward_cpu(features, rois, pooled_height, pooled_width,
            spatial_scale, sample_num, output);
    }
}


int roi_align_backward(at::Tensor top_grad,
                       at::Tensor rois,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int sample_num,
                       at::Tensor bottom_grad)
{
    CHECK_CONTIGUOUS(top_grad);
    CHECK_CONTIGUOUS(rois);
    CHECK_CONTIGUOUS(bottom_grad);

    if(top_grad.type().is_cuda())
    {
        CHECK_CUDA(rois);
        CHECK_CUDA(bottom_grad);
        return roi_align_backward_cuda(top_grad, rois, pooled_height,
            pooled_width, spatial_scale, sample_num, bottom_grad);
    }
    else
    {
        return roi_align_backward_cpu(top_grad, rois, pooled_height,
            pooled_width, spatial_scale, sample_num, bottom_grad);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_align_forward, "Roi_Align forward");
  m.def("backward", &roi_align_backward, "Roi_Align backward");
}
