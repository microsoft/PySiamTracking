#include <ATen/ATen.h>

#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>

void compute_boxes_area(
    const float* boxes_data,
    const int boxes_num,
    float* boxes_area)
{
    int b;
    #pragma omp parallel for
    for (b = 0; b < boxes_num; ++b)
    {
        const float * box = boxes_data + b * 5;
        const float x1 = box[0];
        const float y1 = box[1];
        const float x2 = box[2];
        const float y2 = box[3];

        boxes_area[b] = (y2 - y1) * (x2 - x1);
    }
}


at::Tensor nms_cpu_forward(
    at::Tensor dets,
    const float nms_overlap_thresh)
{

    const int boxes_num = dets.size(0);
    // std::cout << dets.size(1) <<std::endl;
    int num_to_keep = 0;

    int* keep_out_cpu = new int[boxes_num];
    bool* merged = new bool[boxes_num];
    float* boxes_area = new float[boxes_num];
    memset(&keep_out_cpu[0], 0, sizeof(int) * boxes_num);
    memset(&merged[0], false, sizeof(bool) * boxes_num);

    const float* boxes_data = dets.data<float>();

    int i,j;
    const float* i_box = boxes_data;
    const float* j_box = boxes_data;
    float ix1, iy1, ix2, iy2, iarea;
    float xx1, yy1, xx2, yy2;
    float w, h, inter, ovr;

    // calculate box areas
    compute_boxes_area(boxes_data, boxes_num, boxes_area);

    for(i=0;i<boxes_num;i++){
        if(merged[i]){
            i_box += 5;
            continue;
        }
        ix1 = i_box[0];
        iy1 = i_box[1];
        ix2 = i_box[2];
        iy2 = i_box[3];
        // score = i_box[4];
        iarea = boxes_area[i];
        j_box = i_box + 5;
        for(j=i+1;j<boxes_num;j++){
            if(merged[j]){
                j_box += 5;
                continue;
            }
            xx1 = std::max(ix1, j_box[0]);
            yy1 = std::max(iy1, j_box[1]);
            xx2 = std::min(ix2, j_box[2]);
            yy2 = std::min(iy2, j_box[3]);
            w = std::max(0.0f, xx2 - xx1);
            h = std::max(0.0f, yy2 - yy1);
            inter = w * h;
            ovr = inter / (iarea + boxes_area[j] - inter);
            if(ovr >= nms_overlap_thresh){
                merged[j] = true;
            }
            j_box += 5;
        }
        i_box += 5;
        keep_out_cpu[num_to_keep++] = i;
    }

    // auto keep_out = at::CPU(at::kInt).zeros({num_to_keep});
    auto keep_out = at::zeros({num_to_keep}, at::TensorOptions(dets.type()).dtype(at::kInt));
    memcpy(keep_out.data<int>(), keep_out_cpu, sizeof(int)*num_to_keep);

    // free memory
    delete[] boxes_area;
    delete[] merged;
    delete[] keep_out_cpu;

    return keep_out;

}
