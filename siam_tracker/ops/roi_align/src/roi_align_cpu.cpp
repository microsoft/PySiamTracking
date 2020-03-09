#include <ATen/ATen.h>

#include <stdio.h>
#include <math.h>

using std::max;

float bilinear_interpolate(const float *bottom_data, const int height, const int width,
		float y, float x)
{

	// deal with cases that inverse elements are out of feature map boundary
	if (y < -1.0 || y > height || x < -1.0 || x > width) {
		return 0;
	}

	if (y <= 0) y = 0;
	if (x <= 0) x = 0;

	int y_low = (int)y;
	int x_low = (int)x;
	int y_high;
	int x_high;

	if (y_low >= height - 1) {
		y_high = y_low = height - 1;
		y = (float)y_low;
	}
	else {
		y_high = y_low + 1;
	}

	if (x_low >= width - 1) {
		x_high = x_low = width - 1;
		x = (float)x_low;
	}
	else {
		x_high = x_low + 1;
	}

	float ly = y - y_low;
	float lx = x - x_low;
	float hy = 1. - ly;
	float hx = 1. - lx;
	// do bilinear interpolation
	float lt = bottom_data[y_low * width + x_low];
	float rt = bottom_data[y_low * width + x_high];
	float lb = bottom_data[y_high * width + x_low];
	float rb = bottom_data[y_high * width + x_high];
	float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

	float val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

	return val;

}


void roi_align_per_box(
    const float * image_data,
    const float * roi_data,
    const int batch_size,
    const int channels,
    const int image_height,
    const int image_width,
    const int num_rois,
    const int pooled_height,
    const int pooled_width,
    const float spatial_scale,
    const int sample_num,
    float * output_data)
{
    const int image_channel_elements = image_height * image_width;
    const int image_elements = channels * image_channel_elements;

    for (int b = 0; b < num_rois; ++b) {
        const float * box = roi_data + b * 5;
        const int roi_index = static_cast<int>(box[0]);
        if (roi_index < 0 || roi_index >= batch_size) {
            printf("Error: batch_index %d out of range [0, %d)\n", roi_index, batch_size);
            continue;
        }
        float roi_start_w = box[1] * spatial_scale;
		float roi_start_h = box[2] * spatial_scale;
		float roi_end_w = box[3] * spatial_scale;
		float roi_end_h = box[4] * spatial_scale;

		float roi_height = max(roi_end_h - roi_start_h, float(0.f));
		float roi_width = max(roi_end_w - roi_start_w, float(0.f));

		const float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
		const float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

		const float* batch_data = image_data + roi_index * image_elements;
        int sample_num_h = (sample_num > 0)	? sample_num : ceil(roi_height / pooled_height);  // e.g., = 2
		int sample_num_w = (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

        for (int c = 0; c < channels; ++c) {
			for (int ph = 0; ph < pooled_height; ++ph) {
				for (int pw = 0; pw < pooled_width; ++pw) {
                    float output_val = 0.;
					for (int iy = 0; iy < sample_num_h; iy++) {
					    const float y = roi_start_h + ph * bin_size_h +
                            static_cast<float>(iy+1) * bin_size_h / static_cast<float>(sample_num_h+1);
					    for (int ix = 0; ix < sample_num_w; ix++) {
							const float x = roi_start_w + pw * bin_size_w +
                                static_cast<float>(ix+1) * bin_size_w / static_cast<float>(sample_num_w+1);
							float val = bilinear_interpolate(batch_data, image_height, image_width, y, x);
							output_val += val;
						}
					}
					output_val /= (sample_num_h * sample_num_w);
					output_data[ph * pooled_width + pw] = output_val;
				}
			}
			output_data += (pooled_height * pooled_width);
			batch_data += (image_height * image_width);
		} // channel
	} // b
}


int roi_align_forward_cpu(at::Tensor features,
                          at::Tensor rois,
                          int pooled_height,
                          int pooled_width,
                          float spatial_scale,
                          int sample_num,
                          at::Tensor output)
{
    roi_align_per_box(
        features.data<float>(),
        rois.data<float>(),
        features.size(0),
        features.size(1),
        features.size(2),
        features.size(3),
        rois.size(0),
        pooled_height,
        pooled_width,
        spatial_scale,
        sample_num,
        output.data<float>()
    );
    return 1;
}

int roi_align_backward_cpu(at::Tensor top_grad,
                           at::Tensor rois,
                           int pooled_height,
                           int pooled_width,
                           float spatial_scale,
                           int sample_num,
                           at::Tensor bottom_grad)
{
    printf("Not implemented. [RoI align CPU backward function]");
    return 0;
}