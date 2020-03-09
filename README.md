# PySiamTracking

## Introduction

PySiamTracking is an open source toolbox that supports a series of siamese-network-based tracking methods like SiamFC / SiamRPN / SPM. It's built based on PyTorch. 

![](docs/demo.gif)

We follow the modular design in [mmdetection](https://github.com/open-mmlab/mmdetection). The components in a typical siamese-network-based tracker (e.g. backbone, fusion module, post head, loss functions, ...) are decomposed so that it is easy to verify the effectiveness of each component. One can also easily replace some components with the customized modules. We hope this project can help to the research in visual object tracking.


## Benchmark and Model Zoo

We provide dozens of models trained on 4 datasets: COCO, TrackingNet, LaSOT-train, GOT10K. For more information, please visit to [Model Zoo](docs/model_zoo.md).

![](docs/speed_auc_otb.png)



## Getting Started

### Requirement

* Linux (Ubuntu 16.04 is tested)
* Python > 3.5 (Python 3.6 is tested)
* CUDA >= 9.0 (CUDA 9.0 & 10.0 are tested)

### Installation

1. (Recommend)  Create a new virtual environment and install PyTorch >= 1.0.

```bash
conda create -n siamtracking python=3.6 -y
conda activate siamtracking
# please select a suitable version for your device
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

2. Clone this repository.
3. Install some necessary python packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Compile some custom operations.

```bash
bash ./compile.sh
```

### Play with pre-trained models

The pre-trained models can be downloaded from [Model Zoo](docs/model_zoo.md). See [INFERENCE.md](docs/inference.md) for more detailed instruction.

### Training models

Please see [TRAIN.md](docs/train.md) for more details about training.



## Reference

* [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html), *Luca Bertinetto , Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H.S. Torr*, ECCVW 2016.
* [A Twofold Siamese Network for Real-Time Object Tracking](https://arxiv.org/abs/1802.08817), *Anfeng He, Chong Luo, Xinmei Tian, Wenjun Zeng*, CVPR 2018
* [High Performance Visual Tracking with Siamese Region Proposal Network](https://bo-li.info/SiamRPN/), *Bo Li, Junjie Yan, Wei Wu, Zheng Zhu, Xiaolin Hu*, CVPR 2018
* [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://bo-li.info/SiamRPN++/), *Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan*, CVPR 2019
* [SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking](https://arxiv.org/abs/1904.04452), *Guangting Wang, Chong Luo, Zhiwei Xiong, Wenjun Zeng*, CVPR 2019


------
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com)
with any additional questions or comments.