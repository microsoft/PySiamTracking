# Training Models

## Training Data Preparation

In general, 3 video datasets and 1 image dataset are used to to train tracker models. One can download these datasets from official website.

* COCO: http://cocodataset.org/
* LaSOT: https://cis.temple.edu/lasot/
* TrackingNet: https://tracking-net.org/
* GOT10K: http://got-10k.aitestunion.com/

Following SiamFC, the datasets are firstly center-cropped and resized to a fixed size (446 x 446 in our setting). The preprocessing scripts are provided in `preprocessing/` folder. One can run these scripts to generate necessary training data.

```bash
cd preprocessing
# COCO dataset
python coco.py \
--image_dir <coco_root> \
--ann_path <coco_ann_path.json> \
--output_dir <output_root>/coco

# LaSOT train split
python lasot.py \
--root_dir <lasot_root> \
--output_dir <output_root>/lasot

# copy TrackingNet_class_info/ to <trackingnet_root>
# TrackingNet
python trackingnet.py \
--root_dir <trackingnet_root> \
--output_dir <output_root>/trackingnet

# GOT10k
python got10k.py \
--root_dir <got10k_root> \
--output_dir <output_root>/got10k
```

Since some datasets are quite large. We uniformly sample **1 frame in every 10 frames for TrackingNet dataset** and **1 frame in every 3 frames for LaSOT dataset**. The file size after preprocessing is about:

* COCO: 35.7 GB
* LaSOT: 26.7 GB

* TrackingNet: 29.2 GB
* GOT10K: 44.1 GB

We save the files as the following structure.

```
|   coco_clean.pkl
|   COCO.zip
|   got10k_train_clean.pkl
|   got10k_train_images.zip
|   lasot_train_clean.pkl
|   LaSOT_train_images.zip
|   trackingnet_clean.pkl
|   TrackingNet_images.zip
|
+---COCO/
+---got10k_train/
+---LaSOT_trian_images/
+---TrackingNet_images/
```

We also provide the pre-processed zip-file and pkl annotation files: 

- [coco_clean.pkl](https://imgtracking.blob.core.windows.net/pysiamtracking/data/coco_clean.pkl) / [COCO.zip](https://imgtracking.blob.core.windows.net/pysiamtracking/data/COCO.zip) 
- [got10k_train_clean.pkl](https://imgtracking.blob.core.windows.net/pysiamtracking/data/got10k_train_clean.pkl) / [got10k_train_images.zip](https://imgtracking.blob.core.windows.net/pysiamtracking/data/got10k_train_images.zip)
- [lasot_train_clean.pkl](https://imgtracking.blob.core.windows.net/pysiamtracking/data/lasot_train_clean.pkl) / [LaSOT_train_images.zip](https://imgtracking.blob.core.windows.net/pysiamtracking/data/LaSOT_trian_images.zip)
- [trackingnet_clean.pkl](https://imgtracking.blob.core.windows.net/pysiamtracking/data/trackingnet_clean.pkl) / [TrackingNet_images.zip](https://imgtracking.blob.core.windows.net/pysiamtracking/data/TrackingNet_images.zip)

For the users who have sufficient memory (> 160GB) and slow I/O speed (e.g. distributed storage through network), we recommend to pack images into a single .zip file (e.g. COCO.zip) so that it can be pre-loaded into memory for efficient reading.

 

## Download ImageNet Pre-trained Models

The model needs the ImageNet pre-trained models for initialization. One can simply run

```bash
python scripts/convert_model_from_imagenet.py
```

to download models from PyTorch official website and convert to our format. The pre-trained models will be saved to `data/pretrained_models/`. *(Please consider to upgrade torchvision if mobilenetv2 cannot be found).*



## Training Script

It's easy to train tracker models when above steps are ready. One can call

```bash
python tools/train_tracker.py \
--config experiments/model_configs/spm/spm_alexnet.py \
--data_dir <data_dir_path> \
--gpus <num_gpus> 
```

By default, the data will be loaded from .zip files. One can modify `storage_backend` in configuration file or add `--from_file` flag to switch the source of training data. 