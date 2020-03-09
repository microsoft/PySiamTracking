# Inference with PySiamTracking

## Quick Start: Tracking on Demo Video

We provide a demo script for a quick start of PySiamTracking framework. One can simply run

```bash
python tools/demo.py \
--config experiments/model_configs/spm/spm_alexnet.py \
--checkpoint output/spm/spm_alexnet/epoch_50.pth
```

 The `--config` term is the path of the model configuration file, typically saved in `experiments/model_configs/`. The `--checkpoint` term denotes the path of model weights file.  One can add `--cpu` term to switch to CPU-only mode.

By default, the tracker will track the boy shown in **data/demo_video/boy.avi**. The output will be saved to **data/demo_video/demo_out.mp4**. 



## Testing on Benchmarks

We provide scripts to evaluate the performances on several mainstream benchmarks including **OTB, VOT, LaSOT, TrackingNet, GOT10K**.  One can selectively download and evaluate on the necessary benchmarks. 

### Data preparation 

**OTB**

```bash
python scripts/download_otb.py --root_dir <benchmark_root>/otb
```

By default, the data will be saved to `data/benchmark/otb/`

**VOT16/17**

```bash
python scripts/download_vot.py --root_dir <benchmark_root>/vot17 --year 2017
python scripts/download_vot.py --root_dir <benchmark_root>/vot16 --year 2016
```

**LaSOT**

Please download the dataset from official website: https://cis.temple.edu/lasot/. The dataset should be saved (or linked) to `<benchmark_root>/LaSOT/`

**TrackingNet**

Please follow the official instruction in https://github.com/SilvioGiancola/TrackingNet-devkit. The dataset should be saved (or linked) to `<benchmark_root>/TrackingNet/TEST/`

**GOT10K**

Please download from official website: http://got-10k.aitestunion.com/. The dataset should be saved (or linked) to `<benchmark_root>/GOT10K/test/` and `<benchmark_root>/GOT10k/val/`. (Test set and validation set).

### Running tracker

One can use `tools/test_tracker.py` entry to run a tracker on the target benchmark. The following code shows an example to test *SPM* on *OTB100* and *VOT17* benchmarks. One can change the model configuration file by replacing `--config` field and `--checkpoint` field.

```bash
python tool/test_tracker.py \
--benchmark OTB100,VOT17 \
--config experiments/model_configs/spm/spm_alexnet.py \
--checkpoint output/spm/spm_alexnet/epoch_50.pth
```

By default, we assume the benchmark data is saved in `<code_root>/data/benchmark`. One can specify the new benchmark root folder by setting `--data_dir` key.

### Hyper-parameter searching

Tracking is sensitive to some hyper-parameter settings, especially on VOT benchmarks. Generally, we perform hyper-parameter searching following SiamRPN series. One can define the hyper-parameter search space in a configuration file and pass this configuration file by `--eval_config` field.

Here shows an example to search the hyper-parameters on OTB100 benchmark:

```python
# example can be found in experiments/eval_configs/otb100.py
eval_cfgs = [
    dict(
        metrics=dict(type='OPE'),  # evaluation metrics
        dataset=dict(type='OTB100'),  # dataset name
        hypers=dict(
            epoch=list(range(31, 51, 2)),  # checkpoints, [31, 33, 35, ..., 49]
            window=dict(
                weight=[0.200, 0.300, 0.400]  # cosine window weights.
            )
        )
    ),
]
```

```bash
python tools/test_tracker.py \
--eval_config experiments/eval_configs/otb100.py \
--config experiments/model_configs/spm/spm_alexnet.py \
--checkpoint output/spm/spm_alexnet/ \
--hypersearch
```

### Evaluate results

After running tracker on a dataset, a report file named `report_<date>.csv` will be generated in the output directory. One can also perform evaluations with respect to the output results. Here shows an example:

```bash
python tools/evaluate_results.py \
--result output/spm/spm_alexnet/test_otb100/epoch_50/otb100_ope_results.pkl 
```

We can also evaluate multiple results at the same time and present the top-performing results by given metri

```bash
python tools/evaluate_results.py \
--result "output/spm/spm_alexnet/test_otb100/*/otb100_ope_results.pkl" \
--sort_key auc_overlap \
--topk 10
```

The output should be something like:

```
+----+-----------------------------------------------+---------------+-------------------+------------------------+---------+-------------+
|    | name                                          |   auc_overlap |   precision_error |   precision_error_norm |     fps |   fps_wo_io |
|----+-----------------------------------------------+---------------+-------------------+------------------------+---------+-------------|
| 10 | 43-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.694319 |          0.892972 |               0.849831 | 22.1761 |     32.57   |
| 21 | 37-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.689419 |          0.887738 |               0.840778 | 22.2867 |     32.6228 |
| 13 | 33-proposal.nms_iou_thr@0.6-window.weight@0.3 |      0.685654 |          0.878818 |               0.837484 | 22.1693 |     33.0386 |
| 27 | 39-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.684798 |          0.882966 |               0.83677  | 22.4161 |     32.8629 |
| 14 | 43-proposal.nms_iou_thr@0.6-window.weight@0.3 |      0.684506 |          0.878141 |               0.836759 | 22.5037 |     33.0396 |
|  6 | 45-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.684401 |          0.878464 |               0.834788 | 22.1862 |     33.1083 |
|  3 | 37-proposal.nms_iou_thr@0.6-window.weight@0.3 |      0.683597 |          0.878171 |               0.831748 | 21.9199 |     32.6552 |
| 28 | 35-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.683395 |          0.877711 |               0.834258 | 22.4411 |     32.9338 |
|  1 | 49-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.681041 |          0.872172 |               0.827265 | 22.4332 |     32.7945 |
| 22 | 47-proposal.nms_iou_thr@0.6-window.weight@0.4 |      0.680768 |          0.870913 |               0.827921 | 22.573  |     33.4423 |
+----+-----------------------------------------------+---------------+-------------------+------------------------+---------+-------------+
```
