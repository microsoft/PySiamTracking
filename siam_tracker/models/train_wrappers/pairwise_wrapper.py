# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from collections import OrderedDict

from ..builder import build_tracker, TRAIN_WRAPPERS
from ...datasets import TrainPairDataset, build_dataloader
from ...runner import Runner
from ...utils.parallel import MMDataParallel
from ...utils import load_checkpoint


@TRAIN_WRAPPERS.register_module
class PairwiseWrapper(object):

    def __init__(self,
                 train_cfg,
                 model_cfg,
                 work_dir,
                 log_level,
                 resume_from=None,
                 gpus=1):
        """ Training a tracker by image pairs. This is the most common strategy to train a
        siamese-network-based tracker. Generally, two images are randomly sampled from the
        dataset, one for template image (z_img) and another for search region (x_img). The
        tracker model needs to locate the target object in search region.
        """

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        # Step 1, build the tracker model.
        model = build_tracker(model_cfg, is_training=True, train_cfg=train_cfg, test_cfg=None)
        if resume_from is not None:
            load_checkpoint(model, resume_from)
        model = MMDataParallel(model, device_ids=list(range(gpus))).cuda()

        # Step 2, build image-pair datasets
        train_dataset = TrainPairDataset(train_cfg.train_data)
        self.data_loaders = build_dataloader(train_dataset,
                                             train_cfg.samples_per_gpu,
                                             train_cfg.workers_per_gpu,
                                             num_gpus=gpus)

        # Step 3, build a training runner
        # build runner
        self.runner = Runner(model, self.batch_processor, train_cfg.optimizer, work_dir, log_level)
        self.runner.register_training_hooks(train_cfg.lr_config, train_cfg.optimizer_config,
                                       train_cfg.checkpoint_config, train_cfg.log_config)
        if 'status_config' in train_cfg and train_cfg['status_config'] is not None:
            self.runner.register_status_hook(train_cfg['status_config'])

    def run(self):
        self.runner.run(self.data_loaders,
                        self.train_cfg.workflow,
                        self.train_cfg.total_epochs)

    @staticmethod
    def batch_processor(model, data, train_mode):
        losses = model(**data)
        loss, log_vars = PairwiseWrapper.parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['z_imgs'].data))

        return outputs

    @staticmethod
    def parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars
