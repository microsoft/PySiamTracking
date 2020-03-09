# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0.


from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from .center_dist_assigner import CenterDistAssigner
from .max_iou_assigner import MaxIoUAssigner
from .fovea_point_assigner import FoveaPointAssigner

__all__ = ['BaseAssigner', 'AssignResult', 'CenterDistAssigner', 'MaxIoUAssigner', 'FoveaPointAssigner']
